from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import base64
import json
import re
import os
import io

app = FastAPI(title="Face Psychology API")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# .env dosyasından veya ortam değişkeninden API anahtarını yükle
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as _f:
        for _line in _f:
            if _line.strip().startswith("GOOGLE_API_KEY="):
                os.environ.setdefault("GOOGLE_API_KEY", _line.strip().split("=", 1)[1].strip('"\''))
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

class ImagePayload(BaseModel):
    image_b64: str  # base64 encoded image (without data URI prefix)

def analyze_face_with_deepface(image_b64: str) -> dict:
    """Analyze face using DeepFace - returns gender, age, emotion, race"""
    try:
        import deepface
        from deepface import DeepFace
        import numpy as np
        import cv2

        # Decode base64 to numpy array
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        # Run DeepFace analysis
        results = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            silent=True
        )

        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # Extract dominant values
        emotion_scores = result.get('emotion', {})
        race_scores = result.get('race', {})

        analysis = {
            "age": max(1, int(result.get('age', 0)) - 10),
            "gender": result.get('dominant_gender', result.get('gender', 'Unknown')),
            "gender_confidence": 0.0,
            "dominant_emotion": result.get('dominant_emotion', 'neutral'),
            "emotion_scores": {k: round(float(v), 1) for k, v in emotion_scores.items()},
            "dominant_race": result.get('dominant_race', 'unknown'),
            "race_scores": {k: round(float(v), 1) for k, v in race_scores.items()},
            "face_detected": True
        }

        # Handle gender format differences
        gender_data = result.get('gender', {})
        if isinstance(gender_data, dict):
            man_score = gender_data.get('Man', 0)
            woman_score = gender_data.get('Woman', 0)
            if man_score > woman_score:
                analysis['gender'] = 'Man'
                analysis['gender_confidence'] = round(man_score, 1)
            else:
                analysis['gender'] = 'Woman'
                analysis['gender_confidence'] = round(woman_score, 1)
        else:
            analysis['gender_confidence'] = 85.0  # Default confidence

        return analysis

    except ImportError:
        raise HTTPException(status_code=500, detail="DeepFace not installed. Run: pip install deepface")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face analysis error: {str(e)}")


def get_psychology_from_gemini(face_data: dict, image_b64: str) -> dict:
    """Send face analysis data AND the actual image to Gemini for multimodal interpretation"""

    # Image processing for Gemini
    img_data = base64.b64decode(image_b64)
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_data))

    gender_tr = "Erkek" if face_data['gender'] in ['Man', 'man'] else "Kadın"
    emotion_tr_map = {
        'happy': 'Mutlu', 'sad': 'Üzgün', 'angry': 'Kızgın',
        'fear': 'Korkmuş', 'surprise': 'Şaşkın', 'disgust': 'İğrenmiş',
        'neutral': 'Nötr'
    }
    dominant_emotion_tr = emotion_tr_map.get(face_data['dominant_emotion'], face_data['dominant_emotion'])

    prompt = f"""Sen, dünyanın en saygın ve gelişmiş kriminal yüz analiz uzmanı, adli psikolog ve davranış bilimcisisin (Paul Ekman seviyesinde bir mikro ifade ve fizyognomi dehası).
Sana iletilen fotoğrafı ve DeepFace analiz verilerini BİREBİR KOPYALAMADAN, ezbere ve klişe cümleler KULLANMADAN, son derece özgün, derin, spesifik ve klinik bir keskinlikle yorumlaman gerekiyor. Her seferinde aynı veya benzer genel geçer yorumları yapmaktan KESİNLİKLE kaçın.

TEKNİK VERİLER (DeepFace):
- Cinsiyet: {gender_tr} (DİKKAT: DeepFace başörtüsü veya kısa saç gibi durumlarda cinsiyeti yanlış tahmin edebilir. Lütfen GÖRSELİ analiz ederek asıl cinsiyeti belirle)
- Tahmini Yaş: {face_data['age']}
- Tespit Edilen Baskın Duygu: {dominant_emotion_tr}

GÖZLEM VE DERİN ANALİZ REHBERİN (Şunları tek tek, büyüteçle inceliyormuş gibi analiz et):
1. ASİMETRİ VE MİKRO-İFADELER: Yüzün sağ ve sol yarısı arasındaki uyuşmazlıklar (ör. dudakların sadece bir tarafının kalkması, tek kaşın hafif kalkıklığı) gizli veya bastırılmış duyguları ele verir. İğne ucu kadar küçük kas gerilimlerini (göz çevresindeki kaz ayakları, alın kırışıklıkları, çene sıkılması) detaylıca incele.
2. GÖZLERİN DİLİ: Göz bebeklerinin boyutu, bakışın odak noktası (boşluğa mı bakıyor, hedefe mi kilitlenmiş?), göz kapaklarının ağırlığı (yorgunluk veya bıkkınlık) ve gözaltı morlukları/torbaları kişinin ruhsal enerjisini ve geçmiş travmalarını/streslerini nasıl yansıtıyor?
3. FİZYOGNOMİ VE KEMİK YAPISI: Alın genişliği, elmacık kemiklerinin çıkıklığı, burun yapısı ve çene hattının (jawline) sivriliği/kareliği üzerinden baskın karakter özellikleri (ör. otoriter, uyumlu, inatçı, kırılgan) çıkarımlarında bulun. Cilt dokusu ve elastikiyeti, kişinin genel yaşam kalitesi ve biyolojik yorgunluğu hakkında ne söylüyor?
4. KAMERA VE DURUŞ (PROKSEMİK): Başın eğim açısı (yukarıdan mı bakıyor, boynu mu bükük?), kameraya olan mesafe ve ışığın vuruş açısı kişinin şu anki psikolojik zırhı veya savunmasızlığı hakkında hangi ipuçlarını veriyor?

KRİTİK UYARI: 
- ASLA sürekli "gözlemlenebilir verilere dayanarak" veya "olasıdır" kelimelerini her cümlede tekrarlama. Dilin akıcı, profesyonel ama edebi bir psikolojik profil raporu gibi olsun. 
- Her analiz birbirinden TAMAMEN FARKLI ve KİŞİYE ÖZEL olmalı. Rastgele bir şablon doldurma. Kişinin fotoğrafındaki en ufak bir anomaliliği (bir ben, bir yara izi, saçın dağınıklığı, giysinin yakası vb. görünüyorsa) bile psikolojik duruma bağla.
- Teşhis koyma ancak cesur ve çarpıcı hipotezler üret. 

Aşağıdaki JSON formatında (her anahtar için en az 3-4 cümlelik destansı ve derinlemesine bir analiz sun), SADECE GEÇERLİ JSON OLARAK YANIT VER. Markdown, açıklama, fazladan yazı yazma.

{{
  "gercek_cinsiyet": "Kadın veya Erkek (Fotoğrafı dikkatlice incele. Başörtüsü, şapka vb. yüzünden DeepFace yanılmış olabilir. Gerçek cinsiyeti buraya yaz.)",
  "genel_durum": "Sadece bu kişiye özgü olan yüz ifadesinin, genel aurayı, ışığı ve duruşu nasıl etkilediğine dair klinik bir özet.",
  "duygusal_durum": {{
    "baskin_his": "Ör: 'Bastırılmış Öfke ve Kabullenilmiş Melankoli'",
    "yorum": "Yüzün belli bölgelerindeki milimetrik kas hareketlerinden tespit ettiğin çelişkili duyguların detaylı okuması."
  }},
  "kisilik_izlenimleri": [
    "Fiziksel spesifik bir detaya (ör. göz yapısı) dayandırılmış en belirgin 1. derin karakter analizi",
    "Fiziksel spesifik bir detaya (ör. çene/dudak) dayandırılmış 2. derin karakter analizi",
    "Genel asimetri veya ifadeye dayandırılmış 3. derin karakter analizi"
  ],
  "stres_seviyesi": {{
    "seviye": "Düşük/Orta/Yüksek/Kritik",
    "skor": 45,
    "aciklama": "Stresin vücutta ve yüzde nasıl somutlaştığının analizi (örneğin temporal kaslardaki gerginlik)."
  }},
  "sosyal_egitim": {{
    "tip": "İzole İçe Dönük / Gözlemci / Agresif Dışa Dönük vb.",
    "aciklama": "Göz temasının doğası ve başın duruş açısından sosyal iletişim stratejisinin analizi."
  }},
  "yas_psikolojisi": "Biyolojik yaş ({face_data['age']}) ile ruhsal yaşın karşılaştırması ve yüzde biriken yaşam tecrübesinin ağırlığı.",
  "tavsiyeler": [
    "Kişinin bilinçaltı gerginliklerini çözmesine yönelik spesifik bir fiziksel/psikolojik tavsiye.",
    "Baskın duygu durumunu dengelemek için radikal ve kişiye özel bir hayat tavsiyesi.",
    "Potansiyelini veya gizli gücünü açığa çıkarması için odaklanması gereken alan."
  ],
  "gizem_skoru": 72,
  "empati_skoru": 68,
  "enerji_skoru": 55
}}"""

    # Send both prompt and image to Gemini
    response = model.generate_content([prompt, img])
    response_text = response.text.strip()

    # Clean up response
    response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
    response_text = re.sub(r'\s*```$', '', response_text)
    response_text = response_text.strip()

    psychology = json.loads(response_text)
    return psychology


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
async def analyze(payload: ImagePayload):
    """Main analysis endpoint"""
    try:
        # Adım 1: DeepFace analizi
        print("--> Adım 1: DeepFace analizi başlıyor...")
        face_data = analyze_face_with_deepface(payload.image_b64)
        print(f"--> Yüz tespiti başarılı: {face_data['dominant_emotion']}")

        # Adım 2: Gemini görsel ve teknik analize başlıyor...
        psychology = get_psychology_from_gemini(face_data, payload.image_b64)
        print("--> Analiz başarıyla tamamlandı!")

        # Başörtüsü vb. durumlarda DeepFace'in yanılma payını Gemini ile düzeltiyoruz
        if "gercek_cinsiyet" in psychology:
            gercek = psychology["gercek_cinsiyet"].lower()
            if "kadın" in gercek or "kadin" in gercek:
                face_data["gender"] = "Woman"
            elif "erkek" in gercek:
                face_data["gender"] = "Man"

        return {
            "face_data": face_data,
            "psychology": psychology
        }
    except Exception as e:
        print(f"!!! HATA OLUŞTU: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
