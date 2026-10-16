# PSYKE — Yüz Psikoloji Analiz Sistemi

## Mimari
```
Webcam → OpenCV → DeepFace (cinsiyet, yaş, duygu) → Gemini API (psikoloji yorumu) → Web UI
```

## Kurulum

### 1. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 2. API anahtarını ayarla (Google Gemini)
```bash
export GOOGLE_API_KEY="AIzaSy..."
```

### 3. Çalıştır
```bash
cd face-psychology
uvicorn main:app --reload --port 8000
```

### 4. Tarayıcıda aç
```
http://localhost:8000
```

---

## Nasıl Çalışır?

### Adım 1 — DeepFace (Lokal CNN)
- **Cinsiyet tespiti**: VGG-Face modeli
- **Yaş tahmini**: DEX modeli  
- **Duygu analizi**: 7 kategori (mutlu, üzgün, kızgın, korku, şaşkın, iğrenmiş, nötr)
- **Irk analizi**: 6 kategori
- Tamamen **ücretsiz**, **lokal**, internet gerektirmez

### Adım 2 — Gemini API (Psikoloji Yorumu)
- DeepFace çıktılarını alır
- Derin psikolojik yorum üretir:
  - Genel durum analizi
  - Duygusal durum yorumu
  - Kişilik izlenimleri
  - Stres seviyesi
  - Sosyal eğilim (içe/dışa dönük)
  - Yaş psikolojisi
  - Kişisel tavsiyeler
  - Gizem / Empati / Enerji skorları

---

## Proje Yapısı
```
face-psychology/
├── main.py           # FastAPI backend
├── templates/
│   └── index.html    # Web UI
├── requirements.txt
└── README.md
```

---

## İleri Seviye Geliştirmeler
- [ ] Gerçek zamanlı video akışı (WebSocket)
- [ ] Çoklu yüz tespiti
- [ ] Zaman içinde psikoloji değişimi grafiği
- [ ] Psikoloji profili kaydetme/export
- [ ] Özel psikoloji modeli eğitimi (kendi verin ile fine-tune)
