# 🌉 Fremont Bridge Bicycle Traffic Prediction (LSTM)

Bu proje, Seattle'daki **Fremont Köprüsü** üzerindeki saatlik bisiklet trafiğini tahmin etmek için **Derin Öğrenme (Deep Learning)** tekniklerini kullanan bir zaman serisi analiz çalışmasıdır.

## 🎯 Projenin Amacı
Projenin temel amacı, geçmiş trafik yoğunluğu verilerini, takvim etkilerini (hafta içi/hafta sonu) ve mevsimsel döngüleri analiz ederek önümüzdeki saatlerde köprüden kaç bisiklet geçeceğini tahmin etmektir. Bu tür modeller, şehir planlaması ve trafik yönetimi için kritik öneme sahiptir.

## 🧠 Teknik Mimari
Projede zaman serisi verilerindeki uzun vadeli bağımlılıkları öğrenebilen **LSTM (Long Short-Term Memory)** mimarisi kullanılmıştır.



- **Model Yapısı:** - Giriş katmanı (30 saatlik geçmiş veri penceresi).
    - LSTM katmanları (Ardışık verilerdeki desenleri yakalamak için).
    - Dense (Tam bağlantılı) çıkış katmanı.
- **Veri Seti Özellikleri:**
    - Saatlik geçiş sayıları (Hedef değişken).
    - Takvim verileri (Haftanın günü, ay, saat).
    - Hafta sonu/İş günü ayrımı.

## 📊 Kullanılan Teknolojiler
- **TensorFlow/Keras:** Derin öğrenme modelinin inşası için.
- **Scikit-Learn:** Verilerin ölçeklendirilmesi (MinMaxScaler) için.
- **Pandas & Numpy:** Veri manipülasyonu ve matris işlemleri.
- **Streamlit:** Tahminlerin web üzerinden anlık olarak izlenebildiği kullanıcı arayüzü.
- **Joblib:** Modelin ve ölçekleyicinin (scaler) paketlenmesi için.

## 🚀 Kurulum ve Çalıştırma

### 1. Kütüphaneleri Yükleyin
Gerekli tüm bağımlılıkları yüklemek için terminale şu komutu yazın:
```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlatın
Streamlit arayüzünü açmak ve tahminleri görmek için:
```bash
streamlit run app.py
```

## 📈 Uygulama Arayüzü Özellikleri
- **Dinamik Tahmin Aralığı:** Kaydırma çubuğu (slider) yardımıyla 1 saatten 48 saate kadar gelecek tahmini üretebilirsiniz.
- **İnteraktif Grafik:** Tahmin edilen bisiklet trafiği akışını anlık olarak çizgi grafik üzerinde görebilirsiniz.
- **Yapay Zeka Belleği:** Model, son 30 saatlik gerçek veriyi "bellek" (pencere) olarak kullanır ve her tahminden sonra penceresini kaydırarak ilerler.

## 📁 Dosya Yapısı
- `fremont-bridge.ipynb`: Veri ön işleme, görselleştirme ve LSTM modelinin eğitildiği ana dosya.
- `app.py`: Web arayüzü ve modelin canlı tahmin yürütme kodu.
- `fremont_final_paket.joblib`: Eğitilmiş model ağırlıkları, konfigürasyonu ve scaler nesnesini içeren paket.
- `requirements.txt`: Projenin çalışması için gereken yazılım kütüphaneleri.