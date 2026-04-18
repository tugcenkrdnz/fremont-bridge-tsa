import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="Fremont Bridge AI", layout="wide")
st.title("🌉 Fremont Köprüsü Bisiklet Trafiği Tahmini")

# 1. Veri ve Modeli Yükle
bundle = joblib.load('fremont_final_paket.joblib')
sc, window = bundle['scaler'], bundle['last_window']
win_size = bundle.get('window_size', 30)

@st.cache_resource
def get_model():
    m = Sequential.from_config(bundle['model_config'])
    m.set_weights(bundle['model_weights'])
    return m

model = get_model()

# 2. Kullanıcı Girişi
st.sidebar.header("Tahmin Ayarları")
hours_to_predict = st.sidebar.slider("Kaç saat sonrasını görelim?", 1, 48, 24)
# Başlangıç saati (Gerçekçi görünmesi için şu anki saati alıyoruz)
start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

if st.button("Trafiği Tahmin Et"):
    curr, preds, time_labels = window.copy(), [], []
    
    for i in range(hours_to_predict):
        # Tahmin yap
        p = model.predict(curr.reshape(1, win_size, 4), verbose=0)[0,0]
        preds.append(p)
        
        # Saat etiketini oluştur
        next_hour = start_time + timedelta(hours=i+1)
        time_labels.append(next_hour.strftime("%H:%M"))
        
        # Pencereyi kaydır (Sliding Window)
        new_row = np.append([p], curr[-1, 1:]) 
        curr = np.vstack([curr[1:], new_row])

    # 3. Sonuçları Çevir ve Görselleştir
    dummy = np.zeros((len(preds), 4))
    dummy[:, 0] = preds
    res = sc.inverse_transform(dummy)[:, 0].clip(0).astype(int)

    # DataFrame oluştur (Grafikte saatlerin görünmesi için şart)
    chart_data = pd.DataFrame({
        "Saat": time_labels,
        "Tahmin Edilen Bisiklet Sayısı": res
    }).set_index("Saat")

    # Grafiği ve Tabloyu Yan Yana Göster
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Tahmin Grafiği")
        st.line_chart(chart_data)
        
    with col2:
        st.subheader("📋 Liste")
        st.write(chart_data)

    st.success(f"Uygulama {start_time.strftime('%H:%M')} itibariyle tahmine başlamıştır.")
