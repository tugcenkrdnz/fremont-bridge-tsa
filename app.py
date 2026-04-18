import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import Sequential

# 1. Ayarlar ve Veriyi Yükle
st.title("🌉 Fremont Bridge AI")
bundle = joblib.load('fremont_final_paket.joblib')
sc, window = bundle['scaler'], bundle['last_window']

@st.cache_resource
def get_model():
    m = Sequential.from_config(bundle['model_config'])
    m.set_weights(bundle['model_weights'])
    return m

model = get_model()

# 2. Giriş ve Tahmin
hours = st.slider("Saat", 1, 48, 24)

if st.button("Tahmin Et"):
    curr = window.copy()
    preds = []
    
    for _ in range(hours):
        p = model.predict(curr.reshape(1, 30, 4), verbose=0)[0,0]
        preds.append(p)
        # Pencereyi kaydır: En eskiyi at, yeni tahmini ekle
        curr = np.append(curr[1:], [[p, *curr[-1, 1:]]], axis=0)

    # 3. Sonuçları Çevir ve Göster
    dummy = np.zeros((len(preds), 4))
    dummy[:, 0] = preds
    res = sc.inverse_transform(dummy)[:, 0].astype(int)

    st.line_chart(res)
    st.write("Tahminler:", res)