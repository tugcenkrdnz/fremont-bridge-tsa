import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import Sequential

st.title("🌉 Fremont Bridge AI")

# Veriyi ve Modeli Yükle
bundle = joblib.load('fremont_final_paket.joblib')
sc, window = bundle['scaler'], bundle['last_window']
win_size = bundle.get('window_size', 30) 

@st.cache_resource
def get_model():
    m = Sequential.from_config(bundle['model_config'])
    m.set_weights(bundle['model_weights'])
    return m

model = get_model()
hours = st.slider("Tahmin Edilecek Saat", 1, 48, 24)

if st.button("Tahmin Et"):
    curr, preds = window.copy(), []
    
    for _ in range(hours):
        # Tahmin yap ve listeye ekle
        p = model.predict(curr.reshape(1, win_size, 4), verbose=0)[0,0]
        preds.append(p)
        # Pencereyi kaydır (Sliding Window): En eskiyi sil, yeni tahmini ve diğer sütunları ekle
        new_row = np.append([p], curr[-1, 1:]) 
        curr = np.vstack([curr[1:], new_row])

    # Ters Ölçekleme (Inverse Transform)
    dummy = np.zeros((len(preds), 4))
    dummy[:, 0] = preds
    res = sc.inverse_transform(dummy)[:, 0].clip(0).astype(int) # Negatif değerleri engellemek için clip(0)

    st.line_chart(res)
    st.write(f"Önümüzdeki {hours} saat için tahminler:", res)
