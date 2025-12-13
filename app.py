import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

st.set_page_config(page_title="Dog Breed Predictor", layout="centered")

st.title("🐶 Dog Breed Prediction")
st.write("Upload a dog photo and see the predicted breed.")

# Geçmiş (yüklenen köpekler)
if "history" not in st.session_state:
    st.session_state.history = []

# Modeli yükle (bir kere)
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# Foto yükleme alanı
uploaded_file = st.file_uploader(
    "Köpek fotoğrafı yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Görseli oku
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Görsel okunamadı.")
        st.stop()

    # Göster
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Yüklenen foto")

    # Modele hazırla
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    x = preprocess_input(x)

    # Tahmin
    preds = model.predict(x)
    top3 = decode_predictions(preds, top=3)[0]

    # Ana sonuç
    best_label, best_prob = top3[0][1], top3[0][2]
    st.success(f"En yüksek tahmin: **{best_label} (%{best_prob*100:.2f})**")

    # Geçmişe ekle
    st.session_state.history.append({
    "image": img_bgr,
    "label": best_label,
    "prob": best_prob
    })

    # Diğer olasılıklar
    st.subheader("Diğer olasılıklar")
    for i in range(1, 3):
        st.write(f"{i+1}) {top3[i][1]} — %{top3[i][2]*100:.2f}")
    
    # -------------------------------
    # Yüklenen köpekler listesi
    # -------------------------------
    if st.session_state.history:
        st.divider()
        st.subheader("📂 Yüklenen Köpekler")

    for item in st.session_state.history[::-1]:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(
                cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB),
                width=120
            )

        with col2:
            st.write(f"**{item['label']}**")
            st.write(f"Olasılık: %{item['prob']*100:.2f}")

