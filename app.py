import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from PIL import Image
from transformers import pipeline

# -----------------------------
# Sayfa ayarları
# -----------------------------
st.set_page_config(
    page_title="Köpek Cinsi Tahmin Uygulaması",
    page_icon="🐶",
    layout="centered"
)

# -----------------------------
# Model
# -----------------------------
MODEL_ID = "amaye15/google-vit-base-patch16-224-batch64-lr0.005-standford-dogs"

@st.cache_resource
def modeli_yukle():
    return pipeline("image-classification", model=MODEL_ID)

# -----------------------------
# Başlık
# -----------------------------
st.title("🐶 Köpek Cinsi Tahmin Uygulaması")
st.write("Bir köpek fotoğrafı yükle ve modelin tahmin ettiği cinsi gör.")
st.caption("Model: Stanford Dogs veri seti (120 köpek ırkı) ile eğitilmiş hazır ViT modeli.")
st.divider()

# -----------------------------
# Session State (geçmiş)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Fotoğraf yükleme
# -----------------------------
uploaded = st.file_uploader(
    "📤 Köpek fotoğrafı yükle (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    st.subheader("📷 Yüklenen Fotoğraf")
    st.image(img, use_container_width=True)

    with st.spinner("Model tahmin yapıyor..."):
        clf = modeli_yukle()
        preds = clf(img, top_k=5)

    # En iyi tahmin
    best = preds[0]
    best_label = best["label"].replace("_", " ")
    best_score = best["score"] * 100

    st.success(f"✅ En yüksek tahmin: **{best_label}** — **%{best_score:.2f}**")

    # Top-5 listesi
    st.subheader("📌 Top-5 Tahmin Sonuçları")
    for i, p in enumerate(preds, start=1):
        label = p["label"].replace("_", " ")
        score = p["score"] * 100
        st.write(f"{i}) **{label}** — %{score:.2f}")

    # Geçmişe ekle
    st.session_state.history.append({
        "image": img,
        "label": best_label,
        "score": best_score
    })

# -----------------------------
# Önceki yüklenenler
# -----------------------------
if st.session_state.history:
    st.divider()
    st.subheader("📂 Önceki Yüklenen Köpekler")

    for item in reversed(st.session_state.history[:-1]):
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(item["image"], width=120)

        with col2:
            st.write(f"**{item['label']}**")
            st.write(f"Tahmin Olasılığı: %{item['score']:.2f}")

st.divider()
st.caption(
    "Not: Bu uygulama eğitim yapmaz. Hazır (pre-trained) bir derin öğrenme modeli kullanarak tahmin üretir."
)

# streamlit run app.py
