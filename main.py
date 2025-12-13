import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# 1) Modeli yükle
print("Model yükleniyor...")
model = MobileNetV2(weights="imagenet")
print("Model hazır.")

# 2) Fotoğrafı oku
Tk().withdraw()

img_path = askopenfilename(
    title="Köpek fotoğrafını seç",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not img_path:
    print("Dosya seçilmedi. Program kapatıldı.")
    raise SystemExit

img_bgr = cv2.imread(img_path)


if img_bgr is None:
    print("HATA: dog.jpg bulunamadı veya okunamadı. Dosya adı/dizini kontrol et.")
    raise SystemExit

# 3) Modele uygun hale getir (224x224, RGB, preprocess)
img_resized = cv2.resize(img_bgr, (224, 224))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

x = np.expand_dims(img_rgb, axis=0).astype(np.float32)
x = preprocess_input(x)

# 4) Tahmin yap
preds = model.predict(x)

# 5) En iyi 3 sonucu yazdır
top3 = decode_predictions(preds, top=3)[0]
print("\nTahminler (Top 3):")
for i, (cls_id, label, prob) in enumerate(top3, start=1):
    print(f"{i}) {label}  -  %{prob*100:.2f}")

# 6) En yüksek tahmini al
best_label = top3[0][1]
best_prob = top3[0][2]
main_text = f"{best_label} (%{best_prob*100:.2f})"

# 7) Top-3 metnini hazırla (detay liste)
lines = [
    f"2) {top3[1][1]}  %{top3[1][2]*100:.2f}",
    f"3) {top3[2][1]}  %{top3[2][2]*100:.2f}",
]

# 8) Fotoğraf üstüne yaz
output_img = img_bgr.copy()

# Ana başlık (sol üst)
cv2.putText(
    output_img, main_text, (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
)

# Detay liste (sol alt)
h, w = output_img.shape[:2]
start_y = h - 70  # alta yakın başla
for i, line in enumerate(lines):
    y = start_y + i * 25
    cv2.putText(
        output_img, line, (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

# 9) Göster
cv2.imshow("Dog Breed Prediction", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
