# 🐶 Dog Breed Prediction (OpenCV)

This project is an **OpenCV-based dog breed classification application** that predicts the breed of a dog from an uploaded image.  
It combines **image processing with OpenCV**, **deep learning with a pre-trained model**, and a **web interface using Streamlit**.

---

## 🚀 Features
- Upload a dog photo via a web interface
- Predict the dog breed using a pre-trained deep learning model
- Display the most likely breed with confidence score
- Show alternative predictions
- Keep a history of uploaded images and predictions

---

## 🧠 Technologies Used
- **Python**
- **OpenCV (cv2)** – image processing
- **TensorFlow / Keras** – deep learning model (MobileNetV2)
- **NumPy** – numerical operations
- **Streamlit** – web application interface

---

## ⚙️ How It Works
1. The user uploads a dog image through the web interface.
2. The image is processed using OpenCV:
   - Read image
   - Resize to 224x224
   - Convert color format (BGR → RGB)
3. The processed image is passed to a **pre-trained MobileNetV2 model**.
4. The model predicts the most likely dog breed.
5. Results are displayed on the web page along with alternative predictions.

---

## ⚙️ How It Works

1. The user uploads a dog image through the web interface.
2. The image is processed using OpenCV:
   - Read image
   - Resize to 224x224
   - Convert color format (BGR → RGB)
3. The processed image is passed to a **pre-trained MobileNetV2 model**.
4. The model predicts the most likely dog breed.
5. Results are displayed on the web page along with alternative predictions.

---

## ▶️ Run Streamlit App

```bash
# Start the Streamlit web application
streamlit run app.py

# The application will open in your browser at:
# http://localhost:8501

