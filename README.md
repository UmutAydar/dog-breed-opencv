# 🐶 Köpek Cinsi Tahmin Uygulaması (OpenCV + Transformers + Streamlit)

Bu proje, **yüklediğiniz köpek fotoğrafına göre köpek cinsini tahmin eden** basit bir web uygulamasıdır.  
Uygulama **eğitim yapmaz**; hazır (pre-trained) bir modeli kullanarak tahmin üretir.

> ✅ Model: **Stanford Dogs (120 köpek ırkı)** üzerinde eğitilmiş hazır bir görüntü sınıflandırma modeli  
> ✅ Arayüz: **Streamlit** (tarayıcı üzerinden çalışır)

---

## 🎯 Özellikler

- Köpek fotoğrafı yükleyerek **tahmin** alma
- **Top-5** tahmini yüzde oranlarıyla gösterme
- Daha önce yüklenen fotoğrafları **liste halinde** görme (geçmiş)
- Modelde bulunan tüm ırk listesini (`dog_breeds_labels.txt`) görüntüleme

---

## 🧰 Kullanılan Teknolojiler

- **Python**
- **Streamlit** (web arayüz)
- **Transformers** (hazır görüntü modeli)
- **PyTorch (torch)** (modelin çalışması için)
- **Pillow (PIL)** (görüntü okuma/dönüştürme)

---

## 🚀 Kurulum ve Çalıştırma 

Aşağıdaki adımları takip ederek projeyi kendi bilgisayarınızda çalıştırabilirsiniz.

### 1) Gereken Programlar
- **Python 3.10 / 3.11 önerilir**
- **Git** (opsiyonel ama önerilir)

> Not: Python 3.13 bazı kütüphanelerde uyumsuzluk çıkarabilir. En sorunsuz: **3.11**.

---

### 2) Projeyi İndir (Clone)
Terminal/PowerShell açın ve şunu çalıştırın:

```bash
git clone https://github.com/UmutAydar/dog-breed-opencv.git
cd dog-breed-opencv

### 3) Sanal Ortam (venv) Oluştur ve Aktif Et
Windows (PowerShell)

py -3.11 -m venv .venv
.\.venv\Scripts\Activate

Aktif olunca satır başında şunu görürsünüz:
(.venv)

Sonra tekrar:
.\.venv\Scripts\Activate

### 4) Gerekli Paketleri Kur

Sanal ortam aktifken:

pip install -r requirements.txt 


Eğer repoda requirements.txt yoksa şu komutla kurabilirsiniz:

pip install streamlit transformers torch pillow


### 5) Uygulamayı Başlat

streamlit run app.py

Tarayıcıda şu adreste açılır:

http://localhost:8501
