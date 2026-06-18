import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Set judul halaman aplikasi web
st.set_page_config(page_title="Sistem Klasifikasi Hewan", layout="wide")
st.title("🐾 Sistem Klasifikasi Hewan: Mamalia vs Unggas")
st.subheader("Berbasis End-to-End Computer Vision & Deep Learning")
st.write("Tugas UAS - Pemrosesan Citra Digital & Kecerdasan Buatan")

# Memuat model AI yang sudah dilatih sebelumnya
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('UAS_Computer_Vision/models/model_mamalia_unggas.h5')

try:
    model = load_my_model()
    st.success("🧠 Model Kecerdasan Buatan (MobileNetV2) Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model AI. Pastikan file model sudah ada. Error: {e}")

# Komponen Upload Gambar
uploaded_file = st.file_uploader("Pilih dan Upload Foto Hewan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar yang diupload pengguna
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # -----------------------------------------------------------------
    # PIPELINE COMPUTER VISION (Sesuai Syarat Penilaian UAS)
    # -----------------------------------------------------------------
    # 1. Resize gambar ke input standar AI (224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 2. Image Enhancement: Noise Reduction (Gaussian Blur)
    img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    # 3. Image Enhancement: Histogram Equalization (CLAHE)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    # 4. Edge Detection: Canny Edge Detection
    img_edges = cv2.Canny(img_enhanced, threshold1=50, threshold2=150)
    
    # 5. Analisis Deep Learning (Prediksi)
    img_for_pred = img_resized / 255.0  # Normalisasi
    img_input = np.expand_dims(img_for_pred, axis=0) # Tambah dimensi batch
    prediction = model.predict(img_input)
    
    # Menentukan hasil berdasarkan indeks kelas (0: Mamalia, 1: Unggas)
    if prediction > 0.5:
        kelas_prediksi = "MAMALIA"
        tingkat_keyakinan = (1 - prediction[0][0]) * 100
    else:
        kelas_prediksi = "UNGGAS"
        tingkat_keyakinan = prediction[0][0] * 100

    # -----------------------------------------------------------------
    # MENAMPILKAN VISUALISASI SECARA BERURUTAN DI ANTARMUKA (UI)
    # -----------------------------------------------------------------
    st.write("---")
    st.header("🖼️ Hasil Pipeline Pemrosesan Citra (Computer Vision)")
    
    # Membuat susunan 4 kolom berjejer horizontal
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image(img_resized, caption="1. Citra Input (Resized)", use_container_width=True)
        
    with col2:
        st.image(img_blur, caption="2. Noise Reduction", use_container_width=True)
        
    with col3:
        st.image(img_enhanced, caption="3. Histogram Equalization", use_container_width=True)
        
    with col4:
        st.image(img_edges, caption="4. Canny Edge Detection", use_container_width=True)
        
    # Menampilkan Hasil Akhir Prediksi AI dengan Kotak Informasi Menonjol
    st.write("---")
    st.header("🤖 Hasil Analisis Kecerdasan Buatan (Deep Learning)")
    
    if kelas_prediksi == "MAMALIA":
        st.info(f"### Hasil Deteksi: **{kelas_prediksi}** \n\nTingkat Akurasi/Keyakinan Model: **{tingkat_keyakinan:.2f}%**")
    else:
        st.success(f"### Hasil Deteksi: **{kelas_prediksi}** \n\nTingkat Akurasi/Keyakinan Model: **{tingkat_keyakinan:.2f}%**")
