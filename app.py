import streamlit as st
import easyocr
import torch
from PIL import Image
import numpy as np
import cv2
import os


from ultralytics import YOLO
# ========== Konstanta ==========
LABELS = ['jenisKelamin', 'kebangsaan', 'nama', 'nik', 'pekerjaan', 'tempatTanggalLahir']
LABELS_TITLE = {
    "nama": "Nama",
    "nik": "NIK",
    "jenisKelamin": "Jenis Kelamin",
    "kebangsaan": "Kebangsaan",
    "pekerjaan": "Pekerjaan",
    "tempatTanggalLahir": "Tempat & Tanggal Lahir"
}

# ========== Init ==========
st.set_page_config(page_title="KTP Autofill App", layout="wide")
reader = easyocr.Reader(['id'], gpu=True if torch.cuda.is_available() else False)
model = YOLO("runs/detect/ktp_detect/weights/best.pt")  # Ganti dengan model yang sudah kamu latih
os.makedirs('temp', exist_ok=True)

# ========== Fungsi Preprocessing ==========
def preprocess_image(uploaded_image):
    image = np.array(uploaded_image.convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
    _, threshed = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp_path = 'temp/preprocessed.png'
    cv2.imwrite(temp_path, threshed)
    return image, temp_path

# ========== Ekstraksi Field dari Model + OCR ==========
def extract_fields(image, results):
    LABELS = ['jenisKelamin', 'kebangsaan', 'nama', 'nik', 'pekerjaan', 'tempatTanggalLahir']
    
    detections = results[0]  # YOLOv8 ngembaliin list of Results
    boxes = detections.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    bboxes = boxes.xyxy.cpu().numpy()
    
    data = {}
    for cls_id, box in zip(class_ids, bboxes):
        label = LABELS[cls_id] if cls_id < len(LABELS) else f"class_{cls_id}"
        
        x1, y1, x2, y2 = map(int, box)
        field_img = image[y1:y2, x1:x2]

        # OCR disini
        text = reader.readtext(field_img, detail=0)
        data[label] = ' '.join(text)

    return data

# ========== Streamlit UI ==========
st.title("🪪 KTP Auto-Fill Form")
st.markdown("Upload gambar KTP, lalu form akan terisi otomatis pakai model deteksi + OCR.")

uploaded_file = st.file_uploader("📤 Upload Gambar KTP (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    with st.spinner("⏳ Sedang memproses..."):
        image = Image.open(uploaded_file)
        np_image, processed_path = preprocess_image(image)
        result = model(processed_path)
        data = extract_fields(np_image, result)

    st.image(image, caption="📸 Gambar Asli", use_column_width=True)
    st.subheader("📝 Form Data KTP (Otomatis Terisi)")

    # Tampilkan field berdasarkan LABELS
    for label in LABELS:
        st.text_input(LABELS_TITLE[label], value=data.get(label, ""))
        
    st.success("✅ Done! Kamu bisa cek atau koreksi hasilnya di atas.")
