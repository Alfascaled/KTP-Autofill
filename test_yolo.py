<<<<<<< HEAD
from ultralytics import YOLO
import cv2
import os

# Path ke model hasil training
model_path = 'runs/detect/ktp_detect/weights/best.pt'

# Path ke gambar KTP yang mau diprediksi
image_path = 'New_Dataset/valid/images'  # ganti sama file asli kamu

# Load model
model = YOLO(model_path)

# Prediksi
results = model.predict(
    source=image_path,
    conf=0.55,   
    iou = 0.45,     # confidence threshold
    save= False,        # simpan gambar hasil deteksi
    save_txt=True     # simpan hasil ke .txt juga
)

# Tampilin hasil di terminal
for result in results:
    print("===== HASIL DETEKSI =====")
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = box.conf[0].item()
        print(f"{label}: {conf:.2f}")
=======
from ultralytics import YOLO
import cv2
import os

# Path ke model hasil training
model_path = 'runs/detect/ktp_detect/weights/best.pt'

# Path ke gambar KTP yang mau diprediksi
image_path = 'New_Dataset/valid/images'  # ganti sama file asli kamu

# Load model
model = YOLO(model_path)

# Prediksi
results = model.predict(
    source=image_path,
    conf=0.55,   
    iou = 0.45,     # confidence threshold
    save= False,        # simpan gambar hasil deteksi
    save_txt=True     # simpan hasil ke .txt juga
)

# Tampilin hasil di terminal
for result in results:
    print("===== HASIL DETEKSI =====")
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = box.conf[0].item()
        print(f"{label}: {conf:.2f}")
>>>>>>> 9dfa71189359fe7233ac78ccab9f4dad47c71b5c
