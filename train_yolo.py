<<<<<<< HEAD
from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8s.pt")  # atau yolov8s.pt, bebas
    model.train(
        data="New_Dataset/data.yaml",
        epochs=200,
        imgsz=768,  # kamu bisa ubah jadi 960, 1280 dll
        batch=8,
        name="ktp_detect",
        workers=2,
        exist_ok=True
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model()
=======
from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8s.pt")  # atau yolov8s.pt, bebas
    model.train(
        data="New_Dataset/data.yaml",
        epochs=200,
        imgsz=768,  # kamu bisa ubah jadi 960, 1280 dll
        batch=8,
        name="ktp_detect",
        workers=2,
        exist_ok=True
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_model()
>>>>>>> 9dfa71189359fe7233ac78ccab9f4dad47c71b5c
