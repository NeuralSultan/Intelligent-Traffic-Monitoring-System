from ultralytics import YOLO


# =========================
# Configuration
# =========================

DATASET_YAML =r"D:\Intelligent Traffic Monitoring & Violation Detection System\data\processed\unified-traffic\data.yaml"
MODEL_NAME = "yolo11n.pt"

EPOCHS = 80
IMAGE_SIZE = 640
BATCH_SIZE = 16

PROJECT_DIR = (
    r"D:\Intelligent Traffic Monitoring & Violation Detection System"
    r"\outputs\training"
)

RUN_NAME = "final_yolo11n"
# =========================
# Load model
# =========================

model = YOLO(MODEL_NAME)


# =========================
# Train
# =========================

results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,

    project=PROJECT_DIR,
    name=RUN_NAME,

    pretrained=True,

    patience=20,

    workers=4,

    # Keep cache on disk instead of RAM
    cache=False,

    verbose=True,
)