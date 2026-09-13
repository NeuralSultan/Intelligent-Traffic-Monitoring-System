from pathlib import Path


PROJECT_ROOT = Path(
    r"D:\Intelligent Traffic Monitoring & Violation Detection System"
)


# ============================================================
# Model
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "detection"
    / "final_yolo11n.pt"
)


# ============================================================
# Road configuration
# ============================================================

ROAD_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "road_config.json"
)


# ============================================================
# Detection / Tracking
# ============================================================

CONFIDENCE = 0.35

IMAGE_SIZE = 640

TRACKER = "bytetrack.yaml"


# ============================================================
# Vehicle classes
# ============================================================

VEHICLE_CLASSES = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "motorcycle",
    4: "bicycle",
}