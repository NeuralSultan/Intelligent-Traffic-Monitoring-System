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
# Evidence
# ============================================================

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evidence"
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


# ============================================================
# Speed estimation
# ============================================================

# Line A
SPEED_LINE_A = (
    (536, 500),
    (1423, 511),
)


# Line B
SPEED_LINE_B = (
    (44, 700),
    (1788, 664),
)


# Approximate real-world distance between
# the two speed lines.
#
# IMPORTANT:
# This is a project calibration assumption,
# not a legal/physical measurement.
SPEED_REFERENCE_DISTANCE_M = 20.0


# Speed limit used by the project.
# We will use this later when implementing
# the actual speeding violation detector.
SPEED_LIMIT_KMH = 60.0

# Road configuration
# Supported values:
# "one_way"
# "two_way"
ROAD_TYPE = "one_way"