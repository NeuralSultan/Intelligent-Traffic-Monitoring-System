from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = (
    Path(r"D:\Intelligent Traffic Monitoring & Violation Detection System\models\detection\best.pt")
)

VIDEO_PATH = (
    Path(r"D:\Intelligent Traffic Monitoring & Violation Detection System\data\raw\test_videos\YTDown.com_YouTube_4K-Video-of-Highway-Traffic_Media_KBsqQez-O4w_001_1080p.mp4")
)

OUTPUT_DIR = Path("outputs/results")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


model = YOLO(str(MODEL_PATH))


results = model.predict(
    source=str(VIDEO_PATH),
    save=True,
    conf=0.35,
    imgsz=640,
    project=str(OUTPUT_DIR),
    name="detection_test",
)


print("\nDetection completed.")
print(f"Results saved to: {OUTPUT_DIR / 'detection_test'}")