from pathlib import Path
import cv2
from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Intelligent Traffic Monitoring & Violation Detection System"
)

MODEL_PATH = PROJECT_ROOT / "models" / "detection" / "final_yolo11n.pt"

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "test_videos" / "traffic_test.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "tracked_videos"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load model
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# Open video
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# Output video
# ============================================================

output_path = OUTPUT_DIR / "tracked_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video: {output_path}"
    )


# ============================================================
# Tracking loop
# ============================================================

frame_count = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        iou=0.50,
        imgsz=640,
        verbose=False,
    )

    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    cv2.imshow(
        "Intelligent Traffic Monitoring - ByteTrack",
        annotated_frame,
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()


print("\nTracking completed.")
print(f"Frames processed: {frame_count}")
print(f"Output saved to: {output_path}")