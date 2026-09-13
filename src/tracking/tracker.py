from pathlib import Path

from ultralytics import YOLO


class TrafficTracker:
    """
    Handles YOLO + ByteTrack tracking.

    The model is loaded once and tracking state is preserved
    across frames using persist=True.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        image_size: int = 640,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.tracker_config = tracker_config

    def track(self, frame):
        """
        Detect and track objects in the current frame.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        return results[0]