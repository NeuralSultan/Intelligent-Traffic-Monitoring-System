from pathlib import Path

from ultralytics import YOLO


class TrafficDetector:
    """
    Handles YOLO object detection.

    The model is loaded once and reused for every frame.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        image_size: int = 640,
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        """
        Run YOLO detection on one frame.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        return results[0]