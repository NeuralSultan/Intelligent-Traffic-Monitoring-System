from pathlib import Path

from ultralytics import YOLO


class TrafficSignDetector:
    """
    Detects traffic signs using a dedicated YOLO model.

    Very small speed-limit signs are ignored because their
    classification can be unreliable.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        image_size: int = 640,
        min_speed_sign_width: int = 50,
        min_speed_sign_height: int = 50,
    ):
        self.model = YOLO(str(model_path))

        self.confidence = confidence
        self.image_size = image_size

        self.min_speed_sign_width = (
            min_speed_sign_width
        )

        self.min_speed_sign_height = (
            min_speed_sign_height
        )

        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        return results[0]

    def get_detections(self, frame):
        result = self.detect(frame)

        detections = []

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return detections

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        class_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for box, class_id, confidence in zip(
            boxes,
            class_ids,
            confidences,
        ):
            class_name = self.class_names[
                class_id
            ]

            x1, y1, x2, y2 = box

            width = float(x2 - x1)
            height = float(y2 - y1)

            # Ignore very small speed-limit signs.
            if class_name.startswith(
                "Speed Limit"
            ):
                if (
                    width
                    < self.min_speed_sign_width
                    or height
                    < self.min_speed_sign_height
                ):
                    continue

            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox": (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ),
                }
            )

        return detections

    def get_speed_limit_detections(
        self,
        frame,
    ):
        detections = self.get_detections(frame)

        return [
            detection
            for detection in detections
            if detection["class_name"].startswith(
                "Speed Limit"
            )
        ]

    def get_traffic_light_detections(
        self,
        frame,
    ):
        detections = self.get_detections(frame)

        return [
            detection
            for detection in detections
            if detection["class_name"]
            in {
                "Red Light",
                "Green Light",
            }
        ]
