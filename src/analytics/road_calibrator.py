from pathlib import Path
from collections import defaultdict, deque

import cv2
import json
import numpy as np

from ultralytics import YOLO


class RoadCalibrator:
    """
    Automatically estimates the normal traffic direction
    for the left and right sides of a two-way road.

    The calibrator analyzes vehicle trajectories and finds
    the dominant movement direction on each side.
    """

    VEHICLE_CLASSES = {
        0: "car",
        1: "bus",
        2: "truck",
        3: "motorcycle",
        4: "bicycle",
    }

    VALID_DIRECTIONS = {"UP", "DOWN"}

    def __init__(
        self,
        model_path: str | Path,
        video_path: str | Path,
        config_path: str | Path,
        confidence: float = 0.35,
        image_size: int = 640,
        history_size: int = 25,
        min_history: int = 10,
        min_vertical_movement: float = 40.0,
        consistency_threshold: float = 0.75,
        center_margin_ratio: float = 0.12,
        top_ignore_ratio: float = 0.35,
        min_tracks_per_side: int = 3,
        max_calibration_frames: int = 1500,
    ):
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.config_path = Path(config_path)

        self.confidence = confidence
        self.image_size = image_size

        self.history_size = history_size
        self.min_history = min_history
        self.min_vertical_movement = min_vertical_movement
        self.consistency_threshold = consistency_threshold

        self.center_margin_ratio = center_margin_ratio
        self.top_ignore_ratio = top_ignore_ratio

        self.min_tracks_per_side = min_tracks_per_side
        self.max_calibration_frames = (
            max_calibration_frames
        )

        self.trajectories = defaultdict(
            lambda: deque(
                maxlen=self.history_size
            )
        )

        self.track_sides = {}
        self.track_directions = {}
        self.track_consistencies = {}

    def calibrate(self):
        """
        Run automatic road calibration and save
        the resulting configuration.
        """

        print("\n" + "=" * 60)
        print("STARTING ROAD AUTO-CALIBRATION")
        print("=" * 60)

        model = YOLO(
            str(self.model_path)
        )

        cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video:\n"
                f"{self.video_path}"
            )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        center_x = width / 2

        center_margin = (
            width * self.center_margin_ratio
        )

        left_boundary = (
            center_x - center_margin
        )

        right_boundary = (
            center_x + center_margin
        )

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            if (
                frame_number
                > self.max_calibration_frames
            ):
                break

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
            )

            result = results[0]

            if (
                result.boxes is None
                or result.boxes.id is None
            ):
                continue

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

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, class_id, track_id in zip(
                boxes,
                class_ids,
                track_ids,
            ):

                if class_id not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                center_x_vehicle = (
                    (x1 + x2) / 2
                )

                center_y_vehicle = (
                    (y1 + y2) / 2
                )

                track_id = int(track_id)

                # --------------------------------------------
                # Update trajectory
                # --------------------------------------------

                self.trajectories[
                    track_id
                ].append(
                    (
                        float(center_x_vehicle),
                        float(center_y_vehicle),
                    )
                )

                history = self.trajectories[
                    track_id
                ]

                # --------------------------------------------
                # Determine side
                # --------------------------------------------

                if track_id not in self.track_sides:

                    side = self._determine_side(
                        center_x_vehicle,
                        center_y_vehicle,
                        width=width,
                        height=height,
                        left_boundary=left_boundary,
                        right_boundary=right_boundary,
                    )

                    if side in {
                        "LEFT",
                        "RIGHT",
                    }:
                        self.track_sides[
                            track_id
                        ] = side

                side = self.track_sides.get(
                    track_id
                )

                # --------------------------------------------
                # Estimate direction
                # --------------------------------------------

                direction = (
                    self._estimate_direction(
                        history
                    )
                )

                consistency = (
                    self._direction_consistency(
                        history
                    )
                )

                if (
                    direction is not None
                    and consistency
                    >= self.consistency_threshold
                ):

                    self.track_directions[
                        track_id
                    ] = direction

                    self.track_consistencies[
                        track_id
                    ] = consistency

        cap.release()

        # ====================================================
        # Aggregate reliable tracks
        # ====================================================

        left_directions = []
        right_directions = []

        for track_id, direction in (
            self.track_directions.items()
        ):

            side = self.track_sides.get(
                track_id
            )

            consistency = (
                self.track_consistencies.get(
                    track_id,
                    0.0,
                )
            )

            if (
                side is None
                or consistency
                < self.consistency_threshold
            ):
                continue

            if side == "LEFT":
                left_directions.append(
                    direction
                )

            elif side == "RIGHT":
                right_directions.append(
                    direction
                )

        # ====================================================
        # Dominant directions
        # ====================================================

        left_direction, left_confidence = (
            self._dominant_direction(
                left_directions
            )
        )

        right_direction, right_confidence = (
            self._dominant_direction(
                right_directions
            )
        )

        calibration_valid = (
            left_direction is not None
            and right_direction is not None
            and len(left_directions)
            >= self.min_tracks_per_side
            and len(right_directions)
            >= self.min_tracks_per_side
        )

        road_config = {
            "video_width": width,
            "video_height": height,

            "left_side": {
                "expected_direction": left_direction,
                "confidence": round(
                    left_confidence,
                    3,
                ),
                "tracks_used": len(
                    left_directions
                ),
            },

            "right_side": {
                "expected_direction": right_direction,
                "confidence": round(
                    right_confidence,
                    3,
                ),
                "tracks_used": len(
                    right_directions
                ),
            },

            "calibration_valid": (
                calibration_valid
            ),
        }

        self.config_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.config_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                road_config,
                file,
                indent=4,
            )

        # ====================================================
        # Calibration report
        # ====================================================

        print("\n" + "=" * 60)
        print("ROAD AUTO-CALIBRATION COMPLETED")
        print("=" * 60)

        print(
            f"Frames analyzed: {frame_number}"
        )

        print("\nLEFT side:")
        print(
            f"  Direction: {left_direction}"
        )
        print(
            f"  Confidence: "
            f"{left_confidence:.2%}"
        )
        print(
            f"  Tracks used: "
            f"{len(left_directions)}"
        )

        print("\nRIGHT side:")
        print(
            f"  Direction: {right_direction}"
        )
        print(
            f"  Confidence: "
            f"{right_confidence:.2%}"
        )
        print(
            f"  Tracks used: "
            f"{len(right_directions)}"
        )

        print(
            f"\nCalibration valid: "
            f"{calibration_valid}"
        )

        print(
            "\nConfiguration saved to:"
        )

        print(self.config_path)

        if not calibration_valid:
            raise RuntimeError(
                "Automatic road calibration failed. "
                "Not enough reliable traffic tracks "
                "were available on both road sides."
            )

        return road_config

    def _determine_side(
        self,
        x,
        y,
        width,
        height,
        left_boundary,
        right_boundary,
    ):
        """
        Determine road side after the vehicle gets
        far enough from the vanishing point.
        """

        if y < height * self.top_ignore_ratio:
            return None

        if x < left_boundary:
            return "LEFT"

        if x > right_boundary:
            return "RIGHT"

        return None

    def _estimate_direction(
        self,
        history,
    ):
        """
        Estimate vertical movement direction.
        """

        if len(history) < self.min_history:
            return None

        points = np.asarray(
            history,
            dtype=np.float32,
        )

        y_values = points[:, 1]

        if len(y_values) >= 5:

            kernel = np.ones(
                5,
                dtype=np.float32,
            ) / 5.0

            y_values = np.convolve(
                y_values,
                kernel,
                mode="valid",
            )

        delta_y = (
            float(y_values[-1])
            - float(y_values[0])
        )

        if abs(delta_y) < self.min_vertical_movement:
            return None

        if delta_y < 0:
            return "UP"

        return "DOWN"

    def _direction_consistency(
        self,
        history,
    ):
        """
        Calculate directional consistency.
        """

        if len(history) < self.min_history:
            return 0.0

        points = list(history)

        movements = []

        for i in range(
            1,
            len(points),
        ):

            previous_y = (
                points[i - 1][1]
            )

            current_y = points[i][1]

            delta_y = (
                current_y - previous_y
            )

            if abs(delta_y) < 1.5:
                continue

            if delta_y > 0:
                movements.append("DOWN")
            else:
                movements.append("UP")

        if not movements:
            return 0.0

        up = movements.count("UP")
        down = movements.count("DOWN")

        return max(up, down) / len(movements)

    def _dominant_direction(
        self,
        directions,
    ):
        """
        Return dominant direction and its confidence.
        """

        if not directions:
            return None, 0.0

        up_count = directions.count("UP")
        down_count = directions.count("DOWN")

        total = len(directions)

        if up_count >= down_count:

            return (
                "UP",
                up_count / total,
            )

        return (
            "DOWN",
            down_count / total,
        )