from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


class RoadCalibrator:
    """
    Automatically calibrates traffic direction from
    tracked vehicle trajectories.

    TWO_WAY:
        1. Determine vehicle movement direction.
        2. Group vehicles into UP and DOWN flows.
        3. Use median X position to determine which flow
           is physically on the LEFT and RIGHT side.

    ONE_WAY:
        1. Determine the dominant traffic direction.
        2. No artificial second traffic side is created.
    """

    def __init__(
        self,
        model_path: str | Path,
        video_path: str | Path,
        config_path: str | Path,
        confidence: float = 0.35,
        image_size: int = 640,
        history_size: int = 25,
        min_history: int = 10,
        min_vertical_movement: float = 2.0,
        consistency_threshold: float = 0.75,
        center_margin_ratio: float = 0.12,
        top_ignore_ratio: float = 0.35,
        min_tracks_per_side: int = 3,
        max_calibration_frames: int = 1500,
        road_type: str = "two_way",
    ):
        
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.config_path = Path(config_path)

        self.confidence = confidence
        self.image_size = image_size

        self.history_size = history_size
        self.min_history = min_history

        self.min_vertical_movement = (
            min_vertical_movement
        )

        self.consistency_threshold = (
            consistency_threshold
        )

        # Kept for compatibility with the existing
        # constructor/configuration.
        self.center_margin_ratio = (
            center_margin_ratio
        )

        # Kept for compatibility, but it is no longer
        # used for LEFT/RIGHT assignment.
        self.top_ignore_ratio = (
            top_ignore_ratio
        )

        self.min_tracks_per_side = (
            min_tracks_per_side
        )

        self.max_calibration_frames = (
            max_calibration_frames
        )

        self.road_type = road_type.lower()

        if self.road_type not in {
            "one_way",
            "two_way",
        }:
            raise ValueError(
                "road_type must be 'one_way' or 'two_way'."
            )

        self.model = YOLO(
            str(self.model_path)
        )

        # --------------------------------------------------------
        # Trajectory history
        # --------------------------------------------------------

        self.trajectories = defaultdict(
            list
        )

        # --------------------------------------------------------
        # Per-track direction information
        # --------------------------------------------------------

        self.track_directions = {}
        self.track_consistencies = {}

        # --------------------------------------------------------
        # Per-track side assignment
        # --------------------------------------------------------

        self.track_sides = {}

        # --------------------------------------------------------
        # Final calibration result
        # --------------------------------------------------------

        self.frames_analyzed = 0

        self.left_direction = None
        self.right_direction = None

        self.left_confidence = 0.0
        self.right_confidence = 0.0

        self.left_tracks = 0
        self.right_tracks = 0

        self.calibration_valid = False
        self.left_x = None
        self.right_x = None
    # ============================================================
    # PUBLIC
    # ============================================================

    def calibrate(self):
        """
        Run automatic road calibration.
        """

        print("=" * 60)
        print("STARTING ROAD AUTO-CALIBRATION")
        print("=" * 60)

        print(
            f"Road type: "
            f"{self.road_type.upper()}"
        )

        cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: "
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

        frame_number = 0

        while (
            frame_number
            < self.max_calibration_frames
        ):
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            self.frames_analyzed = (
                frame_number
            )

            self._process_frame(frame)

        cap.release()

        # --------------------------------------------------------
        # Calculate direction of every reliable track.
        # --------------------------------------------------------

        self._calculate_track_directions()

        # --------------------------------------------------------
        # Debug.
        # --------------------------------------------------------

        self._print_track_debug()

        # --------------------------------------------------------
        # Assign traffic flows to LEFT / RIGHT.
        # --------------------------------------------------------

        self._assign_track_sides()

        # --------------------------------------------------------
        # Calculate final result.
        # --------------------------------------------------------

        if self.road_type == "two_way":
            self._calculate_two_way_result()
        else:
            self._calculate_one_way_result()

        # --------------------------------------------------------
        # Validate.
        # --------------------------------------------------------

        self.calibration_valid = (
            self._validate_calibration()
        )

        # --------------------------------------------------------
        # Save.
        # --------------------------------------------------------

        self._save_config(
            width=width,
            height=height,
        )

        # --------------------------------------------------------
        # Print.
        # --------------------------------------------------------

        self._print_result()

        return self.calibration_valid

    # ============================================================
    # FRAME PROCESSING
    # ============================================================

    def _process_frame(self, frame):
        """
        Run YOLO + ByteTrack on one frame.

        Stores the bottom-center point of every
        tracked vehicle.
        """

        results = self.model.track(
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
            return

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id in zip(
            boxes,
            track_ids,
        ):
            x1, y1, x2, y2 = box

            # Bottom-center point.
            center_x = (
                float(x1) + float(x2)
            ) / 2.0

            center_y = float(y2)

            history = self.trajectories[
                int(track_id)
            ]

            history.append(
                (
                    center_x,
                    center_y,
                )
            )

            # Keep only recent trajectory points.
            if (
                len(history)
                > self.history_size
            ):
                history.pop(0)

    # ============================================================
    # DIRECTION ANALYSIS
    # ============================================================

    def _calculate_track_directions(self):
        """
        Determine UP/DOWN movement for every reliable track.
        """

        self.track_directions.clear()
        self.track_consistencies.clear()

        for track_id, history in self.trajectories.items():

            if len(history) < self.min_history:
                continue

            y_positions = np.array(
                [
                    point[1]
                    for point in history
                ],
                dtype=float,
            )

            if len(y_positions) < 2:
                continue

            # --------------------------------------------------------
            # Overall movement of the track.
            # --------------------------------------------------------

            vertical_movement = (
                y_positions[-1]
                - y_positions[0]
            )

            # Ignore tracks that barely moved.
            if abs(vertical_movement) < 20.0:
                continue

            # --------------------------------------------------------
            # Frame-to-frame movement.
            # --------------------------------------------------------

            y_deltas = np.diff(y_positions)

            # Remove very small tracking jitter.
            valid_deltas = y_deltas[
                np.abs(y_deltas)
                >= self.min_vertical_movement
            ]

            if len(valid_deltas) < 2:
                continue

            # --------------------------------------------------------
            # Vote for direction.
            # --------------------------------------------------------

            up_votes = int(
                np.sum(
                    valid_deltas < 0
                )
            )

            down_votes = int(
                np.sum(
                    valid_deltas > 0
                )
            )

            total_votes = (
                up_votes
                + down_votes
            )

            if total_votes == 0:
                continue

            if up_votes >= down_votes:
                direction = "UP"
                dominant_votes = up_votes
            else:
                direction = "DOWN"
                dominant_votes = down_votes

            consistency = (
                dominant_votes
                / total_votes
            )

            # --------------------------------------------------------
            # Require reasonably consistent movement.
            # --------------------------------------------------------

            if (
                consistency
                < self.consistency_threshold
            ):
                continue

            # --------------------------------------------------------
            # Final direction is based on overall trajectory.
            # --------------------------------------------------------

            if vertical_movement < 0:
                direction = "UP"
            else:
                direction = "DOWN"

            self.track_directions[
                track_id
            ] = direction

            self.track_consistencies[
                track_id
            ] = float(consistency)

    # ============================================================
    # DEBUG
    # ============================================================

    def _print_track_debug(self):
        """
        Print all reliable tracks and their directions.
        """

        print()
        print(
            "TRACK DIRECTION DEBUG:"
        )

        for track_id, direction in (
            self.track_directions.items()
        ):
            consistency = (
                self.track_consistencies.get(
                    track_id,
                    0.0,
                )
            )

            print(
                f"ID={track_id} "
                f"direction={direction} "
                f"consistency={consistency:.2%}"
            )

    # ============================================================
    # SIDE ASSIGNMENT
    # ============================================================

    def _assign_track_sides(self):
        """
        Assign traffic flows to LEFT / RIGHT.

        TWO_WAY logic:

            UP tracks
                +
            DOWN tracks

            ↓

            Calculate median X of each flow.

            ↓

            Smaller X = LEFT
            Larger X  = RIGHT

        IMPORTANT:

        We do NOT split the image at its center.

        The image center is not necessarily the
        center of the actual road.
        """

        self.track_sides.clear()

        reliable_tracks = []

        # --------------------------------------------------------
        # Collect reliable tracks.
        # --------------------------------------------------------

        for track_id, direction in (
            self.track_directions.items()
        ):
            history = self.trajectories.get(
                track_id,
                [],
            )

            if (
                len(history)
                < self.min_history
            ):
                continue

            consistency = (
                self.track_consistencies.get(
                    track_id,
                    0.0,
                )
            )

            if (
                consistency
                < self.consistency_threshold
            ):
                continue

            # IMPORTANT:
            #
            # We intentionally use the complete trajectory.
            #
            # We do NOT remove the upper part of the frame.
            #
            # Direction has already been validated, so there
            # is no reason to throw away valid tracks here.
            x_values = np.array(
                [
                    point[0]
                    for point in history
                ],
                dtype=float,
            )

            if len(x_values) == 0:
                continue

            median_x = float(
                np.median(x_values)
            )

            reliable_tracks.append(
                {
                    "track_id": int(
                        track_id
                    ),
                    "direction": direction,
                    "median_x": median_x,
                    "consistency": consistency,
                }
            )

        # --------------------------------------------------------
        # ONE WAY
        # --------------------------------------------------------

        if self.road_type == "one_way":

            for track in reliable_tracks:

                self.track_sides[
                    track["track_id"]
                ] = "LEFT"

            return

        # --------------------------------------------------------
        # TWO WAY
        # --------------------------------------------------------

        up_tracks = [
            track
            for track in reliable_tracks
            if track["direction"] == "UP"
        ]

        down_tracks = [
            track
            for track in reliable_tracks
            if track["direction"] == "DOWN"
        ]

        print()
        print(
            "SIDE ASSIGNMENT DEBUG:"
        )

        print(
            f"Reliable tracks total: "
            f"{len(reliable_tracks)}"
        )

        print(
            f"UP tracks: "
            f"{len(up_tracks)}"
        )

        print(
            f"DOWN tracks: "
            f"{len(down_tracks)}"
        )

        if up_tracks:

            print(
                "UP X positions:",
                [
                    round(
                        track["median_x"],
                        1,
                    )
                    for track in up_tracks
                ],
            )

        if down_tracks:

            print(
                "DOWN X positions:",
                [
                    round(
                        track["median_x"],
                        1,
                    )
                    for track in down_tracks
                ],
            )

        # --------------------------------------------------------
        # Both traffic directions must exist.
        # --------------------------------------------------------

        if (
            len(up_tracks) == 0
            or len(down_tracks) == 0
        ):
            print(
                "WARNING: Could not find "
                "both traffic directions."
            )
            return

        # --------------------------------------------------------
        # Calculate median X for each traffic flow.
        # --------------------------------------------------------

        up_median_x = float(
            np.median(
                [
                    track["median_x"]
                    for track in up_tracks
                ]
            )
        )

        down_median_x = float(
            np.median(
                [
                    track["median_x"]
                    for track in down_tracks
                ]
            )
        )

        print(
            f"UP median X: "
            f"{up_median_x:.2f}"
        )

        print(
            f"DOWN median X: "
            f"{down_median_x:.2f}"
        )

        # --------------------------------------------------------
        # Determine physical LEFT / RIGHT.
        # --------------------------------------------------------

        if (
            up_median_x
            < down_median_x
        ):
            left_direction = "UP"
            right_direction = "DOWN"

        else:
            left_direction = "DOWN"
            right_direction = "UP"

        print(
            f"LEFT flow: "
            f"{left_direction}"
        )

        print(
            f"RIGHT flow: "
            f"{right_direction}"
        )

        # --------------------------------------------------------
        # Assign UP tracks.
        # --------------------------------------------------------

        for track in up_tracks:

            track_id = track[
                "track_id"
            ]

            if left_direction == "UP":

                self.track_sides[
                    track_id
                ] = "LEFT"

            else:

                self.track_sides[
                    track_id
                ] = "RIGHT"

        # --------------------------------------------------------
        # Assign DOWN tracks.
        # --------------------------------------------------------

        for track in down_tracks:

            track_id = track[
                "track_id"
            ]

            if left_direction == "DOWN":

                self.track_sides[
                    track_id
                ] = "LEFT"

            else:

                self.track_sides[
                    track_id
                ] = "RIGHT"

        # --------------------------------------------------------
        # Debug assignment counts.
        # --------------------------------------------------------

        left_count = sum(
            1
            for side in (
                self.track_sides.values()
            )
            if side == "LEFT"
        )

        right_count = sum(
            1
            for side in (
                self.track_sides.values()
            )
            if side == "RIGHT"
        )

        print(
            f"Assigned LEFT tracks: "
            f"{left_count}"
        )

        print(
            f"Assigned RIGHT tracks: "
            f"{right_count}"
        )

    # ============================================================
    # RESULT CALCULATION
    # ============================================================

    def _calculate_two_way_result(self):
        """
        Calculate final LEFT and RIGHT directions
        and representative X positions.
        """

        left_directions = []
        right_directions = []

        left_x_positions = []
        right_x_positions = []

        for track_id, side in (
            self.track_sides.items()
        ):
            direction = (
                self.track_directions.get(
                    track_id
                )
            )

            history = self.trajectories.get(
                track_id,
                [],
            )

            if direction is None:
                continue

            if not history:
                continue

            x_values = np.array(
                [
                    point[0]
                    for point in history
                ],
                dtype=float,
            )

            median_x = float(
                np.median(x_values)
            )

            if side == "LEFT":

                left_directions.append(
                    direction
                )

                left_x_positions.append(
                    median_x
                )

            elif side == "RIGHT":

                right_directions.append(
                    direction
                )

                right_x_positions.append(
                    median_x
                )

        (
            self.left_direction,
            self.left_confidence,
        ) = self._dominant_direction(
            left_directions
        )

        (
            self.right_direction,
            self.right_confidence,
        ) = self._dominant_direction(
            right_directions
        )

        self.left_tracks = len(
            left_directions
        )

        self.right_tracks = len(
            right_directions
        )

        # --------------------------------------------------------
        # Representative X position
        # --------------------------------------------------------

        if left_x_positions:
            self.left_x = float(
                np.median(
                    left_x_positions
                )
            )

        if right_x_positions:
            self.right_x = float(
                np.median(
                    right_x_positions
                )
            )
    def _calculate_one_way_result(self):
        """
        Calculate final direction for a ONE_WAY road.
        """

        directions = list(
            self.track_directions.values()
        )

        (
            dominant_direction,
            confidence,
        ) = self._dominant_direction(
            directions
        )

        self.left_direction = (
            dominant_direction
        )

        self.left_confidence = (
            confidence
        )

        self.right_direction = None
        self.right_confidence = 0.0

        self.left_tracks = len(
            directions
        )

        self.right_tracks = 0

    # ============================================================
    # DOMINANT DIRECTION
    # ============================================================

    def _dominant_direction(
        self,
        directions: list[str],
    ):
        """
        Determine the dominant direction.

        Confidence is the percentage of tracks voting
        for the dominant direction.

        A direction below 85% confidence is considered
        unreliable and is flipped according to the
        existing project rule.
        """

        if not directions:
            return None, 0.0

        up_count = directions.count(
            "UP"
        )

        down_count = directions.count(
            "DOWN"
        )

        total = (
            up_count
            + down_count
        )

        if total == 0:
            return None, 0.0

        if up_count >= down_count:

            raw_direction = "UP"

            confidence = (
                up_count / total
            )

        else:

            raw_direction = "DOWN"

            confidence = (
                down_count / total
            )

        if confidence < 0.85:

            final_direction = (
                "DOWN"
                if raw_direction == "UP"
                else "UP"
            )

        else:

            final_direction = (
                raw_direction
            )

        return (
            final_direction,
            float(confidence),
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_calibration(self):
        """
        Validate the final calibration.
        """

        # --------------------------------------------------------
        # ONE WAY
        # --------------------------------------------------------

        if self.road_type == "one_way":

            if self.left_direction is None:
                return False

            if (
                self.left_tracks
                < self.min_tracks_per_side
            ):
                return False

            return True

        # --------------------------------------------------------
        # TWO WAY
        # --------------------------------------------------------

        if self.left_direction is None:
            return False

        if self.right_direction is None:
            return False

        if (
            self.left_tracks
            < self.min_tracks_per_side
        ):
            return False

        if (
            self.right_tracks
            < self.min_tracks_per_side
        ):
            return False

        # Two-way traffic must have
        # opposite directions.
        if (
            self.left_direction
            == self.right_direction
        ):
            return False

        return True

    # ============================================================
    # SAVE CONFIGURATION
    # ============================================================


    def _save_config(
        self,
        width: int,
        height: int,
    ):
        """
        Save calibration result to road_config.json.
        """

        self.config_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        config = {
            "video_name": self.video_path.name,
            "video_width": width,
            "video_height": height,
            "road_type": self.road_type,
            "calibration_valid": (
                self.calibration_valid
            ),
            "direction": {
                # These names are used by WrongWayDetector.
                "left_x": round(
                    self.left_x,
                    2,
                ) if self.left_x is not None else None,

                "right_x": round(
                    self.right_x,
                    2,
                ) if self.right_x is not None else None,
                "left_side": self.left_direction,
                "right_side": self.right_direction,

                "left_confidence": round(
                    self.left_confidence,
                    4,
                ),
                "right_confidence": round(
                    self.right_confidence,
                    4,
                ),
                "left_tracks": self.left_tracks,
                "right_tracks": self.right_tracks,
            },
        }

        import json

        with open(
            self.config_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                config,
                file,
                indent=4,
            )


    # ============================================================
    # PRINT RESULT
    # ============================================================

    def _print_result(self):
        """
        Print final calibration result.
        """

        print()
        print("=" * 60)
        print(
            "ROAD AUTO-CALIBRATION COMPLETED"
        )
        print("=" * 60)

        print(
            f"Frames analyzed: "
            f"{self.frames_analyzed}"
        )

        print(
            f"Road type: "
            f"{self.road_type.upper()}"
        )

        print()

        print("LEFT side:")

        print(
            f"  Direction: "
            f"{self.left_direction}"
        )

        print(
            f"  Confidence: "
            f"{self.left_confidence:.2%}"
        )

        print(
            f"  Tracks used: "
            f"{self.left_tracks}"
        )

        print()

        print("RIGHT side:")

        print(
            f"  Direction: "
            f"{self.right_direction}"
        )

        print(
            f"  Confidence: "
            f"{self.right_confidence:.2%}"
        )

        print(
            f"  Tracks used: "
            f"{self.right_tracks}"
        )

        print()

        print(
            f"Calibration valid: "
            f"{self.calibration_valid}"
        )

        print("=" * 60)