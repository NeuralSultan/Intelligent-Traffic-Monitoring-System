from collections import defaultdict, deque
from typing import Optional

import numpy as np

from src.core.base_detector import BaseViolationDetector
from src.core.schemas import VehicleState, ViolationEvent
from configs.road_config_loader import RoadConfigLoader


class WrongWayDetector(BaseViolationDetector):
    """
    Detect wrong-way driving using:

        - tracked vehicle trajectory
        - road type (one-way / two-way)
        - allowed direction for one-way roads
        - expected road direction for two-way roads
        - robust movement direction estimation
        - direction consistency
        - persistent violation confirmation

    For one-way roads:
        vehicle direction is compared against allowed_direction.

    For two-way roads:
        vehicle direction is compared against the
        calibrated expected direction of its road side.

    The detector is responsible only for detecting
    the violation. Event storage is handled by
    ViolationManager.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        road_config: RoadConfigLoader,
        history_size: int = 30,
        min_history: int = 12,
        min_vertical_movement: float = 50.0,
        consistency_threshold: float = 0.85,
        confirmation_frames: int = 20,
        center_margin_ratio: float = 0.12,
        min_valid_movements: int = 8,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.min_history = min_history
        self.min_vertical_movement = (
            min_vertical_movement
        )
        self.consistency_threshold = (
            consistency_threshold
        )
        self.confirmation_frames = (
            confirmation_frames
        )
        self.min_valid_movements = (
            min_valid_movements
        )

        self.road_config = road_config

        # --------------------------------------------------
        # Road type
        # --------------------------------------------------

        self.road_type = (
            self.road_config.get_road_type()
        )

        self.allowed_direction = (
            self.road_config.get_allowed_direction()
            if self.road_config.is_one_way()
            else None
        )

        # --------------------------------------------------
        # Calibrated road-side positions
        # --------------------------------------------------

        self.left_x = None
        self.right_x = None

        if self.road_type == "two_way":

            self.left_x = (
                self.road_config.get_side_x(
                    "LEFT"
                )
            )

            self.right_x = (
                self.road_config.get_side_x(
                    "RIGHT"
                )
            )

            if (
                self.left_x is None
                or self.right_x is None
            ):
                raise ValueError(
                    "Missing calibrated LEFT/RIGHT "
                    "X positions in road configuration."
                )

        # --------------------------------------------------
        # Fallback center boundaries
        # --------------------------------------------------

        center_x = frame_width / 2.0
        center_margin = (
            frame_width * center_margin_ratio
        )

        self.left_boundary = (
            center_x - center_margin
        )

        self.right_boundary = (
            center_x + center_margin
        )

        # --------------------------------------------------
        # Track trajectories
        # --------------------------------------------------

        self.trajectories = defaultdict(
            lambda: deque(
                maxlen=history_size
            )
        )

        # --------------------------------------------------
        # Locked road side
        # --------------------------------------------------

        self.locked_sides = {}

        # --------------------------------------------------
        # Wrong-way confirmation streak
        # --------------------------------------------------

        self.wrong_way_streak = defaultdict(
            int
        )

        # --------------------------------------------------
        # Confirmed violations
        # --------------------------------------------------

        self.confirmed_violations = set()

    # ======================================================
    # Process
    # ======================================================

    def process(
        self,
        vehicle: VehicleState,
        frame,
        frame_number: int,
        timestamp: float,
    ) -> Optional[ViolationEvent]:

        track_id = int(vehicle.track_id)

        center_x = float(
            vehicle.center_x
        )

        center_y = float(
            vehicle.center_y
        )

        # --------------------------------------------------
        # 1. Update trajectory
        # --------------------------------------------------

        self.trajectories[track_id].append(
            (center_x, center_y)
        )

        history = self.trajectories[
            track_id
        ]

        # --------------------------------------------------
        # 2. Determine road side
        #
        # Only needed for two-way roads.
        # --------------------------------------------------

        side = None

        if self.road_type == "two_way":

            if track_id not in self.locked_sides:

                detected_side = (
                    self._determine_side(
                        center_x,
                        center_y,
                    )
                )

                if detected_side in {
                    "LEFT",
                    "RIGHT",
                }:

                    self.locked_sides[
                        track_id
                    ] = detected_side

            side = self.locked_sides.get(
                track_id
            )

        # --------------------------------------------------
        # 3. Estimate actual movement direction
        # --------------------------------------------------

        direction = (
            self._estimate_direction(
                history
            )
        )

        # --------------------------------------------------
        # 4. Calculate direction consistency
        # --------------------------------------------------

        consistency = (
            self._direction_consistency(
                history
            )
        )

        # --------------------------------------------------
        # 5. Determine expected direction
        #
        # One-way:
        #     use allowed_direction
        #
        # Two-way:
        #     use calibrated side direction
        # --------------------------------------------------

        expected_direction = None

        if self.road_config.is_one_way():

            expected_direction = (
                self.road_config
                .get_allowed_direction()
            )

        elif self.road_config.is_two_way():

            if side is not None:

                expected_direction = (
                    self.road_config
                    .get_expected_direction(
                        side
                    )
                )

        # --------------------------------------------------
        # 6. Determine wrong-way state
        # --------------------------------------------------

        is_wrong_way = False

        if (
            direction is not None
            and expected_direction is not None
            and consistency
            >= self.consistency_threshold
        ):

            is_wrong_way = (
                direction
                != expected_direction
            )

        # --------------------------------------------------
        # 7. Update confirmation streak
        # --------------------------------------------------

        if is_wrong_way:

            self.wrong_way_streak[
                track_id
            ] += 1

        else:

            self.wrong_way_streak[
                track_id
            ] = max(
                0,
                self.wrong_way_streak[
                    track_id
                ] - 2,
            )

        # --------------------------------------------------
        # 8. Confirm violation
        # --------------------------------------------------

        confirmed = (
            self.wrong_way_streak[
                track_id
            ]
            >= self.confirmation_frames
        )

        if (
            confirmed
            and track_id
            not in self.confirmed_violations
        ):

            self.confirmed_violations.add(
                track_id
            )

            confidence = min(
                1.0,
                consistency,
            )

            return ViolationEvent(
                violation_type="wrong_way",

                vehicle_id=track_id,

                frame_number=frame_number,

                timestamp=timestamp,

                class_name=vehicle.class_name,

                direction=direction,

                expected_direction=(
                    expected_direction
                ),

                severity="high",

                confidence=confidence,

                details={
                    "road_type": (
                        self.road_type
                    ),

                    "road_side": side,

                    "direction_consistency": (
                        consistency
                    ),

                    "wrong_way_streak": (
                        self.wrong_way_streak[
                            track_id
                        ]
                    ),

                    "bbox": list(
                        vehicle.bbox
                    ),
                },
            )

        return None

    # ======================================================
    # Road Side
    # ======================================================

    def _determine_side(
        self,
        x,
        y,
    ):
        """
        Determine LEFT/RIGHT using calibrated
        road positions.
        """

        # Ignore vehicles that are still too high
        # in the frame.
        if (
            y
            < self.frame_height * 0.35
        ):
            return None

        # Use calibrated positions when available.
        if (
            self.left_x is not None
            and self.right_x is not None
        ):

            midpoint = (
                float(self.left_x)
                + float(self.right_x)
            ) / 2.0

            if x < midpoint:
                return "LEFT"

            return "RIGHT"

        # Fallback to center-based method.
        if x < self.left_boundary:
            return "LEFT"

        if x > self.right_boundary:
            return "RIGHT"

        return None

    # ======================================================
    # Direction Estimation
    # ======================================================

    def _estimate_direction(
        self,
        history,
    ) -> Optional[str]:

        if (
            len(history)
            < self.min_history
        ):
            return None

        points = np.asarray(
            history,
            dtype=np.float32,
        )

        y_values = points[:, 1]

        # --------------------------------------------------
        # Smooth Y trajectory
        # --------------------------------------------------

        if len(y_values) >= 5:

            kernel = (
                np.ones(
                    5,
                    dtype=np.float32,
                )
                / 5.0
            )

            smoothed_y = np.convolve(
                y_values,
                kernel,
                mode="same",
            )

        else:

            smoothed_y = y_values

        # --------------------------------------------------
        # Calculate frame-to-frame movement
        # --------------------------------------------------

        delta_y = np.diff(
            smoothed_y
        )

        # Ignore tiny tracking noise.
        valid_delta = delta_y[
            np.abs(delta_y) >= 1.5
        ]

        if (
            len(valid_delta)
            < self.min_valid_movements
        ):
            return None

        # --------------------------------------------------
        # Calculate total vertical displacement
        # --------------------------------------------------

        total_movement = (
            smoothed_y[-1]
            - smoothed_y[0]
        )

        if (
            abs(total_movement)
            < self.min_vertical_movement
        ):
            return None

        # --------------------------------------------------
        # Estimate dominant direction
        # --------------------------------------------------

        up_votes = np.sum(
            valid_delta < 0
        )

        down_votes = np.sum(
            valid_delta > 0
        )

        total_votes = (
            up_votes + down_votes
        )

        if total_votes == 0:
            return None

        dominant_ratio = (
            max(
                up_votes,
                down_votes,
            )
            / total_votes
        )

        # Do not make a direction decision
        # if movement is ambiguous.
        if (
            dominant_ratio
            < self.consistency_threshold
        ):
            return None

        if up_votes > down_votes:
            return "UP"

        return "DOWN"

    # ======================================================
    # Direction Consistency
    # ======================================================

    def _direction_consistency(
        self,
        history,
    ) -> float:

        if (
            len(history)
            < self.min_history
        ):
            return 0.0

        points = np.asarray(
            history,
            dtype=np.float32,
        )

        y_values = points[:, 1]

        # Smooth trajectory.
        if len(y_values) >= 5:

            kernel = (
                np.ones(
                    5,
                    dtype=np.float32,
                )
                / 5.0
            )

            smoothed_y = np.convolve(
                y_values,
                kernel,
                mode="same",
            )

        else:

            smoothed_y = y_values

        delta_y = np.diff(
            smoothed_y
        )

        # Ignore tracking noise.
        valid_delta = delta_y[
            np.abs(delta_y) >= 1.5
        ]

        if (
            len(valid_delta)
            < self.min_valid_movements
        ):
            return 0.0

        up_votes = np.sum(
            valid_delta < 0
        )

        down_votes = np.sum(
            valid_delta > 0
        )

        total_votes = (
            up_votes + down_votes
        )

        if total_votes == 0:
            return 0.0

        return (
            max(
                up_votes,
                down_votes,
            )
            / total_votes
        )

    # ======================================================
    # Public Helpers
    # ======================================================

    def get_violations(self):
        """
        Return IDs of vehicles confirmed
        as wrong-way.
        """

        return set(
            self.confirmed_violations
        )

    def remove_track(
        self,
        track_id: int,
    ):
        
        #Remove all stored state for a track.


        self.trajectories.pop(
            track_id,
            None,
        )

        self.locked_sides.pop(
            track_id,
            None,
        )

        self.wrong_way_streak.pop(
            track_id,
            None,
        )
