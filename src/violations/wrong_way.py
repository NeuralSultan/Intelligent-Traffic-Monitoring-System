from collections import defaultdict, deque
from configs.road_config_loader import RoadConfigLoader
import numpy as np


class WrongWayDetector:
    """
    Detects wrong-way driving using:
        - tracked vehicle trajectory
        - locked road side
        - movement direction
        - direction consistency
        - persistent violation confirmation
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        road_config: RoadConfigLoader,
        history_size: int = 25,
        min_history: int = 10,
        min_vertical_movement: float = 40.0,
        consistency_threshold: float = 0.75,
        confirmation_frames: int = 15,
        center_margin_ratio: float = 0.12,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.min_history = min_history
        self.min_vertical_movement = min_vertical_movement
        self.consistency_threshold = consistency_threshold
        self.confirmation_frames = confirmation_frames

        # Road geometry
        center_x = frame_width / 2
        center_margin = frame_width * center_margin_ratio

        self.left_boundary = center_x - center_margin
        self.right_boundary = center_x + center_margin

        # Your road:
        # LEFT  -> DOWN
        # RIGHT -> UP
        self.road_config = road_config

        # Tracking state
        self.trajectories = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        self.locked_sides = {}

        self.wrong_way_streak = defaultdict(int)

        self.confirmed_violations = set()

    def update(
        self,
        track_id: int,
        center_x: float,
        center_y: float,
    ):
        """
        Update one tracked vehicle and return its
        wrong-way status.
        """

        track_id = int(track_id)

        center_x = float(center_x)
        center_y = float(center_y)

        # ----------------------------------------------------
        # Update trajectory
        # ----------------------------------------------------

        self.trajectories[track_id].append(
            (center_x, center_y)
        )

        history = self.trajectories[track_id]

        # ----------------------------------------------------
        # Lock road side
        # ----------------------------------------------------

        if track_id not in self.locked_sides:

            side = self._determine_side(
                center_x,
                center_y,
            )

            if side in {"LEFT", "RIGHT"}:
                self.locked_sides[track_id] = side

        side = self.locked_sides.get(track_id)

        # ----------------------------------------------------
        # Estimate actual direction
        # ----------------------------------------------------

        direction = self._estimate_direction(history)

        consistency = self._direction_consistency(history)

        expected_direction = None

        if side is not None:
            expected_direction = (
                self.road_config.get_expected_direction(side)
            )

        # ----------------------------------------------------
        # Determine whether current movement is wrong
        # ----------------------------------------------------

        is_wrong_way = False

        if (
            direction is not None
            and expected_direction is not None
            and consistency >= self.consistency_threshold
        ):
            is_wrong_way = (
                direction != expected_direction
            )

        # ----------------------------------------------------
        # Persistent confirmation
        # ----------------------------------------------------

        if is_wrong_way:

            self.wrong_way_streak[track_id] += 1

        else:

            self.wrong_way_streak[track_id] = max(
                0,
                self.wrong_way_streak[track_id] - 1,
            )

        confirmed = (
            self.wrong_way_streak[track_id]
            >= self.confirmation_frames
        )

        if confirmed:

            self.confirmed_violations.add(
                track_id
            )

        return {
            "track_id": track_id,
            "side": side,
            "direction": direction,
            "expected_direction": expected_direction,
            "consistency": consistency,
            "wrong_way": is_wrong_way,
            "streak": self.wrong_way_streak[track_id],
            "confirmed": confirmed,
        }

    def _determine_side(
        self,
        x: float,
        y: float,
    ):
        """
        Determine the road side only when the vehicle
        is sufficiently far from the vanishing-point area.
        """

        # Ignore vehicles near the top of the frame.
        if y < self.frame_height * 0.35:
            return None

        if x < self.left_boundary:
            return "LEFT"

        if x > self.right_boundary:
            return "RIGHT"

        return None

    def _estimate_direction(
        self,
        history,
    ):
        """
        Estimate direction from trajectory.

        UP   -> Y decreases
        DOWN -> Y increases
        """

        if len(history) < self.min_history:
            return None

        points = np.asarray(
            history,
            dtype=np.float32,
        )

        y_values = points[:, 1]

        # Smooth trajectory.
        if len(y_values) >= 5:

            kernel = np.ones(
                5,
                dtype=np.float32,
            ) / 5.0

            smoothed = np.convolve(
                y_values,
                kernel,
                mode="valid",
            )

        else:

            smoothed = y_values

        start_y = float(smoothed[0])
        end_y = float(smoothed[-1])

        delta_y = end_y - start_y

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
        Calculate the percentage of meaningful trajectory
        steps that agree on the dominant vertical direction.
        """

        if len(history) < self.min_history:
            return 0.0

        points = list(history)

        movements = []

        for i in range(1, len(points)):

            previous_y = points[i - 1][1]
            current_y = points[i][1]

            delta_y = current_y - previous_y

            if abs(delta_y) < 1.5:
                continue

            if delta_y > 0:
                movements.append("DOWN")
            else:
                movements.append("UP")

        if not movements:
            return 0.0

        up_votes = movements.count("UP")
        down_votes = movements.count("DOWN")

        return max(
            up_votes,
            down_votes,
        ) / len(movements)

    def get_violations(self):
        """
        Return all currently confirmed wrong-way vehicle IDs.
        """

        return set(
            self.confirmed_violations
        )

    def remove_track(self, track_id: int):
        """
        Remove a finished track from memory.
        """

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