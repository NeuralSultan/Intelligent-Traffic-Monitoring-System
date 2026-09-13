from collections import defaultdict, deque

import numpy as np


class DirectionEngine:
    """
    Estimates the movement direction of tracked vehicles
    from their recent trajectory.

    UP   = decreasing Y
    DOWN = increasing Y
    """

    def __init__(
        self,
        history_size: int = 25,
        min_history: int = 10,
        min_vertical_movement: float = 40.0,
        consistency_threshold: float = 0.75,
    ):
        self.history_size = history_size
        self.min_history = min_history
        self.min_vertical_movement = min_vertical_movement
        self.consistency_threshold = consistency_threshold

        # Track ID -> recent (x, y) points
        self.trajectories = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        # Track ID -> latest estimated direction
        self.directions = {}

        # Track ID -> movement consistency
        self.consistencies = {}

    def update(self, track_id: int, center_x: float, center_y: float):
        """
        Add a new position for a tracked object and
        estimate its current direction.
        """

        history = self.trajectories[track_id]

        history.append(
            (
                float(center_x),
                float(center_y),
            )
        )

        direction = self._estimate_direction(history)
        consistency = self._direction_consistency(history)

        self.directions[track_id] = direction
        self.consistencies[track_id] = consistency

        return {
            "track_id": track_id,
            "direction": direction,
            "consistency": consistency,
            "history": list(history),
        }

    def _estimate_direction(self, history):
        """
        Estimate direction from oldest to newest point.
        """

        if len(history) < self.min_history:
            return None

        points = np.array(
            history,
            dtype=np.float32,
        )

        y_values = points[:, 1]

        if len(y_values) >= 5:

            kernel = np.ones(
                5,
                dtype=np.float32,
            ) / 5.0

            smoothed_y = np.convolve(
                y_values,
                kernel,
                mode="valid",
            )

        else:
            smoothed_y = y_values

        start_y = float(smoothed_y[0])
        end_y = float(smoothed_y[-1])

        delta_y = end_y - start_y

        if abs(delta_y) < self.min_vertical_movement:
            return None

        if delta_y < 0:
            return "UP"

        return "DOWN"

    def _direction_consistency(self, history):
        """
        Measure how consistently the vehicle has been
        moving in one vertical direction.
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

    def get_direction(self, track_id: int):
        """
        Return latest direction for a track.
        """

        return self.directions.get(track_id)

    def get_consistency(self, track_id: int):
        """
        Return latest consistency value.
        """

        return self.consistencies.get(
            track_id,
            0.0,
        )

    def get_trajectory(self, track_id: int):
        """
        Return trajectory history.
        """

        return list(
            self.trajectories.get(
                track_id,
                [],
            )
        )

    def remove_track(self, track_id: int):
        """
        Remove a track from memory.
        """

        self.trajectories.pop(
            track_id,
            None,
        )

        self.directions.pop(
            track_id,
            None,
        )

        self.consistencies.pop(
            track_id,
            None,
        )