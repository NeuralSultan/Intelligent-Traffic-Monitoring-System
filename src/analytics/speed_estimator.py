from collections import defaultdict
from typing import Optional

import numpy as np


class TwoLineSpeedEstimator:
    """
    Estimates vehicle speed using two virtual line segments.

    A vehicle crosses:
        Line A -> Line B

    If the real-world distance between the two lines is known,
    speed can be calculated using:

        speed = distance / elapsed_time
    """

    def __init__(
        self,
        line_a,
        line_b,
        distance_meters: float,
        fps: float,
        max_speed_kmh: float = 200.0,
    ):
        if fps <= 0:
            raise ValueError(
                "FPS must be greater than zero."
            )

        if distance_meters <= 0:
            raise ValueError(
                "distance_meters must be greater than zero."
            )

        self.line_a = line_a
        self.line_b = line_b

        self.distance_meters = float(
            distance_meters
        )

        self.fps = float(fps)
        self.max_speed_kmh = float(
            max_speed_kmh
        )

        self.previous_positions = {}

        self.line_a_frames = {}

        self.speed_history = defaultdict(list)

    def _get_bottom_center(self, bbox):
        x1, y1, x2, y2 = bbox

        center_x = (
            float(x1) + float(x2)
        ) / 2.0

        bottom_y = float(y2)

        return center_x, bottom_y

    def _cross_product(
        self,
        a,
        b,
        c,
    ):
        """
        Determines which side of line AB
        point C lies on.
        """

        return (
            (b[0] - a[0])
            * (c[1] - a[1])
            -
            (b[1] - a[1])
            * (c[0] - a[0])
        )

    def _crossed_line(
        self,
        previous_point,
        current_point,
        line,
    ):
        """
        Checks whether movement from
        previous_point to current_point
        crossed the line segment.

        The bounding box of the line is used
        as a simple intersection test.
        """

        a, b = line

        prev_side = self._cross_product(
            a,
            b,
            previous_point,
        )

        current_side = self._cross_product(
            a,
            b,
            current_point,
        )

        # Same side -> no crossing
        if (
            prev_side == 0
            or current_side == 0
            or (
                prev_side > 0
                and current_side < 0
            )
            or (
                prev_side < 0
                and current_side > 0
            )
        ):
            min_x = min(a[0], b[0])
            max_x = max(a[0], b[0])

            min_y = min(a[1], b[1])
            max_y = max(a[1], b[1])

            # Expand slightly to account for
            # movement between frames.
            margin = 20

            min_x -= margin
            max_x += margin
            min_y -= margin
            max_y += margin

            if (
                min_x
                <= current_point[0]
                <= max_x
                and
                min_y
                <= current_point[1]
                <= max_y
            ):
                return True

        return False

    def update(
        self,
        track_id: int,
        bbox,
        frame_number: int,
    ) -> Optional[float]:

        current_point = (
            self._get_bottom_center(bbox)
        )

        previous_point = (
            self.previous_positions.get(
                track_id
            )
        )

        if previous_point is None:
            self.previous_positions[
                track_id
            ] = current_point

            return None

        crossed_a = self._crossed_line(
            previous_point,
            current_point,
            self.line_a,
        )

        if (
            crossed_a
            and track_id
            not in self.line_a_frames
        ):
            self.line_a_frames[
                track_id
            ] = frame_number

        crossed_b = self._crossed_line(
            previous_point,
            current_point,
            self.line_b,
        )

        speed_kmh = None

        if crossed_b:

            line_a_frame = (
                self.line_a_frames.get(
                    track_id
                )
            )

            if line_a_frame is not None:

                frame_delta = (
                    frame_number
                    - line_a_frame
                )

                if frame_delta > 0:

                    elapsed_seconds = (
                        frame_delta
                        / self.fps
                    )

                    speed_mps = (
                        self.distance_meters
                        / elapsed_seconds
                    )

                    calculated_speed = (
                        speed_mps * 3.6
                    )

                    if (
                        0
                        < calculated_speed
                        <= self.max_speed_kmh
                    ):
                        speed_kmh = float(
                            calculated_speed
                        )

                        self.speed_history[
                            track_id
                        ].append(
                            speed_kmh
                        )

                self.line_a_frames.pop(
                    track_id,
                    None,
                )

        self.previous_positions[
            track_id
        ] = current_point

        return speed_kmh

    def get_speed(self, track_id):
        speeds = (
            self.speed_history.get(
                track_id
            )
        )

        if not speeds:
            return None

        return float(
            np.mean(speeds)
        )

    def remove_track(self, track_id):

        self.previous_positions.pop(
            track_id,
            None,
        )

        self.line_a_frames.pop(
            track_id,
            None,
        )

        self.speed_history.pop(
            track_id,
            None,
        )

    def clear(self):

        self.previous_positions.clear()
        self.line_a_frames.clear()
        self.speed_history.clear()