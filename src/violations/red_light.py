from typing import Optional

from src.core.base_detector import BaseViolationDetector
from src.core.schemas import VehicleState, ViolationEvent


class RedLightDetector(BaseViolationDetector):
    """
    Detects red-light violations using:

        Traffic-light state
        +
        Vehicle tracking
        +
        Stop-line crossing

    The detector does not run YOLO.
    """

    def __init__(
        self,
        stop_line,
        traffic_light_state_provider,
    ):
        if (
            stop_line is None
            or len(stop_line) != 2
        ):
            raise ValueError(
                "stop_line must contain two points."
            )

        self.stop_line = (
            tuple(stop_line[0]),
            tuple(stop_line[1]),
        )

        self.traffic_light_state_provider = (
            traffic_light_state_provider
        )

        # Track ID -> previous bottom-center position
        self.previous_positions = {}

        # Track IDs that already produced
        # a red-light violation.
        self.confirmed_violations = set()

    def _cross_product(
        self,
        a,
        b,
        c,
    ):
        return (
            (b[0] - a[0])
            * (c[1] - a[1])
            - (b[1] - a[1])
            * (c[0] - a[0])
        )

    def _crossed_line(
        self,
        previous_point,
        current_point,
    ):
        """
        Check whether the vehicle crossed
        the stop line between two frames.
        """

        a, b = self.stop_line

        previous_side = self._cross_product(
            a,
            b,
            previous_point,
        )

        current_side = self._cross_product(
            a,
            b,
            current_point,
        )

        return (
            (
                previous_side > 0
                and current_side < 0
            )
            or (
                previous_side < 0
                and current_side > 0
            )
            or previous_side == 0
            or current_side == 0
        )

    def process(
        self,
        vehicle: VehicleState,
        frame,
        frame_number: int,
        timestamp: float,
    ) -> Optional[ViolationEvent]:

        track_id = int(vehicle.track_id)

        # --------------------------------------------------
        # Get current traffic-light state
        # --------------------------------------------------

        traffic_light_state = (
            self.traffic_light_state_provider()
        )

        if traffic_light_state is None:
            traffic_light_state = "UNKNOWN"

        traffic_light_state = (
            traffic_light_state.upper()
        )

        if traffic_light_state not in {
            "RED",
            "GREEN",
            "UNKNOWN",
        }:
            traffic_light_state = "UNKNOWN"

        # --------------------------------------------------
        # Vehicle contact point with the road
        # --------------------------------------------------

        x1, y1, x2, y2 = vehicle.bbox

        current_point = (
            (float(x1) + float(x2)) / 2.0,
            float(y2),
        )

        previous_point = (
            self.previous_positions.get(
                track_id
            )
        )

        crossed_stop_line = False

        if previous_point is not None:
            crossed_stop_line = self._crossed_line(
                previous_point,
                current_point,
            )

        self.previous_positions[
            track_id
        ] = current_point

        # --------------------------------------------------
        # Red-light violation
        # --------------------------------------------------

        if (
            traffic_light_state == "RED"
            and crossed_stop_line
            and track_id not in self.confirmed_violations
        ):

            self.confirmed_violations.add(
                track_id
            )

            return ViolationEvent(
                violation_type="red_light",
                vehicle_id=track_id,
                frame_number=frame_number,
                timestamp=timestamp,
                class_name=vehicle.class_name,
                severity="high",
                confidence=vehicle.confidence,
                details={
                    "traffic_light_state": (
                        traffic_light_state
                    ),
                    "stop_line": [
                        list(self.stop_line[0]),
                        list(self.stop_line[1]),
                    ],
                    "bbox": list(vehicle.bbox),
                },
            )

        return None

    def get_violations(self):
        return set(
            self.confirmed_violations
        )

    def get_stop_line(self):
        return self.stop_line

    def remove_track(
        self,
        track_id: int,
    ):
        self.previous_positions.pop(
            int(track_id),
            None,
        )

    def reset(self):
        self.previous_positions.clear()
        self.confirmed_violations.clear()