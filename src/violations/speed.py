from collections import defaultdict, deque
from typing import Optional

from src.core.base_detector import BaseViolationDetector
from src.core.schemas import VehicleState, ViolationEvent


class SpeedViolationDetector(BaseViolationDetector):
    """
    Detects speeding vehicles by comparing their measured
    speed against the currently confirmed road speed limit.

    The detector requires the vehicle to exceed the speed
    limit consistently for several measurements before
    creating a violation event.
    """

    def __init__(
        self,
        speed_limit_manager,
        tolerance_kmh: float = 5.0,
        confirmation_frames: int = 5,
        min_speed_kmh: float = 10.0,
        severity_thresholds: Optional[dict] = None,
    ):
        self.speed_limit_manager = speed_limit_manager

        self.tolerance_kmh = tolerance_kmh
        self.confirmation_frames = confirmation_frames
        self.min_speed_kmh = min_speed_kmh

        if severity_thresholds is None:
            severity_thresholds = {
                "medium": 10.0,
                "high": 20.0,
            }

        self.severity_thresholds = (
            severity_thresholds
        )

        self.violation_streaks = defaultdict(int)

        self.max_observed_speed = {}
        self.triggered_ids = set()

    def _get_severity(
        self,
        excess_speed: float,
    ) -> str:
        """
        Determine violation severity based on how much
        the vehicle exceeds the speed limit.
        """

        if (
            excess_speed
            >= self.severity_thresholds["high"]
        ):
            return "high"

        if (
            excess_speed
            >= self.severity_thresholds["medium"]
        ):
            return "medium"

        return "low"

    def process(
        self,
        vehicle: VehicleState,
        frame,
        frame_number: int,
        timestamp: float,
    ) -> Optional[ViolationEvent]:
        """
        Process one tracked vehicle.

        The vehicle object is expected to contain the
        current speed inside vehicle.details["speed_kmh"].
        """

        speed_limit = (
            self.speed_limit_manager.get_speed_limit()
        )
        
        # No confirmed speed limit yet.
        if speed_limit is None:
            self.violation_streaks[
                vehicle.track_id
            ] = 0

            return None

        speed_kmh = vehicle.speed_kmh

        # No valid speed measurement.
        if speed_kmh is None:
            return None

        speed_kmh = float(speed_kmh)

        # Ignore very low speeds.
        if speed_kmh < self.min_speed_kmh:
            self.violation_streaks[
                vehicle.track_id
            ] = 0

            return None

        # Track maximum observed speed.
        previous_max = self.max_observed_speed.get(
            vehicle.track_id,
            0.0,
        )

        self.max_observed_speed[
            vehicle.track_id
        ] = max(
            previous_max,
            speed_kmh,
        )

        # Allow a small tolerance to prevent borderline
        # violations caused by speed estimation noise.
        violation_threshold = (
            speed_limit
            + self.tolerance_kmh
        )

        if speed_kmh > violation_threshold:

            self.violation_streaks[
                vehicle.track_id
            ] += 1

        else:

            # Decay instead of immediately resetting.
            self.violation_streaks[
                vehicle.track_id
            ] = max(
                0,
                self.violation_streaks[
                    vehicle.track_id
                ] - 1,
            )

            return None

        # Require consistent speeding.
        if (
            self.violation_streaks[
                vehicle.track_id
            ]
            < self.confirmation_frames
        ):
            return None

        excess_speed = (
            speed_kmh - speed_limit
        )

        severity = self._get_severity(
            excess_speed
        )
        if vehicle.track_id in self.triggered_ids:
            return None
        
        self.triggered_ids.add(vehicle.track_id)

        return ViolationEvent(
            violation_type="speeding",
            vehicle_id=vehicle.track_id,
            frame_number=frame_number,
            timestamp=timestamp,
            class_name=vehicle.class_name,
            severity=severity,
            confidence=vehicle.confidence,
            details={
                "speed_kmh": round(
                    speed_kmh,
                    2,
                ),
                "speed_limit_kmh": speed_limit,
                "excess_speed_kmh": round(
                    excess_speed,
                    2,
                ),
                "max_observed_speed_kmh": round(
                    self.max_observed_speed[
                        vehicle.track_id
                    ],
                    2,
                ),
            },
        )

    def reset_track(
        self,
        track_id: int,
    ):
        """
        Remove state for a specific vehicle.
        """

        self.violation_streaks.pop(
            track_id,
            None,
        )

        self.max_observed_speed.pop(
            track_id,
            None,
        )
        self.triggered_ids.discard(
        track_id
    )

    def reset(self):
        """
        Reset all detector state.
        """

        self.violation_streaks.clear()
        self.max_observed_speed.clear()
        self.triggered_ids.clear()