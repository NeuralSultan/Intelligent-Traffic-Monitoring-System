from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ViolationEvent:
    """
    Standard structure for every traffic violation.
    """

    violation_type: str
    vehicle_id: Optional[int]

    frame_number: int
    timestamp: float

    class_name: Optional[str] = None

    direction: Optional[str] = None
    expected_direction: Optional[str] = None

    confidence: Optional[float] = None

    details: Optional[dict] = None


class ViolationManager:
    """
    Collects and manages all detected traffic violations.

    Every future detector should report violations through
    this manager instead of printing or storing them separately.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        self.events = []

        # Used to prevent duplicate events for the same
        # vehicle and violation type.
        self._event_keys = set()

    def add_violation(
        self,
        violation_type: str,
        vehicle_id: Optional[int],
        frame_number: int,
        class_name: Optional[str] = None,
        direction: Optional[str] = None,
        expected_direction: Optional[str] = None,
        confidence: Optional[float] = None,
        details: Optional[dict] = None,
    ):
        """
        Add a violation event.

        The same vehicle cannot generate the same violation
        more than once during a single run.
        """

        event_key = (
            violation_type,
            vehicle_id,
        )

        if event_key in self._event_keys:
            return None

        self._event_keys.add(event_key)

        timestamp = (
            frame_number / self.fps
            if self.fps > 0
            else 0.0
        )

        event = ViolationEvent(
            violation_type=violation_type,
            vehicle_id=vehicle_id,
            frame_number=frame_number,
            timestamp=timestamp,
            class_name=class_name,
            direction=direction,
            expected_direction=expected_direction,
            confidence=confidence,
            details=details,
        )

        self.events.append(event)

        return event

    def get_events(self):
        """
        Return all recorded violation events.
        """

        return list(self.events)

    def get_events_by_type(
        self,
        violation_type: str,
    ):
        """
        Return violations of a specific type.
        """

        return [
            event
            for event in self.events
            if event.violation_type
            == violation_type
        ]

    def count(
        self,
        violation_type: Optional[str] = None,
    ):
        """
        Return total number of violations or violations
        of a specific type.
        """

        if violation_type is None:
            return len(self.events)

        return len(
            self.get_events_by_type(
                violation_type
            )
        )

    def summary(self):
        """
        Return a simple violation summary.
        """

        summary = {}

        for event in self.events:

            violation_type = (
                event.violation_type
            )

            summary[violation_type] = (
                summary.get(
                    violation_type,
                    0,
                )
                + 1
            )

        return summary

    def export_dict(self):
        """
        Convert all events into dictionaries.

        Useful later for JSON/CSV/HTML reports.
        """

        return [
            asdict(event)
            for event in self.events
        ]

    def clear(self):
        """
        Clear all stored events.
        """

        self.events.clear()
        self._event_keys.clear()