from typing import Optional

from src.core.schemas import ViolationEvent


class ViolationManager:
    """
    Collects and manages all detected traffic violations.

    The manager is responsible for:
        - Storing violation events
        - Preventing duplicate events
        - Providing summaries and exports

    It does not detect violations itself.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        self.events: list[ViolationEvent] = []

        # Prevent duplicate events for the same
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
        severity: str = "medium",
        confidence: Optional[float] = None,
        details: Optional[dict] = None,
        evidence_path: Optional[str] = None,
    ) -> Optional[ViolationEvent]:
        """
        Create and store a violation event.

        The same vehicle cannot generate the same
        violation type more than once per run.
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
            severity=severity,
            confidence=confidence,
            details=details or {},
            evidence_path=evidence_path,
        )

        self.events.append(event)

        return event

    def get_events(self) -> list[ViolationEvent]:
        """
        Return all recorded violation events.
        """
        return list(self.events)

    def get_events_by_type(
        self,
        violation_type: str,
    ) -> list[ViolationEvent]:
        """
        Return violations of a specific type.
        """
        return [
            event
            for event in self.events
            if event.violation_type == violation_type
        ]

    def count(
        self,
        violation_type: Optional[str] = None,
    ) -> int:
        """
        Return total number of violations or
        violations of a specific type.
        """

        if violation_type is None:
            return len(self.events)

        return len(
            self.get_events_by_type(
                violation_type
            )
        )

    def summary(self) -> dict:
        """
        Return a simple violation summary.
        """

        summary = {}

        for event in self.events:
            violation_type = event.violation_type

            summary[violation_type] = (
                summary.get(
                    violation_type,
                    0,
                )
                + 1
            )

        return summary

    def export_dict(self) -> list[dict]:
        """
        Convert all events into dictionaries.

        Useful later for JSON/CSV/HTML reports.
        """

        return [
            {
                "violation_type": event.violation_type,
                "vehicle_id": event.vehicle_id,
                "frame_number": event.frame_number,
                "timestamp": event.timestamp,
                "class_name": event.class_name,
                "direction": event.direction,
                "expected_direction": event.expected_direction,
                "severity": event.severity,
                "confidence": event.confidence,
                "details": event.details,
                "evidence_path": event.evidence_path,
            }
            for event in self.events
        ]

    def clear(self):
        """
        Clear all stored events.
        """

        self.events.clear()
        self._event_keys.clear()