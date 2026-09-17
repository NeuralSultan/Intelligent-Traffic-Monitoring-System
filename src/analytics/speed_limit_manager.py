from collections import Counter, deque
from typing import Optional


class SpeedLimitManager:
    """
    Maintains and confirms the speed limit for the current video.

    The first reliable speed-limit detection is confirmed after
    several consistent detections. Once confirmed, the speed
    limit remains fixed for the rest of the video.
    """

    def __init__(
        self,
        confirmation_window: int = 15,
        min_confirmations: int = 5,
        confidence_threshold: float = 0.50,
    ):
        if confirmation_window <= 0:
            raise ValueError(
                "confirmation_window must be greater than zero."
            )

        if min_confirmations <= 0:
            raise ValueError(
                "min_confirmations must be greater than zero."
            )

        if min_confirmations > confirmation_window:
            raise ValueError(
                "min_confirmations cannot exceed "
                "confirmation_window."
            )

        self.confirmation_window = confirmation_window
        self.min_confirmations = min_confirmations
        self.confidence_threshold = confidence_threshold

        self.detection_history = deque(
            maxlen=confirmation_window
        )

        self.current_speed_limit: Optional[int] = None

    def _extract_speed_limit(
        self,
        class_name: str,
    ) -> Optional[int]:
        """
        Convert:

            'Speed Limit 60'

        into:

            60
        """

        if not class_name.startswith("Speed Limit"):
            return None

        parts = class_name.split()

        if not parts:
            return None

        try:
            return int(parts[-1])
        except ValueError:
            return None

    def update(
        self,
        detections: list[dict],
    ) -> Optional[int]:
        """
        Update the speed-limit manager using detections
        from the current frame.
        """

        # Once a speed limit has been confirmed,
        # keep it fixed for this video.
        if self.current_speed_limit is not None:
            return self.current_speed_limit

        frame_limits = []

        for detection in detections:

            confidence = detection.get(
                "confidence",
                0.0,
            )

            class_name = detection.get(
                "class_name",
                "",
            )

            if class_name.startswith("Speed Limit"):
                print(
                    f"[SIGN] {class_name} "
                    f"confidence={confidence:.2f}"
                )

            if confidence < self.confidence_threshold:
                continue

            speed_limit = self._extract_speed_limit(
                class_name
            )

            if speed_limit is None:
                continue

            frame_limits.append(speed_limit)

        if frame_limits:

            counts = Counter(frame_limits)

            detected_limit, _ = (
                counts.most_common(1)[0]
            )

            self.detection_history.append(
                detected_limit
            )

        self._update_confirmed_limit()

        return self.current_speed_limit

    def _update_confirmed_limit(self):
        """
        Confirm the first reliable speed limit.

        Once confirmed, the limit remains fixed until
        reset() is called.
        """

        if self.current_speed_limit is not None:
            return

        if not self.detection_history:
            return

        counts = Counter(
            self.detection_history
        )

        candidate_limit, count = (
            counts.most_common(1)[0]
        )

        if count >= self.min_confirmations:

            self.current_speed_limit = (
                candidate_limit
            )

            print(
                f"[SPEED LIMIT] Confirmed: "
                f"{candidate_limit} km/h"
            )

    def get_speed_limit(self) -> Optional[int]:
        """
        Return the currently confirmed speed limit.
        """

        return self.current_speed_limit

    def reset(self):
        """
        Clear detection history and current limit.
        """

        self.detection_history.clear()

        self.current_speed_limit = None

