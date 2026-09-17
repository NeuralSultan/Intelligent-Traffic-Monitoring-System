from collections import Counter, deque
from typing import Optional


class TrafficLightManager:
    """
    Maintains the current traffic-light state using
    repeated Red Light / Green Light detections.

    The manager does not run YOLO.

    It receives detections from TrafficSignDetector
    and confirms the traffic-light state over multiple
    frames to reduce temporary false detections.
    """

    VALID_STATES = {
        "RED",
        "GREEN",
    }

    def __init__(
        self,
        confirmation_window: int = 15,
        min_confirmations: int = 5,
        change_min_confirmations: int = 8,
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

        if change_min_confirmations <= 0:
            raise ValueError(
                "change_min_confirmations must be greater "
                "than zero."
            )

        if change_min_confirmations > confirmation_window:
            raise ValueError(
                "change_min_confirmations cannot exceed "
                "confirmation_window."
            )

        self.confirmation_window = confirmation_window
        self.min_confirmations = min_confirmations
        self.change_min_confirmations = (
            change_min_confirmations
        )
        self.confidence_threshold = (
            confidence_threshold
        )

        self.detection_history = deque(
            maxlen=confirmation_window
        )

        self.current_state: Optional[str] = None

    def update(
        self,
        detections: list[dict],
    ) -> str:
        """
        Update traffic-light state using detections
        from the current frame.

        Returns:

            RED
            GREEN
            UNKNOWN
        """

        frame_states = []

        for detection in detections:

            confidence = detection.get(
                "confidence",
                0.0,
            )

            class_name = detection.get(
                "class_name",
                "",
            )

            if confidence < self.confidence_threshold:
                continue

            if class_name == "Red Light":
                frame_states.append("RED")

            elif class_name == "Green Light":
                frame_states.append("GREEN")

        # --------------------------------------------------
        # No valid traffic-light detection
        # --------------------------------------------------

        if not frame_states:
            return self.get_state()

        # --------------------------------------------------
        # Choose the state with the strongest presence
        # in this frame.
        # --------------------------------------------------

        counts = Counter(frame_states)

        detected_state, _ = counts.most_common(1)[0]

        self.detection_history.append(
            detected_state
        )

        self._update_confirmed_state()

        return self.get_state()

    def _update_confirmed_state(self):
        """
        Confirm the current traffic-light state using
        the recent detection history.
        """

        if not self.detection_history:
            return

        counts = Counter(
            self.detection_history
        )

        candidate_state, count = (
            counts.most_common(1)[0]
        )

        # --------------------------------------------------
        # First state
        # --------------------------------------------------

        if self.current_state is None:

            if count >= self.min_confirmations:

                self.current_state = (
                    candidate_state
                )

                print(
                    "[TRAFFIC LIGHT] "
                    f"Confirmed: {self.current_state}"
                )

            return

        # --------------------------------------------------
        # Same state
        # --------------------------------------------------

        if candidate_state == self.current_state:
            return

        # --------------------------------------------------
        # State change
        # --------------------------------------------------

        if count >= self.change_min_confirmations:

            previous_state = (
                self.current_state
            )

            self.current_state = (
                candidate_state
            )

            print(
                "[TRAFFIC LIGHT] "
                f"Changed: {previous_state} "
                f"-> {self.current_state}"
            )

    def get_state(self) -> str:
        """
        Return the currently confirmed state.

        Returns UNKNOWN until a state is confirmed.
        """

        if self.current_state is None:
            return "UNKNOWN"

        return self.current_state

    def reset(self):
        """
        Reset traffic-light state.
        """

        self.detection_history.clear()
        self.current_state = None