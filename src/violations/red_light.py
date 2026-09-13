from collections import defaultdict


class RedLightDetector:
    """
    Detects red-light violations from:

    1. Traffic-light state
    2. Vehicle tracking
    3. Stop-line crossing

    This module does NOT run YOLO.
    It receives already processed track information
    from the main pipeline.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        confirmation_frames: int = 5,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.confirmation_frames = (
            confirmation_frames
        )

        # ----------------------------------------------------
        # Stop line
        # ----------------------------------------------------
        #
        # Starter configuration:
        # horizontal line at 70% of frame height.
        #
        # We will make this configurable later.
        #

        self.stop_line_y = int(
            frame_height * 0.70
        )

        # Track ID -> previous Y position
        self.previous_positions = {}

        # Track ID -> consecutive red-light violation frames
        self.violation_streak = defaultdict(int)

        # Confirmed violation IDs
        self.confirmed_violations = set()

    def update(
        self,
        track_id: int,
        center_y: float,
        traffic_light_state: str,
    ):
        """
        Update one tracked vehicle.

        traffic_light_state must be one of:
            RED
            YELLOW
            GREEN
            UNKNOWN
        """

        track_id = int(track_id)
        center_y = float(center_y)

        state = traffic_light_state.upper()

        if state not in {
            "RED",
            "YELLOW",
            "GREEN",
            "UNKNOWN",
        }:
            state = "UNKNOWN"

        previous_y = self.previous_positions.get(
            track_id
        )

        crossed_stop_line = False

        if previous_y is not None:

            # Vehicle moving downward across line
            if (
                previous_y < self.stop_line_y
                and center_y >= self.stop_line_y
            ):
                crossed_stop_line = True

        self.previous_positions[track_id] = center_y

        # ----------------------------------------------------
        # Red-light condition
        # ----------------------------------------------------

        is_red_violation = (
            state == "RED"
            and crossed_stop_line
        )

        if is_red_violation:

            self.violation_streak[track_id] += 1

        else:

            self.violation_streak[track_id] = 0

        # ----------------------------------------------------
        # Confirm violation
        # ----------------------------------------------------

        confirmed = (
            self.violation_streak[track_id]
            >= self.confirmation_frames
        )

        if confirmed:

            self.confirmed_violations.add(
                track_id
            )

        return {
            "track_id": track_id,
            "traffic_light_state": state,
            "crossed_stop_line": crossed_stop_line,
            "red_light_violation": is_red_violation,
            "streak": self.violation_streak[track_id],
            "confirmed": confirmed,
        }

    def get_violations(self):
        """
        Return confirmed red-light violations.
        """

        return set(
            self.confirmed_violations
        )

    def get_stop_line_y(self):
        """
        Return stop-line position.
        """

        return self.stop_line_y

    def remove_track(self, track_id: int):
        """
        Remove finished track from memory.
        """

        self.previous_positions.pop(
            track_id,
            None,
        )

        self.violation_streak.pop(
            track_id,
            None,
        )