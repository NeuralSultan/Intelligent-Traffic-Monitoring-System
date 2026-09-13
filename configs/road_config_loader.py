import json
from pathlib import Path


class RoadConfigLoader:
    """
    Loads and validates the automatically generated
    road configuration.
    """

    VALID_DIRECTIONS = {"UP", "DOWN"}

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Road configuration not found:\n"
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self._validate()

    def _validate(self):
        """
        Validate required fields.
        """

        if not self.config.get(
            "calibration_valid",
            False,
        ):
            raise ValueError(
                "Road calibration is marked as invalid."
            )

        for side in ("left_side", "right_side"):

            if side not in self.config:
                raise ValueError(
                    f"Missing '{side}' in road configuration."
                )

            direction = self.config[side].get(
                "expected_direction"
            )

            if direction not in self.VALID_DIRECTIONS:
                raise ValueError(
                    f"Invalid direction for {side}: "
                    f"{direction}"
                )

    def get_expected_direction(
        self,
        side: str,
    ):
        """
        Return expected direction for LEFT or RIGHT.
        """

        side = side.upper()

        if side == "LEFT":
            return self.config["left_side"][
                "expected_direction"
            ]

        if side == "RIGHT":
            return self.config["right_side"][
                "expected_direction"
            ]

        return None

    def get_confidence(
        self,
        side: str,
    ):
        """
        Return calibration confidence.
        """

        side = side.upper()

        if side == "LEFT":
            return self.config["left_side"].get(
                "confidence",
                0.0,
            )

        if side == "RIGHT":
            return self.config["right_side"].get(
                "confidence",
                0.0,
            )

        return 0.0

    def get_tracks_used(
        self,
        side: str,
    ):
        """
        Return number of tracks used for calibration.
        """

        side = side.upper()

        if side == "LEFT":
            return self.config["left_side"].get(
                "tracks_used",
                0,
            )

        if side == "RIGHT":
            return self.config["right_side"].get(
                "tracks_used",
                0,
            )

        return 0

    def get_video_dimensions(self):
        """
        Return dimensions stored during calibration.
        """

        return (
            self.config.get("video_width"),
            self.config.get("video_height"),
        )

    def get_all(self):
        """
        Return the full configuration.
        """

        return self.config