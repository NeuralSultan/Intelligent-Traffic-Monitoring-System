import json
from pathlib import Path


class RoadConfigLoader:
    """
    Loads and validates road configuration.

    Supports:
        - One-way roads
        - Two-way roads
        - Road direction calibration
        - Perspective calibration
        - Multiple camera/scene configurations

    Current two-way calibration schema:

        "direction": {
            "left_side": "DOWN",
            "right_side": "UP",
            "left_confidence": 1.0,
            "right_confidence": 1.0,
            "left_tracks": 13,
            "right_tracks": 16
        }
    """

    VALID_DIRECTIONS = {"UP", "DOWN"}
    VALID_ROAD_TYPES = {"one_way", "two_way"}

    def get_side_x(self, side: str):
        """
        Return the representative calibrated X position
        for a road side.
        """

        side = side.upper()

        direction_config = self._get_direction_config()

        if side == "LEFT":
            return direction_config.get(
                "left_x"
            )

        if side == "RIGHT":
            return direction_config.get(
                "right_x"
            )

        raise ValueError(
            "side must be 'LEFT' or 'RIGHT'."
        )
        
    def __init__(
        self,
        config_path: str | Path,
        scene_id: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.scene_id = scene_id

        if scene_id is not None:
            self.config_path = (
                self.config_path.parent
                / "calibrations"
                / f"{scene_id}.json"
            )

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

    # =========================================================
    # Road Type
    # =========================================================

    def get_road_type(self) -> str:
        """
        Return the road type.

        Supported:
            - one_way
            - two_way
        """

        road_type = self.config.get(
            "road_type",
            "two_way",
        )

        if road_type not in self.VALID_ROAD_TYPES:
            raise ValueError(
                f"Invalid road_type: {road_type}. "
                f"Expected one of: "
                f"{self.VALID_ROAD_TYPES}"
            )

        return road_type

    def is_one_way(self) -> bool:
        return self.get_road_type() == "one_way"

    def is_two_way(self) -> bool:
        return self.get_road_type() == "two_way"

    # =========================================================
    # One-Way Direction
    # =========================================================

    def get_allowed_direction(self):
        """
        Return the allowed direction for a one-way road.
        """

        if not self.is_one_way():
            return None

        direction = self.config.get(
            "allowed_direction"
        )

        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                "Invalid or missing "
                "'allowed_direction' for one-way road."
            )

        return direction

    # =========================================================
    # Two-Way Direction Data
    # =========================================================

    def _get_direction_config(self):
        """
        Return the direction calibration section.

        Current schema:

            config["direction"]

        This method keeps the schema access in one place.
        """

        direction_config = self.config.get(
            "direction"
        )

        if not isinstance(
            direction_config,
            dict,
        ):
            raise ValueError(
                "Missing or invalid "
                "'direction' section in road configuration."
            )

        return direction_config

    # =========================================================
    # Road Direction Calibration
    # =========================================================

    def has_road_calibration(self) -> bool:
        """
        Check whether road direction calibration
        is valid for the configured road type.
        """

        if not self.config.get(
            "calibration_valid",
            False,
        ):
            return False

        road_type = self.get_road_type()

        # -----------------------------------------------------
        # One-way road
        # -----------------------------------------------------

        if road_type == "one_way":

            direction = self.config.get(
                "allowed_direction"
            )

            return (
                direction
                in self.VALID_DIRECTIONS
            )

        # -----------------------------------------------------
        # Two-way road
        # -----------------------------------------------------

        try:
            direction_config = (
                self._get_direction_config()
            )
        except ValueError:
            return False

        left_direction = direction_config.get(
            "left_side"
        )

        right_direction = direction_config.get(
            "right_side"
        )

        return (
            left_direction
            in self.VALID_DIRECTIONS
            and right_direction
            in self.VALID_DIRECTIONS
        )

    def validate_road_calibration(self):
        """
        Validate road direction calibration
        according to the road type.
        """

        if not self.config.get(
            "calibration_valid",
            False,
        ):
            raise ValueError(
                "Road direction calibration is "
                "marked as invalid."
            )

        road_type = self.get_road_type()

        # -----------------------------------------------------
        # One-way road
        # -----------------------------------------------------

        if road_type == "one_way":

            direction = self.config.get(
                "allowed_direction"
            )

            if direction not in self.VALID_DIRECTIONS:
                raise ValueError(
                    "Invalid or missing "
                    "'allowed_direction' for "
                    "one-way road."
                )

            return

        # -----------------------------------------------------
        # Two-way road
        # -----------------------------------------------------

        direction_config = (
            self._get_direction_config()
        )

        left_direction = direction_config.get(
            "left_side"
        )

        right_direction = direction_config.get(
            "right_side"
        )

        if left_direction is None:
            raise ValueError(
                "Missing 'left_side' "
                "in road configuration."
            )

        if right_direction is None:
            raise ValueError(
                "Missing 'right_side' "
                "in road configuration."
            )

        if left_direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction for "
                f"left_side: {left_direction}"
            )

        if right_direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction for "
                f"right_side: {right_direction}"
            )

    def get_expected_direction(
        self,
        side: str,
    ):
        """
        Return expected direction for LEFT or RIGHT.

        Only applicable to two-way roads.
        """

        if self.is_one_way():
            return self.get_allowed_direction()

        self.validate_road_calibration()

        direction_config = (
            self._get_direction_config()
        )

        side = side.upper()

        if side == "LEFT":
            return direction_config[
                "left_side"
            ]

        if side == "RIGHT":
            return direction_config[
                "right_side"
            ]

        raise ValueError(
            f"Invalid side: {side}. "
            "Expected LEFT or RIGHT."
        )

    def get_confidence(
        self,
        side: str,
    ):
        """
        Return road calibration confidence.
        """

        if self.is_one_way():
            return (
                1.0
                if self.get_allowed_direction()
                else 0.0
            )

        self.validate_road_calibration()

        direction_config = (
            self._get_direction_config()
        )

        side = side.upper()

        if side == "LEFT":
            return float(
                direction_config.get(
                    "left_confidence",
                    0.0,
                )
            )

        if side == "RIGHT":
            return float(
                direction_config.get(
                    "right_confidence",
                    0.0,
                )
            )

        return 0.0

    def get_tracks_used(
        self,
        side: str,
    ):
        """
        Return number of tracks used for
        road direction calibration.
        """

        if self.is_one_way():
            return 0

        self.validate_road_calibration()

        direction_config = (
            self._get_direction_config()
        )

        side = side.upper()

        if side == "LEFT":
            return int(
                direction_config.get(
                    "left_tracks",
                    0,
                )
            )

        if side == "RIGHT":
            return int(
                direction_config.get(
                    "right_tracks",
                    0,
                )
            )

        return 0

    # =========================================================
    # Video Information
    # =========================================================

    def get_video_dimensions(self):
        """
        Return dimensions stored during calibration.
        """

        return (
            self.config.get("video_width"),
            self.config.get("video_height"),
        )

    def matches_video(
        self,
        video_path: str | Path,
        width: int,
        height: int,
        road_type: str | None = None,
    ) -> bool:

        video_path = Path(video_path)

        saved_video_name = self.config.get(
            "video_name"
        )

        saved_width, saved_height = (
            self.get_video_dimensions()
        )

        video_matches = (
            saved_video_name == video_path.name
            and saved_width == width
            and saved_height == height
            and self.config.get(
                "calibration_valid",
                False,
            )
        )

        if not video_matches:
            return False

        if road_type is not None:

            saved_road_type = self.config.get(
                "road_type"
            )

            if saved_road_type != road_type:
                return False

        return True

    # =========================================================
    # Perspective Calibration
    # =========================================================

    def has_perspective_calibration(self) -> bool:
        """
        Check whether perspective calibration exists.
        """

        perspective = self.config.get(
            "perspective"
        )

        if not perspective:
            return False

        return perspective.get(
            "calibration_valid",
            False,
        )

    def get_perspective(self):
        """
        Return complete perspective calibration.
        """

        if not self.has_perspective_calibration():
            raise ValueError(
                "Perspective calibration is not available."
            )

        return self.config[
            "perspective"
        ]

    def get_source_points(self):
        """
        Return the four source points.
        """

        perspective = self.get_perspective()

        return perspective[
            "source_points"
        ]

    def get_destination_points(self):
        """
        Return the four destination points.
        """

        perspective = self.get_perspective()

        return perspective[
            "destination_points"
        ]

    def get_homography_matrix(self):
        """
        Return the saved homography matrix.
        """

        perspective = self.get_perspective()

        return perspective[
            "homography_matrix"
        ]

    def get_real_dimensions(self):
        """
        Return real-world dimensions.

        Returns:
            (width_meters, height_meters)
        """

        perspective = self.get_perspective()

        return (
            perspective.get(
                "real_width_meters"
            ),
            perspective.get(
                "real_height_meters"
            ),
        )

    # =========================================================
    # Full Configuration
    # =========================================================

    def get_all(self):
        """
        Return the full configuration.
        """

        return self.config

