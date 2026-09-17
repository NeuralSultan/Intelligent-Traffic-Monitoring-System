import cv2
from pathlib import Path

from src.tracking.tracker import TrafficTracker

from src.analytics.vehicle_counter import VehicleCounter
from src.analytics.direction import DirectionEngine
from src.analytics.road_calibrator import RoadCalibrator
from src.analytics.speed_estimator import TwoLineSpeedEstimator
from src.analytics.speed_line_selector import SpeedLineSelector
from src.analytics.speed_limit_manager import SpeedLimitManager
from src.analytics.stop_line_selector import StopLineSelector
from src.analytics.traffic_light_manager import TrafficLightManager

from src.detection.traffic_sign_detector import TrafficSignDetector

from src.violations.wrong_way import WrongWayDetector
from src.violations.speed import SpeedViolationDetector
from src.violations.red_light import RedLightDetector
from src.violations.violation_manager import ViolationManager

from src.core.schemas import VehicleState
from src.core.base_detector import BaseViolationDetector

from configs.road_config_loader import RoadConfigLoader

from src.evidence.evidence_collector import EvidenceCollector

from configs.traffic_config import (
    MODEL_PATH,
    CONFIDENCE,
    IMAGE_SIZE,
    TRACKER,
    VEHICLE_CLASSES,
    ROAD_CONFIG_PATH,
    EVIDENCE_PATH,
    SPEED_REFERENCE_DISTANCE_M,
)


class TrafficPipeline:
    """
    Main traffic monitoring pipeline.

    Flow:

        Video
          ↓
        YOLO Vehicle Detection
          ↓
        ByteTrack
          ↓
        VehicleState
          ↓
        Analytics
          ├── Vehicle Counter
          ├── Direction
          └── Speed Estimation
          ↓
        Traffic Sign / Light Detection
          ├── Speed Limit Manager
          └── Traffic Light Manager
          ↓
        Violation Detectors
          ├── Wrong Way
          ├── Speeding
          └── Red Light
          ↓
        Evidence Collector
          ↓
        Violation Manager
    """

    def __init__(
        self,
        video_path: str,
        model_path: str = MODEL_PATH,
        road_type: str = "one_way",
        enabled_violations: list[str] | None = None,
    ):

        self.video_path = video_path
        self.model_path = model_path
        self.frame_number = 0
        self.road_type = road_type

        # --------------------------------------------------
        # Track lifecycle
        # --------------------------------------------------

        self.track_missed_frames = {}
        self.track_cleanup_threshold = 35

        self.enabled_violations = (
            enabled_violations
            if enabled_violations is not None
            else [
                "wrong_way",
                "speeding",
                "red_light",
            ]
        )

        valid_violations = {
            "wrong_way",
            "speeding",
            "red_light",
        }

        invalid_violations = (
            set(self.enabled_violations)
            - valid_violations
        )

        if invalid_violations:
            raise ValueError(
                f"Invalid violations: "
                f"{sorted(invalid_violations)}. "
                f"Valid options: "
                f"{sorted(valid_violations)}"
            )

        # --------------------------------------------------
        # Evidence
        # --------------------------------------------------

        self.evidence_collector = EvidenceCollector(
            output_dir=EVIDENCE_PATH
        )

        # --------------------------------------------------
        # Vehicle Tracking
        # --------------------------------------------------

        self.tracker = TrafficTracker(
            model_path=self.model_path,
            confidence=CONFIDENCE,
            image_size=IMAGE_SIZE,
            tracker_config=TRACKER,
        )

        # --------------------------------------------------
        # Traffic Sign / Light Detection
        # --------------------------------------------------

        traffic_sign_model_path = (
            Path(self.model_path).parent.parent
            / "traffic_sign"
            / "best.pt"
        )

        self.traffic_sign_detector = None

        if (
            "speeding" in self.enabled_violations
            or "red_light" in self.enabled_violations
        ):

            self.traffic_sign_detector = (
                TrafficSignDetector(
                    model_path=traffic_sign_model_path,
                    confidence=0.35,
                    image_size=640,
                )
            )

        # --------------------------------------------------
        # Analytics
        # --------------------------------------------------

        self.counter = None
        self.direction_engine = None
        self.speed_estimator = None

        # --------------------------------------------------
        # Speed Lines
        # --------------------------------------------------

        self.speed_line_a = None
        self.speed_line_b = None

        # --------------------------------------------------
        # Stop Line
        # --------------------------------------------------

        self.stop_line = None

        # --------------------------------------------------
        # Speed Limit
        # --------------------------------------------------

        self.speed_limit_manager = None
        self.speed_violation_detector = None

        # --------------------------------------------------
        # Traffic Light
        # --------------------------------------------------

        self.traffic_light_manager = None
        self.red_light_detector = None

        # --------------------------------------------------
        # Road Configuration
        # --------------------------------------------------

        self.road_config = None

        # --------------------------------------------------
        # Violation System
        # --------------------------------------------------

        self.violation_manager = None
        self.wrong_way_detector = None

        self.violation_detectors: list[
            BaseViolationDetector
        ] = []

        # --------------------------------------------------
        # Video Information
        # --------------------------------------------------

        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def initialize(
        self,
        frame_width: int,
        frame_height: int,
        fps: float,
        line_a,
        line_b,
        stop_line,
    ):

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps if fps > 0 else 30.0

        # Reset track lifecycle.
        self.track_missed_frames.clear()

        # --------------------------------------------------
        # Save Selected Lines
        # --------------------------------------------------

        self.speed_line_a = line_a
        self.speed_line_b = line_b
        self.stop_line = stop_line

        # --------------------------------------------------
        # Speed Estimation
        # --------------------------------------------------

        if "speeding" in self.enabled_violations:

            self.speed_estimator = TwoLineSpeedEstimator(
                line_a=self.speed_line_a,
                line_b=self.speed_line_b,
                distance_meters=SPEED_REFERENCE_DISTANCE_M,
                fps=self.fps,
            )

        else:

            self.speed_estimator = None

        # --------------------------------------------------
        # Vehicle Counter
        # --------------------------------------------------

        self.counter = VehicleCounter(
            frame_height=frame_height,
            vehicle_classes=VEHICLE_CLASSES,
        )

        # --------------------------------------------------
        # Violation Manager
        # --------------------------------------------------

        self.violation_manager = ViolationManager(
            fps=self.fps
        )

        # --------------------------------------------------
        # Speed Limit Manager
        # --------------------------------------------------

        if "speeding" in self.enabled_violations:

            self.speed_limit_manager = (
                SpeedLimitManager(
                    confirmation_window=15,
                    min_confirmations=5,
                    confidence_threshold=0.50,
                )
            )

            self.speed_violation_detector = (
                SpeedViolationDetector(
                    speed_limit_manager=(
                        self.speed_limit_manager
                    ),
                    tolerance_kmh=5.0,
                    confirmation_frames=5,
                    min_speed_kmh=10.0,
                )
            )

        else:

            self.speed_limit_manager = None
            self.speed_violation_detector = None

        # --------------------------------------------------
        # Traffic Light Manager
        # --------------------------------------------------

        if "red_light" in self.enabled_violations:

            self.traffic_light_manager = (
                TrafficLightManager(
                    confirmation_window=15,
                    min_confirmations=5,
                    change_min_confirmations=8,
                    confidence_threshold=0.50,
                )
            )

            self.red_light_detector = (
                RedLightDetector(
                    stop_line=self.stop_line,
                    traffic_light_state_provider=(
                        self.traffic_light_manager.get_state
                    ),
                )
            )

        else:

            self.traffic_light_manager = None
            self.red_light_detector = None

        # --------------------------------------------------
        # Direction Engine
        # --------------------------------------------------

        self.direction_engine = None

        if "wrong_way" in self.enabled_violations:

            self.direction_engine = DirectionEngine(
                history_size=25,
                min_history=10,
                min_vertical_movement=40.0,
                consistency_threshold=0.75,
            )

        # --------------------------------------------------
        # Road Configuration
        # --------------------------------------------------

        self.road_config = None

        if "wrong_way" in self.enabled_violations:

            config_exists = Path(
                ROAD_CONFIG_PATH
            ).exists()

            if config_exists:

                try:

                    existing_config = RoadConfigLoader(
                        ROAD_CONFIG_PATH
                    )

                    if existing_config.matches_video(
                        self.video_path,
                        frame_width,
                        frame_height,
                        road_type=self.road_type,
                    ):

                        print(
                            "\nExisting road "
                            "configuration found."
                        )

                        print(
                            "Using saved road "
                            "configuration."
                        )

                        self.road_config = (
                            existing_config
                        )

                        if self.road_config.is_one_way():

                            print(
                                "Road type: ONE-WAY"
                            )

                            print(
                                "Allowed direction: "
                                f"{self.road_config.get_allowed_direction()}"
                            )

                        else:

                            print(
                                "Road type: TWO-WAY"
                            )

                            print(
                                "Using calibrated "
                                "left/right directions."
                            )

                    else:

                        print(
                            "\nSaved road configuration "
                            "does not match this video."
                        )

                        print(
                            "Running new road "
                            "auto-calibration..."
                        )

                        calibrator = RoadCalibrator(
                            model_path=self.model_path,
                            video_path=self.video_path,
                            road_type=self.road_type,
                            config_path=ROAD_CONFIG_PATH,
                        )

                        calibrator.calibrate()

                        self.road_config = (
                            RoadConfigLoader(
                                ROAD_CONFIG_PATH
                            )
                        )

                except (
                    FileNotFoundError,
                    ValueError,
                ) as exc:

                    print(
                        "\nSaved road configuration "
                        "is invalid."
                    )

                    print(
                        f"Reason: {exc}"
                    )

                    print(
                        "Running new road "
                        "auto-calibration..."
                    )

                    calibrator = RoadCalibrator(
                        model_path=self.model_path,
                        video_path=self.video_path,
                        road_type=self.road_type,
                        config_path=ROAD_CONFIG_PATH,
                    )

                    calibrator.calibrate()

                    self.road_config = (
                        RoadConfigLoader(
                            ROAD_CONFIG_PATH
                        )
                    )

            else:

                print(
                    "\nNo saved road configuration "
                    "found."
                )

                print(
                    "Running road "
                    "auto-calibration..."
                )

                calibrator = RoadCalibrator(
                    model_path=self.model_path,
                    video_path=self.video_path,
                    config_path=ROAD_CONFIG_PATH,
                    road_type=self.road_type,
                )

                calibrator.calibrate()

                self.road_config = (
                    RoadConfigLoader(
                        ROAD_CONFIG_PATH
                    )
                )

        else:

            print(
                "\nWrong-way detection disabled."
            )

            print(
                "Skipping road configuration "
                "and direction calibration."
            )

        # --------------------------------------------------
        # Wrong Way Detector
        # --------------------------------------------------

        if "wrong_way" in self.enabled_violations:

            self.wrong_way_detector = (
                WrongWayDetector(
                    frame_width=frame_width,
                    frame_height=frame_height,
                    road_config=self.road_config,
                    history_size=25,
                    min_history=10,
                    min_vertical_movement=40.0,
                    consistency_threshold=0.75,
                    confirmation_frames=15,
                    center_margin_ratio=0.12,
                )
            )

        else:

            self.wrong_way_detector = None

        # --------------------------------------------------
        # Unified Detector List
        # --------------------------------------------------

        self.violation_detectors = [
            detector
            for detector in [
                self.wrong_way_detector,
                self.speed_violation_detector,
                self.red_light_detector,
            ]
            if detector is not None
        ]

        print(
            "\nViolation detectors:"
        )

        for detector in self.violation_detectors:

            print(
                f"  - "
                f"{detector.__class__.__name__}"
            )

    # ======================================================
    # TRACK CLEANUP
    # ======================================================

    def _cleanup_expired_tracks(
        self,
        current_track_ids,
    ):
        """
        Remove stale track state after a track has been
        absent for enough consecutive frames.

        This prevents reused ByteTrack IDs from inheriting
        old trajectory / side / streak information.
        """

        current_track_ids = {
            int(track_id)
            for track_id in current_track_ids
        }

        # --------------------------------------------------
        # Update missed-frame counters.
        # --------------------------------------------------

        for track_id in list(
            self.track_missed_frames
        ):

            if track_id in current_track_ids:

                self.track_missed_frames[
                    track_id
                ] = 0

            else:

                self.track_missed_frames[
                    track_id
                ] += 1

        # --------------------------------------------------
        # Register new tracks.
        # --------------------------------------------------

        for track_id in current_track_ids:

            if (
                track_id
                not in self.track_missed_frames
            ):

                self.track_missed_frames[
                    track_id
                ] = 0

        # --------------------------------------------------
        # Find expired tracks.
        # --------------------------------------------------

        expired_track_ids = [
            track_id
            for track_id, missed_frames
            in self.track_missed_frames.items()
            if missed_frames
            >= self.track_cleanup_threshold
        ]

        # --------------------------------------------------
        # Remove expired tracks.
        # --------------------------------------------------

        for track_id in expired_track_ids:

            if self.wrong_way_detector is not None:

                self.wrong_way_detector.remove_track(
                    track_id
                )

            if self.speed_estimator is not None:

                self.speed_estimator.remove_track(
                    track_id
                )

            self.track_missed_frames.pop(
                track_id,
                None,
            )

    # ======================================================
    # PROCESS FRAME
    # ======================================================

    def process_frame(self, frame):

        self.frame_number += 1

        timestamp = (
            self.frame_number / self.fps
        )

        # --------------------------------------------------
        # Traffic Light Detection
        # --------------------------------------------------

        if "red_light" in self.enabled_violations:

            try:

                traffic_light_detections = (
                    self.traffic_sign_detector
                    .get_traffic_light_detections(
                        frame
                    )
                )

                if (
                    self.traffic_light_manager
                    is not None
                ):

                    previous_state = (
                        self.traffic_light_manager
                        .get_state()
                    )

                    current_state = (
                        self.traffic_light_manager
                        .update(
                            traffic_light_detections
                        )
                    )

                    if (
                        current_state
                        != previous_state
                    ):

                        print(
                            "\n[TRAFFIC LIGHT]"
                        )

                        print(
                            "Confirmed state: "
                            f"{current_state}"
                        )

            except Exception as exc:

                print(
                    "[WARNING] "
                    "Traffic light detection failed: "
                    f"{exc}"
                )

        # --------------------------------------------------
        # Speed Limit Detection
        # --------------------------------------------------

        if "speeding" in self.enabled_violations:

            try:

                speed_limit_detections = (
                    self.traffic_sign_detector
                    .get_speed_limit_detections(
                        frame
                    )
                )

                previous_speed_limit = (
                    self.speed_limit_manager
                    .get_speed_limit()
                    if self.speed_limit_manager
                    is not None
                    else None
                )

                if (
                    self.speed_limit_manager
                    is not None
                ):

                    current_speed_limit = (
                        self.speed_limit_manager.update(
                            speed_limit_detections
                        )
                    )

                    if (
                        current_speed_limit is not None
                        and current_speed_limit
                        != previous_speed_limit
                    ):

                        print(
                            "\n[SPEED LIMIT]"
                        )

                        print(
                            "Confirmed speed limit: "
                            f"{current_speed_limit} km/h"
                        )

            except Exception as exc:

                print(
                    "[WARNING] "
                    "Speed limit detection failed: "
                    f"{exc}"
                )

        # --------------------------------------------------
        # YOLO + Tracking
        # --------------------------------------------------

        tracking_result = self.tracker.track(
            frame
        )

        # --------------------------------------------------
        # Vehicle Counter
        # --------------------------------------------------

        if self.counter is not None:

            self.counter.update(
                tracking_result
            )

        # --------------------------------------------------
        # Extract Detections
        # --------------------------------------------------

        boxes = tracking_result.boxes

        if boxes is None:

            self._cleanup_expired_tracks(
                []
            )

            return tracking_result, []

        try:

            class_ids = (
                boxes.cls.cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                boxes.conf.cpu()
                .numpy()
            )

            track_ids = (
                boxes.id.cpu()
                .numpy()
                .astype(int)
                if boxes.id is not None
                else []
            )

            xyxy = (
                boxes.xyxy.cpu()
                .numpy()
            )

        except Exception as exc:

            print(
                "[WARNING] "
                "Failed to extract "
                f"detections: {exc}"
            )

            return tracking_result, []

        # --------------------------------------------------
        # Track Lifecycle Cleanup
        # --------------------------------------------------

        self._cleanup_expired_tracks(
            track_ids
        )

        # --------------------------------------------------
        # No Tracking IDs
        # --------------------------------------------------

        if len(track_ids) == 0:

            return tracking_result, []

        # --------------------------------------------------
        # Process Every Tracked Vehicle
        # --------------------------------------------------

        events = []

        for i, track_id in enumerate(
            track_ids
        ):

            class_id = int(
                class_ids[i]
            )

            confidence = float(
                confidences[i]
            )

            # --------------------------------------------------
            # Ignore Non-Vehicle Classes
            # --------------------------------------------------

            if class_id not in VEHICLE_CLASSES:

                continue

            # --------------------------------------------------
            # Class Name
            # --------------------------------------------------

            class_name = (
                VEHICLE_CLASSES[class_id]
            )

            # --------------------------------------------------
            # Bounding Box
            # --------------------------------------------------

            x1, y1, x2, y2 = xyxy[i]

            center_x = (
                float(x1) + float(x2)
            ) / 2.0

            center_y = (
                float(y1) + float(y2)
            ) / 2.0

            # --------------------------------------------------
            # Vehicle State
            # --------------------------------------------------

            vehicle = VehicleState(
                track_id=int(track_id),
                class_id=class_id,
                class_name=class_name,
                center_x=center_x,
                center_y=center_y,
                bbox=(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ),
                confidence=confidence,
            )

            # --------------------------------------------------
            # Speed Estimation
            # --------------------------------------------------

            speed_kmh = None

            if self.speed_estimator is not None:

                speed_kmh = (
                    self.speed_estimator.update(
                        track_id=int(track_id),
                        bbox=vehicle.bbox,
                        frame_number=self.frame_number,
                    )
                )

                vehicle.speed_kmh = speed_kmh

            # --------------------------------------------------
            # Run Violation Detectors
            # --------------------------------------------------

            for detector in (
                self.violation_detectors
            ):

                try:

                    event = detector.process(
                        vehicle=vehicle,
                        frame=frame,
                        frame_number=(
                            self.frame_number
                        ),
                        timestamp=timestamp,
                    )

                except Exception as exc:

                    print(
                        "[WARNING] "
                        f"{detector.__class__.__name__} "
                        f"failed for track "
                        f"{track_id}: {exc}"
                    )

                    continue

                # --------------------------------------------------
                # No Violation
                # --------------------------------------------------

                if event is None:

                    continue

                # --------------------------------------------------
                # Add Bounding Box
                # --------------------------------------------------

                event.details.setdefault(
                    "bbox",
                    list(vehicle.bbox),
                )

                # --------------------------------------------------
                # Save Evidence
                # --------------------------------------------------

                evidence_path = (
                    self.evidence_collector
                    .save_evidence(
                        frame,
                        event,
                    )
                )

                event.evidence_path = (
                    evidence_path
                )

                # --------------------------------------------------
                # Store Event
                # --------------------------------------------------

                stored_event = (
                    self.violation_manager
                    .add_violation(
                        violation_type=(
                            event.violation_type
                        ),
                        vehicle_id=(
                            event.vehicle_id
                        ),
                        frame_number=(
                            event.frame_number
                        ),
                        class_name=(
                            event.class_name
                        ),
                        direction=(
                            event.direction
                        ),
                        expected_direction=(
                            event.expected_direction
                        ),
                        severity=(
                            event.severity
                        ),
                        confidence=(
                            event.confidence
                        ),
                        details=(
                            event.details
                        ),
                        evidence_path=(
                            event.evidence_path
                        ),
                    )
                )

                # --------------------------------------------------
                # New Event Accepted
                # --------------------------------------------------

                if stored_event is None:

                    continue

                events.append(
                    stored_event
                )

                print(
                    "\n[VIOLATION]"
                )

                print(
                    f"Type: "
                    f"{stored_event.violation_type}"
                )

                print(
                    f"Vehicle: "
                    f"{stored_event.vehicle_id}"
                )

                print(
                    f"Frame: "
                    f"{stored_event.frame_number}"
                )

                print(
                    f"Evidence: "
                    f"{stored_event.evidence_path}"
                )

        return tracking_result, events

    # ======================================================
    # DRAW RESULTS
    # ======================================================

    def draw_results(
        self,
        frame,
        tracking_result,
    ):

        # --------------------------------------------------
        # Draw YOLO Tracking Results
        # --------------------------------------------------

        output = tracking_result.plot()

        # --------------------------------------------------
        # Draw Speed Lines
        # --------------------------------------------------

        if (
            self.speed_line_a is not None
            and self.speed_line_b is not None
        ):

            cv2.line(
                output,
                self.speed_line_a[0],
                self.speed_line_a[1],
                (255, 255, 0),
                3,
            )

            cv2.line(
                output,
                self.speed_line_b[0],
                self.speed_line_b[1],
                (0, 255, 255),
                3,
            )

            cv2.putText(
                output,
                "Speed Line A",
                (
                    int(
                        self.speed_line_a[0][0]
                    ),
                    int(
                        self.speed_line_a[0][1]
                    ) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                output,
                "Speed Line B",
                (
                    int(
                        self.speed_line_b[0][0]
                    ),
                    int(
                        self.speed_line_b[0][1]
                    ) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # --------------------------------------------------
        # Draw Stop Line
        # --------------------------------------------------

        if self.stop_line is not None:

            cv2.line(
                output,
                self.stop_line[0],
                self.stop_line[1],
                (0, 0, 255),
                3,
            )

            cv2.putText(
                output,
                "Stop Line",
                (
                    int(
                        self.stop_line[0][0]
                    ),
                    int(
                        self.stop_line[0][1]
                    ) - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # --------------------------------------------------
        # Traffic Light State
        # --------------------------------------------------

        if self.traffic_light_manager is not None:

            traffic_light_state = (
                self.traffic_light_manager
                .get_state()
            )

            cv2.putText(
                output,
                (
                    "Traffic Light: "
                    f"{traffic_light_state}"
                ),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # --------------------------------------------------
        # Counter Information
        # --------------------------------------------------

        if self.counter is not None:

            stats = (
                self.counter.get_statistics()
            )

            cv2.putText(
                output,
                (
                    f"Total Vehicles: "
                    f"{stats.get('total', 0)}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # --------------------------------------------------
        # Speed Limit Information
        # --------------------------------------------------

        if self.speed_limit_manager is not None:

            speed_limit = (
                self.speed_limit_manager
                .get_speed_limit()
            )

            if speed_limit is not None:

                cv2.putText(
                    output,
                    (
                        f"Speed Limit: "
                        f"{speed_limit} km/h"
                    ),
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

        # --------------------------------------------------
        # Violation Information
        # --------------------------------------------------

        if (
            self.violation_manager
            is not None
        ):

            total_violations = (
                self.violation_manager.count()
            )

            cv2.putText(
                output,
                (
                    f"Violations: "
                    f"{total_violations}"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        return output

    # ======================================================
    # FINAL STATISTICS
    # ======================================================

    def get_final_statistics(self):

        statistics = {}

        # --------------------------------------------------
        # Vehicle Statistics
        # --------------------------------------------------

        if self.counter is not None:

            statistics.update(
                self.counter.get_statistics()
            )

        # --------------------------------------------------
        # Violation Statistics
        # --------------------------------------------------

        if (
            self.violation_manager
            is not None
        ):

            statistics[
                "violations"
            ] = (
                self.violation_manager
                .summary()
            )

            statistics[
                "total_violations"
            ] = (
                self.violation_manager
                .count()
            )

        # --------------------------------------------------
        # Confirmed Speed Limit
        # --------------------------------------------------

        if self.speed_limit_manager is not None:

            statistics[
                "speed_limit_kmh"
            ] = (
                self.speed_limit_manager
                .get_speed_limit()
            )

        # --------------------------------------------------
        # Traffic Light State
        # --------------------------------------------------

        if self.traffic_light_manager is not None:

            statistics[
                "traffic_light_state"
            ] = (
                self.traffic_light_manager
                .get_state()
            )

        # --------------------------------------------------
        # Red-Light Statistics
        # --------------------------------------------------

        if self.red_light_detector is not None:

            statistics[
                "red_light_vehicles"
            ] = list(
                self.red_light_detector
                .get_violations()
            )

        # --------------------------------------------------
        # Wrong-Way Statistics
        # --------------------------------------------------

        if (
            self.wrong_way_detector
            is not None
        ):

            statistics[
                "wrong_way_vehicles"
            ] = list(
                self.wrong_way_detector
                .get_violations()
            )

        return statistics

    # ======================================================
    # RUN VIDEO
    # ======================================================

    def run_video(
        self,
        display: bool = True,
    ):

        cap = cv2.VideoCapture(
            self.video_path
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{self.video_path}"
            )

        # --------------------------------------------------
        # Video Information
        # --------------------------------------------------

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:

            fps = 30.0

        print(
            f"\nVideo: "
            f"{width}x{height} "
            f"@ {fps:.2f} FPS"
        )

        # --------------------------------------------------
        # Read First Frame
        # --------------------------------------------------

        ret, first_frame = cap.read()

        if not ret:

            cap.release()

            raise RuntimeError(
                "Could not read the first "
                "frame of the video."
            )

        # ==================================================
        # SPEED LINE CALIBRATION
        # ==================================================

        line_a = None
        line_b = None

        if "speeding" in self.enabled_violations:

            print(
                "\n=============================="
            )

            print(
                "SPEED LINE CALIBRATION"
            )

            print(
                "=============================="
            )

            print(
                "Select Line A using 2 clicks."
            )

            print(
                "Press ENTER to confirm."
            )

            print(
                "Then select Line B."
            )

            print(
                "Press ENTER to confirm."
            )

            print(
                "Press ESC to cancel."
            )

            selector = SpeedLineSelector(
                first_frame
            )

            line_a, line_b = selector.select()

            if (
                line_a is None
                or line_b is None
            ):

                cap.release()

                cv2.destroyAllWindows()

                print(
                    "\nSpeed line calibration "
                    "cancelled."
                )

                return None

            self.speed_line_a = line_a
            self.speed_line_b = line_b

            print(
                "\nSelected speed lines:"
            )

            print(
                f"Line A: {line_a}"
            )

            print(
                f"Line B: {line_b}"
            )

        # ==================================================
        # STOP LINE CALIBRATION
        # ==================================================

        stop_line = None

        if "red_light" in self.enabled_violations:

            print(
                "\n=============================="
            )

            print(
                "STOP LINE CALIBRATION"
            )

            print(
                "=============================="
            )

            print(
                "Select the stop line "
                "using 2 clicks."
            )

            print(
                "Press ENTER to confirm."
            )

            print(
                "Press R to reset."
            )

            print(
                "Press ESC to cancel."
            )

            stop_line_selector = (
                StopLineSelector(
                    first_frame
                )
            )

            stop_line = (
                stop_line_selector.select()
            )

            if stop_line is None:

                cap.release()

                cv2.destroyAllWindows()

                print(
                    "\nStop line calibration "
                    "cancelled."
                )

                return None

            self.stop_line = stop_line

            print(
                "\nSelected stop line:"
            )

            print(
                self.stop_line
            )

        # --------------------------------------------------
        # Reset Video
        # --------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0,
        )

        # --------------------------------------------------
        # Initialize Pipeline
        # --------------------------------------------------

        self.initialize(
            frame_width=width,
            frame_height=height,
            fps=fps,
            line_a=line_a,
            line_b=line_b,
            stop_line=stop_line,
        )

        print(
            "\nStarting traffic analysis..."
        )

        # --------------------------------------------------
        # Main Loop
        # --------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            # --------------------------------------------------
            # Process Frame Once
            # --------------------------------------------------

            tracking_result, events = (
                self.process_frame(frame)
            )

            # --------------------------------------------------
            # Draw Results
            # --------------------------------------------------

            output = self.draw_results(
                frame,
                tracking_result,
            )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            if display:

                cv2.imshow(
                    "Traffic Monitoring",
                    output,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    break

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------

        cap.release()

        cv2.destroyAllWindows()

        # --------------------------------------------------
        # Final Statistics
        # --------------------------------------------------

        statistics = (
            self.get_final_statistics()
        )

        print(
            "\n=============================="
        )

        print(
            "FINAL STATISTICS"
        )

        print(
            "=============================="
        )

        for key, value in (
            statistics.items()
        ):

            print(
                f"{key}: {value}"
            )

        print(
            "=============================="
        )

        return statistics