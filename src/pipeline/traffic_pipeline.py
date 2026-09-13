from pathlib import Path

import cv2

from configs.traffic_config import (
    PROJECT_ROOT,
    MODEL_PATH,
    CONFIDENCE,
    IMAGE_SIZE,
    TRACKER,
    VEHICLE_CLASSES,
    CONFIG_PATH,
)

from src.violations.violation_manager import (
    ViolationManager,
)

from configs.road_config_loader import (
    RoadConfigLoader,
)

from src.tracking.tracker import (
    TrafficTracker,
)

from src.analytics.vehicle_counter import (
    VehicleCounter,
)

from src.analytics.direction import (
    DirectionEngine,
)

from src.analytics.road_calibrator import (
    RoadCalibrator,
)

from src.violations.wrong_way import (
    WrongWayDetector,
)


class TrafficPipeline:
    """
    Main intelligent traffic-processing pipeline.

    Pipeline flow:

        Video
          ↓
        Auto Road Calibration
          ↓
        Road Config
          ↓
        YOLO + ByteTrack
          ↓
        Vehicle Counter
          ↓
        Direction Engine
          ↓
        Wrong-Way Detector
          ↓
        Future:
            Speed
            Red Light
            Lane Violations
            Congestion
            Reporting
    """

    def __init__(
        self,
        video_path: str | Path,
        model_path: str | Path = MODEL_PATH,
    ):
        self.video_path = Path(
            video_path
        )

        self.model_path = Path(
            model_path
        )

        self.violation_manager = None

        self.frame_number = 0

        # ====================================================
        # Main tracker
        # ====================================================

        self.tracker = TrafficTracker(
            model_path=self.model_path,
            confidence=CONFIDENCE,
            image_size=IMAGE_SIZE,
            tracker_config=TRACKER,
        )

        # ====================================================
        # Analytics modules
        # ====================================================

        self.counter = None

        self.direction_engine = (
            DirectionEngine(
                history_size=25,
                min_history=10,
                min_vertical_movement=40,
                consistency_threshold=0.75,
            )
        )

        # ====================================================
        # Road configuration
        # ====================================================

        self.road_config = None

        self.wrong_way_detector = None

        self.width = None
        self.height = None

    def initialize(
        self,
        frame_width: int,
        frame_height: int,
        fps: float,
        
    ):
        """
        Initialize dimension-dependent modules.

        Auto-calibration happens here if needed.
        """

        self.width = frame_width
        self.height = frame_height

        # ====================================================
        # Vehicle counter
        # ====================================================

        self.counter = VehicleCounter(
            frame_height=frame_height,
            line_position=0.55,
            vehicle_classes=VEHICLE_CLASSES,
        )

        self.violation_manager = ViolationManager(
            fps=fps
        )        
        # ====================================================
        # Automatic road calibration
        # ====================================================

        calibrator = RoadCalibrator(
            model_path=self.model_path,
            video_path=self.video_path,
            config_path=CONFIG_PATH,
            confidence=CONFIDENCE,
            image_size=IMAGE_SIZE,
            history_size=25,
            min_history=10,
            min_vertical_movement=40,
            consistency_threshold=0.75,
            center_margin_ratio=0.12,
            top_ignore_ratio=0.35,
            min_tracks_per_side=3,
            max_calibration_frames=1500,
        )

        calibrator.calibrate()

        # ====================================================
        # Load generated configuration
        # ====================================================

        self.road_config = (
            RoadConfigLoader(
                CONFIG_PATH
            )
        )

        # ====================================================
        # Wrong-way detector
        # ====================================================

        self.wrong_way_detector = (
            WrongWayDetector(
                frame_width=frame_width,
                frame_height=frame_height,
                road_config=self.road_config,
                history_size=25,
                min_history=10,
                min_vertical_movement=40,
                consistency_threshold=0.75,
                confirmation_frames=15,
                center_margin_ratio=0.12,
            )
        )

        print("\nRoad configuration loaded:")
        print(
            f"LEFT  -> "
            f"{self.road_config.get_expected_direction('LEFT')}"
        )

        print(
            f"RIGHT -> "
            f"{self.road_config.get_expected_direction('RIGHT')}"
        )

    def process_frame(
        self,
        frame,
    ):
        """
        Process one frame.

        Returns:
            result
            statistics
            track_data
        """
        self.frame_number += 1

        if self.counter is None:
            raise RuntimeError(
                "Pipeline has not been initialized."
            )

        if self.wrong_way_detector is None:
            raise RuntimeError(
                "Wrong-way detector has not been initialized."
            )

        # ====================================================
        # YOLO + ByteTrack
        # ====================================================

        result = self.tracker.track(
            frame
        )

        # ====================================================
        # Vehicle counting
        # ====================================================

        statistics = (
            self.counter.update(
                result
            )
        )

        # ====================================================
        # Track data
        # ====================================================

        track_data = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            class_ids = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, class_id, track_id in zip(
                boxes,
                class_ids,
                track_ids,
            ):

                if class_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                center_x = float(
                    (x1 + x2) / 2
                )

                center_y = float(
                    (y1 + y2) / 2
                )

                # =================================================
                # Direction
                # =================================================

                direction_info = (
                    self.direction_engine.update(
                        track_id=int(track_id),
                        center_x=center_x,
                        center_y=center_y,
                    )
                )

                # =================================================
                # Wrong-way
                # =================================================

                wrong_way_info = (
                    self.wrong_way_detector.update(
                        track_id=int(track_id),
                        center_x=center_x,
                        center_y=center_y,
                    )
                )
                if wrong_way_info["confirmed"]:

                    event = self.violation_manager.add_violation(
                        violation_type="wrong_way",
                        vehicle_id=int(track_id),
                        frame_number=self.frame_number,
                        class_name=VEHICLE_CLASSES[class_id],
                        direction=wrong_way_info["direction"],
                        expected_direction=(
                            wrong_way_info["expected_direction"]
                        ),
                        details={
                            "road_side": wrong_way_info["side"],
                            "consistency": (
                                wrong_way_info["consistency"]
                            ),
                        },
                    )

                    # Print only when a NEW event is created
                    if event is not None:

                        print(
                            "\n[VIOLATION DETECTED]"
                        )

                        print(
                            f"Type: {event.violation_type}"
                        )

                        print(
                            f"Vehicle ID: {event.vehicle_id}"
                        )

                        print(
                            f"Class: {event.class_name}"
                        )

                        print(
                            f"Direction: {event.direction}"
                        )

                        print(
                            f"Expected: {event.expected_direction}"
                        )

                        print(
                            f"Frame: {event.frame_number}"
                        )

                        print(
                            f"Timestamp: {event.timestamp:.2f}s"
                        )
                # =================================================
                # Store all information
                # =================================================

                track_data.append(
                    {
                        "track_id": int(track_id),

                        "class_id": int(
                            class_id
                        ),

                        "class_name": (
                            VEHICLE_CLASSES[
                                class_id
                            ]
                        ),

                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],

                        "center": [
                            center_x,
                            center_y,
                        ],

                        "direction": (
                            direction_info[
                                "direction"
                            ]
                        ),

                        "consistency": (
                            direction_info[
                                "consistency"
                            ]
                        ),

                        "trajectory": (
                            direction_info[
                                "history"
                            ]
                        ),

                        "wrong_way": (
                            wrong_way_info[
                                "wrong_way"
                            ]
                        ),

                        "wrong_way_streak": (
                            wrong_way_info[
                                "streak"
                            ]
                        ),

                        "wrong_way_confirmed": (
                            wrong_way_info[
                                "confirmed"
                            ]
                        ),

                        "road_side": (
                            wrong_way_info[
                                "side"
                            ]
                        ),

                        "expected_direction": (
                            wrong_way_info[
                                "expected_direction"
                            ]
                        ),
                    }
                )

        return (
            result,
            statistics,
            track_data,
        )

    def draw_results(
        self,
        frame,
        result,
        statistics,
        track_data,
    ):
        """
        Draw current pipeline output.
        """

        annotated_frame = result.plot()

        # ====================================================
        # Counting line
        # ====================================================

        line_y = (
            self.counter.get_line_position()
        )

        cv2.line(
            annotated_frame,
            (0, line_y),
            (
                annotated_frame.shape[1],
                line_y,
            ),
            (255, 255, 255),
            3,
        )

        # ====================================================
        # Total
        # ====================================================

        x = 20
        y = 35

        cv2.putText(
            annotated_frame,
            f"Total: {statistics['total']}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        y += 30

        # ====================================================
        # Vehicle classes
        # ====================================================

        for class_name in (
            VEHICLE_CLASSES.values()
        ):

            cv2.putText(
                annotated_frame,
                (
                    f"{class_name}: "
                    f"{statistics[class_name]}"
                ),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            y += 25

        # ====================================================
        # Directions
        # ====================================================

        cv2.putText(
            annotated_frame,
            f"UP: {statistics['up']}",
            (
                annotated_frame.shape[1] - 170,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"DOWN: {statistics['down']}",
            (
                annotated_frame.shape[1] - 170,
                65,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # Wrong-way total
        # ====================================================

        wrong_way_total = len(
            self.wrong_way_detector
            .get_violations()
        )

        cv2.putText(
            annotated_frame,
            f"Wrong-way: {wrong_way_total}",
            (
                20,
                annotated_frame.shape[0] - 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # Per-vehicle information
        # ====================================================

        for track in track_data:

            x1, y1, x2, y2 = (
                track["bbox"]
            )

            track_id = (
                track["track_id"]
            )

            direction = (
                track["direction"]
            )

            side = (
                track["road_side"]
            )

            # Direction
            if direction is not None:

                cv2.putText(
                    annotated_frame,
                    (
                        f"ID {track_id}: "
                        f"{direction}"
                    ),
                    (
                        int(x1),
                        int(y2) + 20,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Road side
            if side is not None:

                cv2.putText(
                    annotated_frame,
                    f"Side: {side}",
                    (
                        int(x1),
                        int(y2) + 40,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

            # Confirmed violation
            if track[
                "wrong_way_confirmed"
            ]:

                cv2.putText(
                    annotated_frame,
                    "WRONG WAY",
                    (
                        int(x1),
                        int(y2) + 65,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

        return annotated_frame

    def get_final_statistics(
        self,
    ):
        """
        Return final pipeline statistics.
        """

        if self.counter is None:
            return {}

        statistics = (
            self.counter.get_statistics()
        )

        wrong_way = len(
            self.wrong_way_detector
            .get_violations()
        )

        statistics[
            "wrong_way"
        ] = wrong_way

        return statistics


def run_video(
    video_path: str | Path,
):
    """
    Run the complete intelligent traffic pipeline.
    """

    video_path = Path(
        video_path
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )
 # ================================================
    # Create pipeline FIRST
    # ================================================

    pipeline = TrafficPipeline(
        video_path=video_path
    )
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
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    pipeline.initialize(
        frame_width=width,
        frame_height=height,
        fps=fps,
    )

    latest_statistics = None

    frame_count = 0

    # ========================================================
    # Main monitoring loop
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        (
            result,
            statistics,
            track_data,
        ) = pipeline.process_frame(
            frame
        )

        latest_statistics = statistics

        annotated_frame = (
            pipeline.draw_results(
                frame,
                result,
                statistics,
                track_data,
            )
        )

        cv2.imshow(
            "Intelligent Traffic Monitoring",
            annotated_frame,
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ========================================================
    # Final report
    # ========================================================

    print("\n" + "=" * 60)
    print(
        "TRAFFIC PIPELINE COMPLETED"
    )
    print("=" * 60)

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    if latest_statistics is not None:

        print(
            f"Total vehicles: "
            f"{latest_statistics['total']}"
        )

        for class_name in (
            VEHICLE_CLASSES.values()
        ):

            print(
                f"{class_name}: "
                f"{latest_statistics[class_name]}"
            )

        print(
            f"Up: "
            f"{latest_statistics['up']}"
        )

        print(
            f"Down: "
            f"{latest_statistics['down']}"
        )

    wrong_way_violations = (
        pipeline.wrong_way_detector
        .get_violations()
    )

    print("\nViolations:")
    print("-" * 30)

    print(
        f"Wrong Way: "
        f"{len(wrong_way_violations)}"
    )

    if wrong_way_violations:

        print(
            "\nWrong-way vehicle IDs:"
        )

        for vehicle_id in sorted(
            wrong_way_violations
        ):

            print(
                f"  Vehicle ID: "
                f"{vehicle_id}"
            )

    else:

        print(
            "  No wrong-way violations detected."
        )



if __name__ == "__main__":

    video_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "test_videos"
        / "traffic_test.mp4"
    )

    run_video(
        video_path
    )

    