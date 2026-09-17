from pathlib import Path
import json

import cv2


class SpeedCalibrator:
    """
    Interactive calibration tool for two-line speed estimation.

    The user selects:
        1. Two points for Line A
        2. Two points for Line B
        3. Real-world distance between the two lines

    Controls:
        A       Start selecting Line A
        B       Start selecting Line B
        R       Reset current selection
        ENTER   Save calibration
        ESC     Cancel
    """

    def __init__(
        self,
        video_path: str | Path,
        output_path: str | Path,
        scene_id: str,
    ):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.scene_id = scene_id

        self.line_a_points = []
        self.line_b_points = []

        self.current_line = "A"

        self.frame = None
        self.display_frame = None

    # ---------------------------------------------------------
    # Mouse callback
    # ---------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.current_line == "A":

            if len(self.line_a_points) < 2:
                self.line_a_points.append((x, y))

                print(
                    f"Line A point "
                    f"{len(self.line_a_points)}: ({x}, {y})"
                )

        elif self.current_line == "B":

            if len(self.line_b_points) < 2:
                self.line_b_points.append((x, y))

                print(
                    f"Line B point "
                    f"{len(self.line_b_points)}: ({x}, {y})"
                )

        self._draw()

    # ---------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------

    def _draw(self):

        if self.frame is None:
            return

        self.display_frame = self.frame.copy()

        # ---------------------------------------------
        # Draw Line A
        # ---------------------------------------------

        if len(self.line_a_points) >= 1:

            for point in self.line_a_points:
                cv2.circle(
                    self.display_frame,
                    point,
                    6,
                    (0, 255, 0),
                    -1,
                )

        if len(self.line_a_points) == 2:

            cv2.line(
                self.display_frame,
                self.line_a_points[0],
                self.line_a_points[1],
                (0, 255, 0),
                3,
            )

            cv2.putText(
                self.display_frame,
                "LINE A",
                self.line_a_points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # ---------------------------------------------
        # Draw Line B
        # ---------------------------------------------

        if len(self.line_b_points) >= 1:

            for point in self.line_b_points:
                cv2.circle(
                    self.display_frame,
                    point,
                    6,
                    (0, 0, 255),
                    -1,
                )

        if len(self.line_b_points) == 2:

            cv2.line(
                self.display_frame,
                self.line_b_points[0],
                self.line_b_points[1],
                (0, 0, 255),
                3,
            )

            cv2.putText(
                self.display_frame,
                "LINE B",
                self.line_b_points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ---------------------------------------------
        # Instructions
        # ---------------------------------------------

        instructions = [
            "A: select Line A",
            "B: select Line B",
            "R: reset",
            "ENTER: save",
            "ESC: cancel",
        ]

        y = 30

        for text in instructions:

            cv2.putText(
                self.display_frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            y += 28

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def _reset(self):

        self.line_a_points = []
        self.line_b_points = []

        self.current_line = "A"

        print("\nCalibration reset.")
        print("Select Line A again.")

        self._draw()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def _save(self):

        if len(self.line_a_points) != 2:
            print("Line A must contain exactly two points.")
            return

        if len(self.line_b_points) != 2:
            print("Line B must contain exactly two points.")
            return

        try:
            distance = float(
                input(
                    "\nEnter real distance between "
                    "Line A and Line B in meters: "
                )
            )

        except ValueError:

            print(
                "Invalid distance. "
                "Please enter a number."
            )

            return

        if distance <= 0:

            print(
                "Distance must be greater than zero."
            )

            return

        video_width = self.frame.shape[1]
        video_height = self.frame.shape[0]

        config = {
            "scene_id": self.scene_id,
            "video_name": self.video_path.name,
            "video_width": video_width,
            "video_height": video_height,
            "speed_calibration": {
                "line_a": [
                    list(self.line_a_points[0]),
                    list(self.line_a_points[1]),
                ],
                "line_b": [
                    list(self.line_b_points[0]),
                    list(self.line_b_points[1]),
                ],
                "distance_meters": distance,
                "calibration_valid": True,
            },
        }

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                config,
                file,
                indent=4,
            )

        print(
            f"\nCalibration saved to:\n"
            f"{self.output_path}"
        )

        print("\nCalibration:")
        print(json.dumps(config, indent=4))

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(self):

        if not self.video_path.exists():

            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{self.video_path}"
            )

        success, frame = cap.read()

        cap.release()

        if not success:

            raise RuntimeError(
                "Could not read the first frame."
            )

        self.frame = frame

        window_name = "Speed Calibration"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            window_name,
            self._mouse_callback,
        )

        self._draw()

        print("\n===================================")
        print("      SPEED CALIBRATION")
        print("===================================")

        print("\nStep 1:")
        print("Press A and select TWO points for Line A.")

        print("\nStep 2:")
        print("Press B and select TWO points for Line B.")

        print("\nStep 3:")
        print(
            "Press ENTER and enter the real distance "
            "between the two lines."
        )

        print("\nControls:")
        print("A     -> Line A")
        print("B     -> Line B")
        print("R     -> Reset")
        print("ENTER -> Save")
        print("ESC   -> Cancel")

        while True:

            cv2.imshow(
                window_name,
                self.display_frame,
            )

            key = cv2.waitKey(30) & 0xFF

            # A
            if key in (ord("a"), ord("A")):

                self.current_line = "A"

                print(
                    "\nSelecting Line A."
                )

            # B
            elif key in (ord("b"), ord("B")):

                self.current_line = "B"

                print(
                    "\nSelecting Line B."
                )

            # Reset
            elif key in (ord("r"), ord("R")):

                self._reset()

            # ENTER
            elif key in (13, 10):

                self._save()

                if self.output_path.exists():
                    break

            # ESC
            elif key == 27:

                print(
                    "\nCalibration cancelled."
                )

                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    VIDEO_PATH = (
        r"D:\Intelligent Traffic Monitoring & "
        r"Violation Detection System"
        r"\data\raw\test_videos\traffic_test.mp4"
    )

    OUTPUT_PATH = (
        r"D:\Intelligent Traffic Monitoring & "
        r"Violation Detection System"
        r"\configs\calibrations\camera_01_speed.json"
    )

    calibrator = SpeedCalibrator(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        scene_id="camera_01",
    )

    calibrator.run()

