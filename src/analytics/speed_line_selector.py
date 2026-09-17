import cv2


class SpeedLineSelector:
    """
    Allows the user to select two virtual speed lines
    from the first frame of a video.

    Controls:
        Left click  -> select point
        Enter       -> confirm current line
        R           -> reset current line
        ESC         -> cancel
    """

    def __init__(self, frame):
        self.original_frame = frame.copy()

        self.frame = frame.copy()

        self.current_points = []

        self.line_a = None
        self.line_b = None

        self.current_line = "A"

        self.window_name = "Speed Line Calibration"

        self.finished = False
        self.cancelled = False

    def _mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Only allow two points for the current line
        if len(self.current_points) >= 2:
            return

        self.current_points.append(
            (x, y)
        )

        self._redraw()

    def _redraw(self):
        self.frame = (
            self.original_frame.copy()
        )

        # Draw Line A
        if self.line_a is not None:
            cv2.line(
                self.frame,
                self.line_a[0],
                self.line_a[1],
                (0, 255, 255),
                3,
            )

            cv2.putText(
                self.frame,
                "Line A",
                self.line_a[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # Draw Line B
        if self.line_b is not None:
            cv2.line(
                self.frame,
                self.line_b[0],
                self.line_b[1],
                (0, 255, 0),
                3,
            )

            cv2.putText(
                self.frame,
                "Line B",
                self.line_b[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Draw currently selected points
        for point in self.current_points:
            cv2.circle(
                self.frame,
                point,
                7,
                (0, 0, 255),
                -1,
            )

        # Draw temporary line
        if len(self.current_points) == 2:

            cv2.line(
                self.frame,
                self.current_points[0],
                self.current_points[1],
                (0, 0, 255),
                2,
            )

        # Instructions
        if self.current_line == "A":
            instruction = (
                "Line A: click 2 points | "
                "ENTER = confirm | R = reset | ESC = cancel"
            )
        else:
            instruction = (
                "Line B: click 2 points | "
                "ENTER = confirm | R = reset | ESC = cancel"
            )

        cv2.putText(
            self.frame,
            instruction,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    def select(self):
        """
        Opens the calibration window and waits
        for the user to select Line A and Line B.
        """

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.setMouseCallback(
            self.window_name,
            self._mouse_callback,
        )

        self._redraw()

        while True:

            cv2.imshow(
                self.window_name,
                self.frame,
            )

            key = cv2.waitKey(20) & 0xFF

            # Enter
            if key in (13, 10):

                if len(self.current_points) != 2:
                    continue

                selected_line = (
                    tuple(
                        self.current_points[0]
                    ),
                    tuple(
                        self.current_points[1]
                    ),
                )

                if self.current_line == "A":

                    self.line_a = selected_line

                    self.current_points = []

                    self.current_line = "B"

                    self._redraw()

                else:

                    self.line_b = selected_line

                    self.finished = True

                    break

            # Reset current line
            elif key in (ord("r"), ord("R")):

                self.current_points = []

                self._redraw()

            # Cancel
            elif key == 27:

                self.cancelled = True

                break

        cv2.destroyWindow(
            self.window_name
        )

        if self.cancelled:
            return None, None

        if not self.finished:
            return None, None

        return (
            self.line_a,
            self.line_b,
        )