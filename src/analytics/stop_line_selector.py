import cv2


class StopLineSelector:
    """
    Allows the user to select a stop line
    from the first frame of a video.

    Controls:
        Left click -> select point
        Enter      -> confirm line
        R          -> reset
        ESC        -> cancel
    """

    def __init__(self, frame):

        self.original_frame = frame.copy()
        self.frame = frame.copy()

        self.points = []

        self.line = None

        self.window_name = (
            "Stop Line Calibration"
        )

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

        if len(self.points) >= 2:
            return

        self.points.append(
            (x, y)
        )

        self._redraw()

    def _redraw(self):

        self.frame = (
            self.original_frame.copy()
        )

        # ------------------------------------------
        # Draw selected points
        # ------------------------------------------

        for point in self.points:

            cv2.circle(
                self.frame,
                point,
                7,
                (0, 0, 255),
                -1,
            )

        # ------------------------------------------
        # Draw current line
        # ------------------------------------------

        if len(self.points) == 2:

            cv2.line(
                self.frame,
                self.points[0],
                self.points[1],
                (0, 0, 255),
                3,
            )

            cv2.putText(
                self.frame,
                "STOP LINE",
                (
                    self.points[0][0],
                    self.points[0][1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ------------------------------------------
        # Instructions
        # ------------------------------------------

        instruction = (
            "Click 2 points | "
            "ENTER = confirm | "
            "R = reset | "
            "ESC = cancel"
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

            key = (
                cv2.waitKey(20)
                & 0xFF
            )

            # --------------------------------------
            # Confirm
            # --------------------------------------

            if key in (13, 10):

                if len(self.points) != 2:
                    continue

                self.line = (
                    tuple(self.points[0]),
                    tuple(self.points[1]),
                )

                self.finished = True

                break

            # --------------------------------------
            # Reset
            # --------------------------------------

            elif key in (
                ord("r"),
                ord("R"),
            ):

                self.points = []

                self._redraw()

            # --------------------------------------
            # Cancel
            # --------------------------------------

            elif key == 27:

                self.cancelled = True

                break

        cv2.destroyWindow(
            self.window_name
        )

        if (
            self.cancelled
            or not self.finished
        ):

            return None

        return self.line
