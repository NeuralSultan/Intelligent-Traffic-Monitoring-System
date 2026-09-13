from collections import defaultdict


class VehicleCounter:
    """
    Counts vehicles when their tracked center crosses
    a horizontal counting line.

    The same Track ID can only be counted once.
    """

    def __init__(
        self,
        frame_height: int,
        line_position: float = 0.55,
        vehicle_classes=None,
    ):
        self.frame_height = frame_height
        self.line_position = line_position

        self.counting_line_y = int(
            frame_height * line_position
        )

        if vehicle_classes is None:
            vehicle_classes = {
                0: "car",
                1: "bus",
                2: "truck",
                3: "motorcycle",
                4: "bicycle",
            }

        self.vehicle_classes = vehicle_classes

        # Track ID -> previous center Y
        self.previous_positions = {}

        # Track IDs that already crossed the line
        self.counted_ids = set()

        # Counts by class
        self.counts = {
            name: 0
            for name in self.vehicle_classes.values()
        }

        # Direction counts
        self.up_count = 0
        self.down_count = 0

    def update(self, result):
        """
        Process one YOLO + ByteTrack result.
        """

        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            return self.get_statistics()

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

            if class_id not in self.vehicle_classes:
                continue

            x1, y1, x2, y2 = box

            center_y = float(
                (y1 + y2) / 2
            )

            previous_y = self.previous_positions.get(
                track_id
            )

            # ------------------------------------------------
            # Detect crossing
            # ------------------------------------------------

            if previous_y is not None:

                crossed_down = (
                    previous_y < self.counting_line_y
                    and center_y >= self.counting_line_y
                )

                crossed_up = (
                    previous_y > self.counting_line_y
                    and center_y <= self.counting_line_y
                )

                if (
                    track_id not in self.counted_ids
                    and (crossed_down or crossed_up)
                ):

                    self.counted_ids.add(track_id)

                    class_name = (
                        self.vehicle_classes[class_id]
                    )

                    self.counts[class_name] += 1

                    if crossed_down:
                        self.down_count += 1
                    else:
                        self.up_count += 1

            self.previous_positions[track_id] = center_y

        return self.get_statistics()

    def get_statistics(self):
        """
        Return current counting statistics.
        """

        total = sum(
            self.counts.values()
        )

        return {
            "total": total,
            **self.counts,
            "up": self.up_count,
            "down": self.down_count,
        }

    def get_line_position(self):
        """
        Return the Y coordinate of the counting line.
        """

        return self.counting_line_y