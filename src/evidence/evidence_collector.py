from pathlib import Path
from typing import Optional

import cv2

from src.core.schemas import ViolationEvent


class EvidenceCollector:
    """
    Saves annotated evidence frames for detected traffic violations.
    """

    def __init__(
        self,
        output_dir: str | Path,
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_evidence(
        self,
        frame,
        event: ViolationEvent,
    ) -> Optional[str]:
        """
        Save an annotated frame as evidence.

        The saved image contains:
            - Bounding box around the violating vehicle
            - Violation type
            - Vehicle ID
            - Expected direction
            - Actual direction

        Returns:
            Path to saved evidence image,
            or None if saving failed.
        """
        print("[EVIDENCE] save_evidence() called")

        if frame is None:
            print("[EVIDENCE] Frame is None")
            return None

        # Copy frame so we don't modify
        # the original pipeline frame.
        evidence_frame = frame.copy()

        # --------------------------------------------------
        # Get bounding box
        # --------------------------------------------------

        bbox = event.details.get("bbox")

        if bbox is not None:
            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            # Draw bounding box
            cv2.rectangle(
                evidence_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3,
            )

            # --------------------------------------------------
            # Vehicle label
            # --------------------------------------------------

            vehicle_label = (
                f"{event.class_name or 'vehicle'} "
                f"| ID: {event.vehicle_id}"
            )

            cv2.putText(
                evidence_frame,
                vehicle_label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # Build violation information
        # --------------------------------------------------

        violation_text = (
            f"VIOLATION: "
            f"{event.violation_type.upper()}"
        )

        cv2.putText(
            evidence_frame,
            violation_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Expected direction
        if event.expected_direction is not None:
            expected_text = (
                f"Expected: "
                f"{event.expected_direction}"
            )

            cv2.putText(
                evidence_frame,
                expected_text,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Actual direction
        if event.direction is not None:
            actual_text = (
                f"Actual: "
                f"{event.direction}"
            )

            cv2.putText(
                evidence_frame,
                actual_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # --------------------------------------------------
        # Save evidence
        # --------------------------------------------------

        violation_dir = (
            self.output_dir
            / event.violation_type
        )

        violation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vehicle_id = (
            f"vehicle_{event.vehicle_id}"
            if event.vehicle_id is not None
            else "unknown_vehicle"
        )

        filename = (
            f"{vehicle_id}"
            f"_frame_{event.frame_number}.jpg"
        )

        output_path = (
            violation_dir / filename
        )

        success = cv2.imwrite(
            str(output_path),
            evidence_frame,
        )

        if not success:
            return None

        return str(output_path)