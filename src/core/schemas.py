from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleState:
    track_id: int
    class_id: int
    class_name: str

    center_x: float
    center_y: float

    bbox: tuple[float, float, float, float]

    confidence: float

    direction: Optional[str] = None
    road_side: Optional[str] = None
    speed_kmh: Optional[float] = None

@dataclass
class ViolationEvent:
    violation_type: str
    vehicle_id: Optional[int]

    frame_number: int
    timestamp: float

    class_name: Optional[str] = None

    direction: Optional[str] = None
    expected_direction: Optional[str] = None

    severity: str = "medium"

    confidence: Optional[float] = None

    details: dict = field(default_factory=dict)

    evidence_path: Optional[str] = None