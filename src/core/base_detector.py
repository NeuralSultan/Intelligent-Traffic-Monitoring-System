from abc import ABC, abstractmethod
from typing import Optional

from src.core.schemas import VehicleState, ViolationEvent


class BaseViolationDetector(ABC):

    @abstractmethod
    def process(
        self,
        vehicle: VehicleState,
        frame,
        frame_number: int,
        timestamp: float,
    ) -> Optional[ViolationEvent]:
        """
        Process one tracked vehicle and optionally return a violation event.
        """
        pass