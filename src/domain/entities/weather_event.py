"""Weather event entity."""

from dataclasses import dataclass
from typing import Optional
from .growth_stage import GrowthStage


@dataclass(frozen=True)
class WeatherEvent:
    """Represents a discretized weather event for a growth stage."""

    stage: str
    temperature_label: str  # 'Cool', 'Moderate', 'Hot'
    precipitation_label: str  # 'Dry', 'Moderate', 'Wet'
    event_label: str  # Combined label like 'Nóng-Khô'

    def __str__(self) -> str:
        return f"{self.stage}_{self.event_label}"

