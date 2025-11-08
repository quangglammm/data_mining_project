"""Growth stage entity."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GrowthStage:
    """Represents a rice growth stage with day range."""

    name: str  # e.g., 'stage_1', 'stage_2', etc.
    start_day: int  # Days from season start
    end_day: int  # Days from season start

    @property
    def day_range(self) -> Tuple[int, int]:
        """Get day range as tuple."""
        return (self.start_day, self.end_day)

    def __str__(self) -> str:
        return self.name

