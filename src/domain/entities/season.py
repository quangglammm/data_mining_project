"""Season entity."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class Season:
    """Represents a rice growing season."""

    name: str  # e.g., 'winter_spring', 'summer_autumn', 'main_season'
    start_month: int
    start_day: int
    end_month: int
    end_day: int
    year_offset: int  # -1 if season starts in previous year

    @classmethod
    def from_dict(cls, name: str, definition: Dict[str, Any]) -> "Season":
        """Create Season from dictionary definition."""
        return cls(
            name=name,
            start_month=definition["start_month"],
            start_day=definition["start_day"],
            end_month=definition["end_month"],
            end_day=definition["end_day"],
            year_offset=definition.get("year_offset", 0),
        )

    def __str__(self) -> str:
        return self.name

