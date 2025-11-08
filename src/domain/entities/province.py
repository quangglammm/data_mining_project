"""Province entity."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Province:
    """Represents a province in the Mekong Delta."""

    name: str
    code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def __str__(self) -> str:
        return self.name

