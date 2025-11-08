"""Rice yield data entity."""

from dataclasses import dataclass
from typing import Optional
from .yield_class import YieldClass


@dataclass
class RiceYieldData:
    """Represents rice yield data for a province-season-year combination."""

    province: str
    year: int
    season: str
    cultivated_area: float  # in hectares
    rice_yield: float  # in tons/hectare
    rice_production: Optional[float] = None  # in tons
    yield_class: Optional[YieldClass] = None
    expected_yield: Optional[float] = None
    residual: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.province}_{self.year}_{self.season}"

