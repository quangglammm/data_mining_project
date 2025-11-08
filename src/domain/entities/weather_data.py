"""Weather data entity."""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class WeatherData:
    """Represents daily weather data."""

    province: str
    date: date
    max_temp: Optional[float] = None  # Celsius
    min_temp: Optional[float] = None  # Celsius
    mean_temp: Optional[float] = None  # Celsius
    precipitation_sum: Optional[float] = None  # mm
    humidity_mean: Optional[float] = None  # percentage
    et0_mm: Optional[float] = None  # mm (evapotranspiration)
    weather_code: Optional[int] = None

    @property
    def dtr(self) -> Optional[float]:
        """Diurnal temperature range."""
        if self.max_temp is not None and self.min_temp is not None:
            return self.max_temp - self.min_temp
        return None

