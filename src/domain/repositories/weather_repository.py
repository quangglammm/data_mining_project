"""Weather repository interface."""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional
from ..entities.weather_data import WeatherData


class WeatherRepository(ABC):
    """Abstract repository for weather data access."""

    @abstractmethod
    def get_weather_data(
        self,
        province: str,
        start_date: date,
        end_date: date,
    ) -> List[WeatherData]:
        """
        Retrieve weather data for a province within a date range.

        Args:
            province: Province name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of WeatherData entities
        """
        pass

    @abstractmethod
    def save_weather_data(self, data: List[WeatherData]) -> None:
        """
        Save weather data.

        Args:
            data: List of WeatherData entities to save
        """
        pass

