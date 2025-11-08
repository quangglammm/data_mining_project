"""Use case for collecting weather data."""

import logging
from datetime import date
from typing import List
from ..entities.weather_data import WeatherData
from ..repositories.weather_repository import WeatherRepository

logger = logging.getLogger(__name__)


class CollectWeatherDataUseCase:
    """Use case to collect weather data from repository."""

    def __init__(self, repository: WeatherRepository):
        """
        Initialize use case.

        Args:
            repository: Repository for weather data access
        """
        self.repository = repository

    def execute(
        self,
        province: str,
        start_date: date,
        end_date: date,
    ) -> List[WeatherData]:
        """
        Execute the use case.

        Args:
            province: Province name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of WeatherData entities
        """
        logger.info(
            f"Collecting weather data: province={province}, "
            f"start={start_date}, end={end_date}"
        )
        data = self.repository.get_weather_data(province, start_date, end_date)
        logger.info(f"Collected {len(data)} weather records")
        return data

