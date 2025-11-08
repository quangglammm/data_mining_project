"""Use case for collecting rice yield data."""

import logging
from typing import List
from ..entities.rice_yield_data import RiceYieldData
from ..repositories.rice_yield_repository import RiceYieldRepository

logger = logging.getLogger(__name__)


class CollectRiceYieldDataUseCase:
    """Use case to collect rice yield data from repository."""

    def __init__(self, repository: RiceYieldRepository):
        """
        Initialize use case.

        Args:
            repository: Repository for rice yield data access
        """
        self.repository = repository

    def execute(
        self,
        province: str = None,
        year: int = None,
        season: str = None,
    ) -> List[RiceYieldData]:
        """
        Execute the use case.

        Args:
            province: Optional province filter
            year: Optional year filter
            season: Optional season filter

        Returns:
            List of RiceYieldData entities
        """
        logger.info(
            f"Collecting rice yield data: province={province}, year={year}, season={season}"
        )
        data = self.repository.get_yield_data(province=province, year=year, season=season)
        logger.info(f"Collected {len(data)} rice yield records")
        return data

