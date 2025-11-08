"""Rice yield repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.rice_yield_data import RiceYieldData


class RiceYieldRepository(ABC):
    """Abstract repository for rice yield data access."""

    @abstractmethod
    def get_yield_data(
        self,
        province: Optional[str] = None,
        year: Optional[int] = None,
        season: Optional[str] = None,
    ) -> List[RiceYieldData]:
        """
        Retrieve rice yield data.

        Args:
            province: Filter by province name (optional)
            year: Filter by year (optional)
            season: Filter by season (optional)

        Returns:
            List of RiceYieldData entities
        """
        pass

    @abstractmethod
    def save_yield_data(self, data: List[RiceYieldData]) -> None:
        """
        Save rice yield data.

        Args:
            data: List of RiceYieldData entities to save
        """
        pass

