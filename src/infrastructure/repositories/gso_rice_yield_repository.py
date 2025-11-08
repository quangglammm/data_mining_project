"""GSO (General Statistics Office) rice yield repository implementation."""

import logging
from typing import List, Optional
import pandas as pd
from pathlib import Path
from ...domain.entities.rice_yield_data import RiceYieldData
from ...domain.repositories.rice_yield_repository import RiceYieldRepository

logger = logging.getLogger(__name__)


class GSORiceYieldRepository(RiceYieldRepository):
    """Repository for rice yield data from GSO CSV files."""

    def __init__(self, data_file: str):
        """
        Initialize repository.

        Args:
            data_file: Path to CSV file with rice yield data
        """
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Rice yield data file not found: {data_file}")

    def get_yield_data(
        self,
        province: Optional[str] = None,
        year: Optional[int] = None,
        season: Optional[str] = None,
    ) -> List[RiceYieldData]:
        """Retrieve rice yield data from CSV file."""
        logger.info(f"Loading rice yield data from {self.data_file}")

        try:
            df = pd.read_csv(self.data_file)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

        # Convert wide to long format
        df_long = pd.wide_to_long(
            df,
            stubnames=["cultivated_area", "rice_yield", "rice_production"],
            i=["province", "year"],
            j="season",
            sep="_",
            suffix=".*",
        ).reset_index()

        # Apply filters
        if province:
            df_long = df_long[df_long["province"] == province]
        if year:
            df_long = df_long[df_long["year"] == year]
        if season:
            df_long = df_long[df_long["season"] == season]

        # Convert to entities
        result = []
        for _, row in df_long.iterrows():
            yield_data = RiceYieldData(
                province=row["province"],
                year=int(row["year"]),
                season=row["season"],
                cultivated_area=float(row["cultivated_area"]),
                rice_yield=float(row["rice_yield"]),
                rice_production=(
                    float(row["rice_production"])
                    if pd.notna(row["rice_production"])
                    else None
                ),
            )
            result.append(yield_data)

        logger.info(f"Loaded {len(result)} rice yield records")
        return result

    def save_yield_data(self, data: List[RiceYieldData]) -> None:
        """Save rice yield data to CSV file."""
        logger.info(f"Saving {len(data)} rice yield records to {self.data_file}")

        # Convert entities to DataFrame
        records = [
            {
                "province": d.province,
                "year": d.year,
                "season": d.season,
                "cultivated_area": d.cultivated_area,
                "rice_yield": d.rice_yield,
                "rice_production": d.rice_production,
                "yield_class": d.yield_class.value if d.yield_class else None,
                "expected_yield": d.expected_yield,
                "residual": d.residual,
            }
            for d in data
        ]

        df = pd.DataFrame(records)
        df.to_csv(self.data_file, index=False)
        logger.info("Rice yield data saved successfully")

