"""Use case for detrending and labeling yield data."""

import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ..entities.rice_yield_data import RiceYieldData
from ..entities.yield_class import YieldClass

logger = logging.getLogger(__name__)


class DetrendAndLabelYieldUseCase:
    """Use case to detrend yield data and assign labels."""

    def __init__(self, q: int = 3):
        """
        Initialize use case.

        Args:
            q: Number of quantiles for labeling (default: 3)
        """
        self.q = q

    def robust_qcut(self, x: pd.Series, labels: List[str]) -> pd.Series:
        """
        Robust quantile cut that handles duplicates.

        Args:
            x: Series to discretize
            labels: Labels for quantiles

        Returns:
            Series with labels
        """
        try:
            return pd.qcut(x, q=self.q, labels=labels, duplicates="drop")
        except ValueError:
            # Handle case where bins are not unique
            return pd.qcut(x.rank(method="first"), q=self.q, labels=labels, duplicates="drop")

    def execute(self, data: List[RiceYieldData]) -> List[RiceYieldData]:
        """
        Execute detrending and labeling.

        Args:
            data: List of RiceYieldData entities

        Returns:
            List of RiceYieldData entities with labels assigned
        """
        logger.info(f"Detrending and labeling {len(data)} yield records")

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "province": d.province,
                    "year": d.year,
                    "season": d.season,
                    "rice_yield": d.rice_yield,
                    "cultivated_area": d.cultivated_area,
                    "rice_production": d.rice_production,
                }
                for d in data
            ]
        )

        # Filter invalid samples
        df = df[df["cultivated_area"] > 1.0].copy()

        # Group by province-season and fit linear regression
        labels = ["Low", "Medium", "High"]
        df["yield_class"] = None
        df["expected_yield"] = None
        df["residual"] = None

        for (province, season), group in df.groupby(["province", "season"]):
            if len(group) < self.q:
                logger.warning(
                    f"Insufficient data for {province}-{season}, skipping detrending"
                )
                continue

            # Fit linear regression: rice_yield ~ year
            X = group[["year"]].values
            y = group["rice_yield"].values

            model = LinearRegression()
            model.fit(X, y)

            # Compute expected yield and residuals
            expected = model.predict(X)
            residuals = y - expected

            # Assign labels based on residuals
            group_labels = self.robust_qcut(pd.Series(residuals), labels)
            group_expected = expected
            group_residuals = residuals

            # Update DataFrame
            mask = (df["province"] == province) & (df["season"] == season)
            df.loc[mask, "yield_class"] = group_labels.values
            df.loc[mask, "expected_yield"] = group_expected
            df.loc[mask, "residual"] = group_residuals

        # Drop rows without labels
        df = df.dropna(subset=["yield_class"])

        # Convert back to entities
        result = []
        for _, row in df.iterrows():
            yield_data = RiceYieldData(
                province=row["province"],
                year=int(row["year"]),
                season=row["season"],
                cultivated_area=row["cultivated_area"],
                rice_yield=row["rice_yield"],
                rice_production=row.get("rice_production"),
                yield_class=YieldClass(row["yield_class"]),
                expected_yield=float(row["expected_yield"]),
                residual=float(row["residual"]),
            )
            result.append(yield_data)

        logger.info(f"Detrended and labeled {len(result)} records")
        return result

