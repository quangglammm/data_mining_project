"""Tests for DetrendAndLabelYieldUseCase."""

import pytest
from src.domain.entities.rice_yield_data import RiceYieldData
from src.domain.entities.yield_class import YieldClass
from src.domain.use_cases.detrend_and_label_yield import DetrendAndLabelYieldUseCase


def test_detrend_and_label_basic():
    """Test basic detrending and labeling."""
    use_case = DetrendAndLabelYieldUseCase(q=3)

    # Create sample data
    data = []
    for year in range(1995, 2000):
        for season in ["winter_spring", "summer_autumn"]:
            data.append(
                RiceYieldData(
                    province="An Giang",
                    year=year,
                    season=season,
                    cultivated_area=10.0,
                    rice_yield=5.0 + (year - 1995) * 0.1,  # Trend
                )
            )

    result = use_case.execute(data)

    # Check that all records have labels
    assert len(result) > 0
    for record in result:
        assert record.yield_class in [YieldClass.HIGH, YieldClass.MEDIUM, YieldClass.LOW]
        assert record.expected_yield is not None
        assert record.residual is not None


def test_detrend_insufficient_data():
    """Test handling of insufficient data."""
    use_case = DetrendAndLabelYieldUseCase(q=3)

    # Create data with insufficient samples per group
    data = [
        RiceYieldData(
            province="An Giang",
            year=1995,
            season="winter_spring",
            cultivated_area=10.0,
            rice_yield=5.0,
        )
    ]

    result = use_case.execute(data)
    # Should handle gracefully (may return empty or skip detrending)
    assert isinstance(result, list)

