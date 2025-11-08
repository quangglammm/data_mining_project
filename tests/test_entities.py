"""Tests for domain entities."""

import pytest
from datetime import date
from src.domain.entities.province import Province
from src.domain.entities.season import Season
from src.domain.entities.yield_class import YieldClass
from src.domain.entities.rice_yield_data import RiceYieldData
from src.domain.entities.weather_data import WeatherData
from src.domain.entities.growth_stage import GrowthStage


def test_province():
    """Test Province entity."""
    province = Province(name="An Giang", code="87", latitude=10.38, longitude=105.42)
    assert province.name == "An Giang"
    assert str(province) == "An Giang"


def test_season():
    """Test Season entity."""
    definition = {
        "start_month": 11,
        "start_day": 15,
        "end_month": 3,
        "end_day": 15,
        "year_offset": -1,
    }
    season = Season.from_dict("winter_spring", definition)
    assert season.name == "winter_spring"
    assert season.year_offset == -1


def test_yield_class():
    """Test YieldClass enum."""
    assert YieldClass.HIGH.value == "High"
    assert YieldClass.HIGH.to_vietnamese() == "Năng suất Cao"


def test_rice_yield_data():
    """Test RiceYieldData entity."""
    data = RiceYieldData(
        province="An Giang",
        year=2020,
        season="winter_spring",
        cultivated_area=100.0,
        rice_yield=6.5,
        yield_class=YieldClass.HIGH,
    )
    assert data.province == "An Giang"
    assert data.yield_class == YieldClass.HIGH


def test_weather_data():
    """Test WeatherData entity."""
    weather = WeatherData(
        province="An Giang",
        date=date(2020, 1, 1),
        max_temp=35.0,
        min_temp=25.0,
        mean_temp=30.0,
    )
    assert weather.dtr == 10.0
    assert weather.province == "An Giang"


def test_growth_stage():
    """Test GrowthStage entity."""
    stage = GrowthStage(name="stage_1", start_day=0, end_day=20)
    assert stage.day_range == (0, 20)
    assert str(stage) == "stage_1"

