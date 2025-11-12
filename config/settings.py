"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RICE_DATA_FILE = DATA_DIR / "DBSCL_agriculture_1995_2024.csv"
WEATHER_DATA_FILE = DATA_DIR / "DBSCL_weather_1994_2025.xlsx"

# Model directory
MODEL_DIR = BASE_DIR / "models"

# Season definitions
SEASON_DEFINITIONS = {
    "winter_spring": {
        "start_month": 11,
        "start_day": 15,
        "end_month": 3,
        "end_day": 15,
        "year_offset": -1,
    },
    "summer_autumn": {
        "start_month": 4,
        "start_day": 15,
        "end_month": 8,
        "end_day": 15,
        "year_offset": 0,
    },
    "main_season": {
        "start_month": 5,
        "start_day": 15,
        "end_month": 11,
        "end_day": 30,
        "year_offset": 0,
    },
}

# Growth stage definitions
GROWTH_STAGE_DEFINITIONS = {
    "stage_1": (0, 20),  # Seedling
    "stage_2": (21, 45),  # Tillering
    "stage_3": (46, 60),  # Booting
    "stage_4": (61, 80),  # Heading
    "stage_5": (81, 105),  # Ripening
}

# ML model settings
MODEL_SETTINGS = {
    "n_splits": 5,
    "model_type": "xgboost",
    "random_state": 42,
    "min_support": 0.1,
    "minlen": 2,
    "maxlen": 4,
}

# API settings
API_SETTINGS = {
    "title": "Rice Yield Prediction API",
    "description": "API for predicting rice yield in Mekong Delta",
    "version": "1.0.0",
}

# LLM settings (for explanations)
LLM_SETTINGS = {
    "enabled": os.getenv("LLM_ENABLED", "false").lower() == "true",
    "api_url": os.getenv("LLM_API_URL", ""),
    "api_key": os.getenv("LLM_API_KEY", ""),
}

