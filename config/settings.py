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

# FIXED THRESHOLDS
FIXED_THRESHOLDS = {
    "stage_1":        {"temp": {"Mát": (-99, 27.0),  "Vừa": (27.0, 28.5),  "Nóng": (28.5, 999)},
                       "precip": {"Khô": (-99, 70),   "Vừa": (70, 150),     "Ướt": (150, 9999)}},
    "stage_2":  {"temp": {"Mát": (-99, 26.5),  "Vừa": (26.5, 28.0),  "Nóng": (28.0, 999)},
                 "precip": {"Khô": (-99, 100),  "Vừa": (100, 200),    "Ướt": (200, 9999)}},
    "stage_3":  {"temp": {"Mát": (-99, 26.5),  "Vừa": (26.5, 27.5),  "Nóng": (27.5, 999)},
                 "precip": {"Khô": (-99, 50),   "Vừa": (50, 150),     "Ướt": (150, 9999)}},
    "stage_4":  {"temp": {"Mát": (-99, 26.2),  "Vừa": (26.2, 27.0),  "Nóng": (27.0, 999)},
                 "precip": {"Khô": (-99, 70),   "Vừa": (70, 190),     "Ướt": (190, 9999)}},
    "stage_5":      {"temp": {"Mát": (-99, 26.5),  "Vừa": (26.5, 27.5),  "Nóng": (27.5, 999)},
                     "precip": {"Khô": (-99, 80),   "Vừa": (80, 230),     "Ướt": (230, 9999)}}
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
