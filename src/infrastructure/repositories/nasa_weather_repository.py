"""NASA POWER API weather repository implementation."""

import logging
from datetime import date, datetime
from typing import List, Optional
import pandas as pd
from pathlib import Path
from ...domain.entities.weather_data import WeatherData
from ...domain.repositories.weather_repository import WeatherRepository

logger = logging.getLogger(__name__)


class NASAWeatherRepository(WeatherRepository):
    """Repository for weather data from NASA POWER API or local files."""

    def __init__(self, data_file: Optional[str] = None, use_api: bool = False):
        """
        Initialize repository.

        Args:
            data_file: Path to Excel/CSV file with weather data (for local mode)
            use_api: Whether to use NASA POWER API (not implemented, uses file)
        """
        self.data_file = Path(data_file) if data_file else None
        self.use_api = use_api

        if self.data_file and not self.data_file.exists():
            raise FileNotFoundError(f"Weather data file not found: {data_file}")

    def get_weather_data(
        self,
        province: str,
        start_date: date,
        end_date: date,
    ) -> List[WeatherData]:
        """Retrieve weather data from file or API."""
        if self.use_api:
            return self._fetch_from_api(province, start_date, end_date)
        else:
            return self._load_from_file(province, start_date, end_date)

    def _load_from_file(
        self, province: str, start_date: date, end_date: date
    ) -> List[WeatherData]:
        """Load weather data from local file."""
        if not self.data_file:
            raise ValueError("No data file specified")

        logger.info(
            f"Loading weather data from {self.data_file} "
            f"for {province} from {start_date} to {end_date}"
        )

        try:
            if self.data_file.suffix == ".xlsx":
                df = pd.read_excel(self.data_file, engine="openpyxl")
            else:
                df = pd.read_csv(self.data_file)

            # Ensure date column is datetime
            df["date"] = pd.to_datetime(df["date"])

            # Filter by province and date range
            df_filtered = df[
                (df["province"] == province)
                & (df["date"] >= pd.to_datetime(start_date))
                & (df["date"] <= pd.to_datetime(end_date))
            ].copy()

            # Handle missing values with interpolation
            numeric_cols = [
                "max_temp",
                "min_temp",
                "mean_temp",
                "precipitation_sum",
                "humidity_mean",
                "et0_mm",
            ]
            for col in numeric_cols:
                if col in df_filtered.columns:
                    df_filtered[col] = df_filtered.groupby("province")[col].transform(
                        lambda x: x.interpolate(method="time")
                    )

            # Calculate DTR if needed
            if "max_temp" in df_filtered.columns and "min_temp" in df_filtered.columns:
                df_filtered["dtr"] = (
                    df_filtered["max_temp"] - df_filtered["min_temp"]
                )

            # Convert to entities
            result = []
            for _, row in df_filtered.iterrows():
                weather_data = WeatherData(
                    province=row["province"],
                    date=row["date"].date(),
                    max_temp=float(row["max_temp"]) if pd.notna(row["max_temp"]) else None,
                    min_temp=float(row["min_temp"]) if pd.notna(row["min_temp"]) else None,
                    mean_temp=float(row["mean_temp"]) if pd.notna(row["mean_temp"]) else None,
                    precipitation_sum=(
                        float(row["precipitation_sum"])
                        if pd.notna(row["precipitation_sum"])
                        else None
                    ),
                    humidity_mean=(
                        float(row["humidity_mean"])
                        if pd.notna(row["humidity_mean"])
                        else None
                    ),
                    et0_mm=float(row["et0_mm"]) if pd.notna(row["et0_mm"]) else None,
                    weather_code=(
                        int(row["weather_code"]) if pd.notna(row.get("weather_code")) else None
                    ),
                )
                result.append(weather_data)

            logger.info(f"Loaded {len(result)} weather records")
            return result

        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            raise

    def _fetch_from_api(
        self, province: str, start_date: date, end_date: date
    ) -> List[WeatherData]:
        """Fetch weather data from NASA POWER API (placeholder for future implementation)."""
        logger.warning("API fetching not yet implemented, using file mode")
        # TODO: Implement NASA POWER API integration
        raise NotImplementedError("API fetching not yet implemented")

    def save_weather_data(self, data: List[WeatherData]) -> None:
        """Save weather data to file."""
        if not self.data_file:
            raise ValueError("No data file specified for saving")

        logger.info(f"Saving {len(data)} weather records to {self.data_file}")

        records = [
            {
                "province": d.province,
                "date": d.date,
                "max_temp": d.max_temp,
                "min_temp": d.min_temp,
                "mean_temp": d.mean_temp,
                "precipitation_sum": d.precipitation_sum,
                "humidity_mean": d.humidity_mean,
                "et0_mm": d.et0_mm,
                "weather_code": d.weather_code,
            }
            for d in data
        ]

        df = pd.DataFrame(records)
        if self.data_file.suffix == ".xlsx":
            df.to_excel(self.data_file, index=False, engine="openpyxl")
        else:
            df.to_csv(self.data_file, index=False)

        logger.info("Weather data saved successfully")

