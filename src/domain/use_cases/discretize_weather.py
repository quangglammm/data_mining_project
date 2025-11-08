"""Use case for discretizing weather data into events."""

import logging
from datetime import timedelta
from typing import List, Dict, Tuple, Any
import pandas as pd
from ..entities.weather_data import WeatherData
from ..entities.growth_stage import GrowthStage
from ..entities.season import Season

logger = logging.getLogger(__name__)


class DiscretizeWeatherUseCase:
    """Use case to discretize weather data into event sequences."""

    def __init__(self, growth_stages: Dict[str, GrowthStage]):
        """
        Initialize use case.

        Args:
            growth_stages: Dictionary mapping stage names to GrowthStage entities
        """
        self.growth_stages = growth_stages

    def _calculate_thresholds(
        self, weather_sequences: List[pd.DataFrame]
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        """
        Calculate discretization thresholds for temperature and precipitation.

        Args:
            weather_sequences: List of DataFrames with daily weather data

        Returns:
            Tuple of (temp_thresholds, precip_thresholds) dictionaries
        """
        temp_agg_data = {stage: [] for stage in self.growth_stages.keys()}
        precip_agg_data = {stage: [] for stage in self.growth_stages.keys()}

        for daily_seq in weather_sequences:
            if daily_seq.empty:
                continue
            start_date = pd.to_datetime(daily_seq["date"]).min()

            for stage_name, stage in self.growth_stages.items():
                stage_start_date = start_date + timedelta(days=stage.start_day)
                stage_end_date = start_date + timedelta(days=stage.end_day)
                stage_weather = daily_seq[
                    (pd.to_datetime(daily_seq["date"]) >= stage_start_date)
                    & (pd.to_datetime(daily_seq["date"]) <= stage_end_date)
                ]
                if stage_weather.empty:
                    continue

                temp_agg_data[stage_name].append(stage_weather["mean_temp"].mean())
                precip_agg_data[stage_name].append(
                    stage_weather["precipitation_sum"].sum()
                )

        temp_thresholds = {
            stage: (pd.Series(data).quantile(1 / 3), pd.Series(data).quantile(2 / 3))
            for stage, data in temp_agg_data.items()
            if data
        }
        precip_thresholds = {
            stage: (pd.Series(data).quantile(1 / 3), pd.Series(data).quantile(2 / 3))
            for stage, data in precip_agg_data.items()
            if data
        }

        return temp_thresholds, precip_thresholds

    def _get_event_label(
        self,
        temp: float,
        precip: float,
        t_thresh: Tuple[float, float],
        p_thresh: Tuple[float, float],
    ) -> str:
        """
        Get event label from temperature and precipitation.

        Args:
            temp: Mean temperature
            precip: Total precipitation
            t_thresh: Temperature thresholds (low, high)
            p_thresh: Precipitation thresholds (low, high)

        Returns:
            Event label string
        """
        if pd.isna(temp) or pd.isna(precip):
            return None
        if pd.isna(t_thresh[0]) or pd.isna(p_thresh[0]):
            return None

        if temp <= t_thresh[0]:
            t_label = "Mát"  # Cool
        elif temp <= t_thresh[1]:
            t_label = "Vừa"  # Moderate
        else:
            t_label = "Nóng"  # Hot

        if precip <= p_thresh[0]:
            p_label = "Khô"  # Dry
        elif precip <= p_thresh[1]:
            p_label = "Vừa"  # Moderate
        else:
            p_label = "Ướt"  # Wet

        return f"{t_label}-{p_label}"

    def execute(
        self,
        aligned_data: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute discretization.

        Args:
            aligned_data: List of dictionaries with keys:
                - 'id_vụ': identifier
                - 'year': year
                - 'yield_class': yield class
                - 'daily_weather_sequence': DataFrame with daily weather

        Returns:
            Tuple of (aggregated_features_df, event_sequences_df)
        """
        logger.info(f"Discretizing weather for {len(aligned_data)} seasons")

        # Extract weather sequences
        weather_sequences = [
            row["daily_weather_sequence"] for row in aligned_data
        ]

        # Calculate thresholds
        temp_thresholds, precip_thresholds = self._calculate_thresholds(
            weather_sequences
        )

        aggregated_stages = []
        event_sequences = []

        for row in aligned_data:
            daily_seq = row["daily_weather_sequence"]
            if daily_seq.empty:
                continue

            start_date = pd.to_datetime(daily_seq["date"]).min()

            stages_for_season = {
                "id_vụ": row["id_vụ"],
                "year": row["year"],
                "yield_class": row["yield_class"],
            }
            sequence_for_season = []

            for stage_name, stage in self.growth_stages.items():
                stage_start_date = start_date + timedelta(days=stage.start_day)
                stage_end_date = start_date + timedelta(days=stage.end_day)
                stage_weather = daily_seq[
                    (pd.to_datetime(daily_seq["date"]) >= stage_start_date)
                    & (pd.to_datetime(daily_seq["date"]) <= stage_end_date)
                ]

                if stage_weather.empty:
                    continue

                # Calculate numeric features
                avg_temp = stage_weather["mean_temp"].mean()
                total_precip = stage_weather["precipitation_sum"].sum()

                stages_for_season[f"{stage_name}_avg_temp"] = avg_temp
                stages_for_season[f"{stage_name}_total_precip"] = total_precip
                stages_for_season[f"{stage_name}_count_heat_days"] = (
                    stage_weather["max_temp"] > 35
                ).sum()
                stages_for_season[f"{stage_name}_avg_et0"] = stage_weather[
                    "et0_mm"
                ].mean()

                # Create event label
                event = self._get_event_label(
                    avg_temp,
                    total_precip,
                    temp_thresholds.get(stage_name, (0, 0)),
                    precip_thresholds.get(stage_name, (0, 0)),
                )

                if event:
                    sequence_for_season.append(f"{stage_name}_{event}")

            aggregated_stages.append(stages_for_season)
            if sequence_for_season:
                event_sequences.append(
                    {
                        "id_vụ": row["id_vụ"],
                        "year": row["year"],
                        "yield_class": row["yield_class"],
                        "event_sequence": sequence_for_season,
                    }
                )

        if not aggregated_stages or not event_sequences:
            logger.error("No data after discretization")
            return None, None

        df_agg = pd.DataFrame(aggregated_stages)
        df_sequences = pd.DataFrame(event_sequences)

        logger.info(f"Created {len(df_sequences)} event sequences")
        return df_agg, df_sequences

