"""Use case for discretizing weather data into events."""

import logging
from datetime import timedelta
from typing import List, Dict, Tuple, Any
import pandas as pd
from ..entities.growth_stage import GrowthStage

logger = logging.getLogger(__name__)


class DiscretizeWeatherUseCase:
    """Use case to discretize weather data into event sequences."""

    def __init__(self, growth_stages: Dict[str, GrowthStage], fixed_thresholds: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]):
        """
        Initialize use case.

        Args:
            growth_stages: Dictionary mapping stage names to GrowthStage entities
        """
        self.growth_stages = growth_stages
        self.fixed_thresholds = fixed_thresholds

    def _get_label(self, value, thresholds):
        for label, (low, high) in thresholds.items():
            if low <= value < high:
                return label
        return list(thresholds.keys())[-1]

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
                stages_for_season[f"{stage_name}_avg_et0"] = stage_weather["et0_mm"].mean(
                )

                # Create event label
                temp_event = self._get_label(
                    avg_temp,
                    self.fixed_thresholds[stage_name]["temp"],
                )

                precip_event = self._get_label(
                    total_precip,
                    self.fixed_thresholds[stage_name]["precip"],
                )

                if temp_event and precip_event:
                    sequence_for_season.append(
                        f"{stage_name}_{temp_event}-{precip_event}")

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
