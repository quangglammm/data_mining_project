"""Main service for rice yield prediction workflow."""

import logging
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from ...domain.entities.season import Season
from ...domain.entities.growth_stage import GrowthStage
from ...domain.entities.yield_class import YieldClass
from ...domain.repositories.rice_yield_repository import RiceYieldRepository
from ...domain.repositories.weather_repository import WeatherRepository
from ...domain.repositories.model_repository import ModelRepository
from ...domain.use_cases.collect_rice_yield_data import CollectRiceYieldDataUseCase
from ...domain.use_cases.collect_weather_data import CollectWeatherDataUseCase
from ...domain.use_cases.detrend_and_label_yield import DetrendAndLabelYieldUseCase
from ...domain.use_cases.discretize_weather import DiscretizeWeatherUseCase
from ...domain.use_cases.mine_sequential_patterns import MineSequentialPatternsUseCase
from ...domain.use_cases.build_feature_matrix import BuildFeatureMatrixUseCase
from ...domain.use_cases.train_model import TrainModelUseCase
from ...domain.use_cases.predict_and_explain import PredictAndExplainUseCase

logger = logging.getLogger(__name__)


class RiceYieldPredictorService:
    """Main service orchestrating the rice yield prediction workflow."""

    def __init__(
        self,
        rice_yield_repo: RiceYieldRepository,
        weather_repo: WeatherRepository,
        model_repo: ModelRepository,
        season_definitions: Dict[str, Dict[str, Any]],
        growth_stage_definitions: Dict[str, Tuple[int, int]],
    ):
        """
        Initialize service.

        Args:
            rice_yield_repo: Repository for rice yield data
            weather_repo: Repository for weather data
            model_repo: Repository for ML models
            season_definitions: Dictionary of season definitions
            growth_stage_definitions: Dictionary of growth stage definitions
        """
        self.rice_yield_repo = rice_yield_repo
        self.weather_repo = weather_repo
        self.model_repo = model_repo

        # Convert season definitions to entities
        self.seasons = {
            name: Season.from_dict(name, definition)
            for name, definition in season_definitions.items()
        }

        # Convert growth stage definitions to entities
        self.growth_stages = {
            name: GrowthStage(name, start_day, end_day)
            for name, (start_day, end_day) in growth_stage_definitions.items()
        }

        # Initialize use cases
        self.collect_yield_use_case = CollectRiceYieldDataUseCase(rice_yield_repo)
        self.collect_weather_use_case = CollectWeatherDataUseCase(weather_repo)
        self.detrend_use_case = DetrendAndLabelYieldUseCase()
        self.discretize_use_case = DiscretizeWeatherUseCase(self.growth_stages)
        self.mine_patterns_use_case = MineSequentialPatternsUseCase()
        self.build_features_use_case = BuildFeatureMatrixUseCase()
        self.train_model_use_case = TrainModelUseCase()
        self.predict_use_case = None  # Will be set after training
        self.trained_patterns = None  # Store patterns from training
        self.feature_names = None  # Store feature names from training

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data by executing the full pipeline.

        Returns:
            Tuple of (aggregated_features_df, event_sequences_df)
        """
        logger.info("Starting training data preparation")

        # Step 1: Collect rice yield data
        yield_data = self.collect_yield_use_case.execute()

        # Step 2: Detrend and label
        labeled_data = self.detrend_use_case.execute(yield_data)

        # Step 3: Align weather data
        aligned_data = []
        for yield_record in labeled_data:
            province = yield_record.province
            year = yield_record.year
            season_name = yield_record.season

            if season_name not in self.seasons:
                logger.warning(f"Unknown season: {season_name}, skipping")
                continue

            season = self.seasons[season_name]
            year_offset = season.year_offset

            start_date = date(year + year_offset, season.start_month, season.start_day)
            end_date = date(year, season.end_month, season.end_day)

            # Collect weather data
            weather_data = self.collect_weather_use_case.execute(
                province, start_date, end_date
            )

            if not weather_data:
                logger.warning(
                    f"No weather data for {province} {year} {season_name}, skipping"
                )
                continue

            # Convert weather data to DataFrame
            weather_df = pd.DataFrame(
                [
                    {
                        "date": w.date,
                        "max_temp": w.max_temp,
                        "min_temp": w.min_temp,
                        "mean_temp": w.mean_temp,
                        "precipitation_sum": w.precipitation_sum,
                        "humidity_mean": w.humidity_mean,
                        "et0_mm": w.et0_mm,
                        "weather_code": w.weather_code,
                    }
                    for w in weather_data
                ]
            )

            aligned_data.append(
                {
                    "id_vụ": f"{province}_{year}_{season_name}",
                    "year": year,
                    "yield_class": yield_record.yield_class.value,
                    "daily_weather_sequence": weather_df,
                }
            )

        # Step 4: Discretize weather
        df_agg, df_sequences = self.discretize_use_case.execute(aligned_data)

        logger.info("Training data preparation completed")
        return df_agg, df_sequences

    def train_model(
        self, df_agg: pd.DataFrame, df_sequences: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the ML model.

        Args:
            df_agg: Aggregated features DataFrame
            df_sequences: Event sequences DataFrame

        Returns:
            Tuple of (trained_model, evaluation_metrics)
        """
        logger.info("Starting model training")

        # Step 5: Mine sequential patterns
        all_patterns = self.mine_patterns_use_case.execute(df_sequences)
        self.trained_patterns = all_patterns  # Store for prediction

        # Step 6: Build feature matrix
        X, y, feature_names, class_labels = self.build_features_use_case.execute(
            df_agg, df_sequences, all_patterns
        )
        self.feature_names = feature_names  # Store for prediction

        # Step 7: Train model
        model, metrics = self.train_model_use_case.execute(X, y, class_labels)

        # Step 8: Save model
        model_path = self.model_repo.save_model(
            model,
            metadata={
                "feature_names": feature_names,
                "class_labels": class_labels.tolist(),
                "patterns": [list(p) for p in all_patterns],  # Store patterns
            },
        )

        # Initialize predict use case
        self.predict_use_case = PredictAndExplainUseCase(
            model, feature_names, class_labels
        )

        logger.info(f"Model training completed, saved to {model_path}")
        return model, metrics

    def predict(
        self,
        province: str,
        season: str,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict yield class for a given province and season.

        Args:
            province: Province name
            season: Season name
            year: Year (defaults to current year if not provided)

        Returns:
            Dictionary with prediction and explanation
        """
        if self.predict_use_case is None:
            raise ValueError("Model not trained. Call train_model() first.")

        if year is None:
            year = date.today().year

        if season not in self.seasons:
            raise ValueError(f"Unknown season: {season}")

        logger.info(f"Predicting for {province} {season} {year}")

        # Get season dates
        season_obj = self.seasons[season]
        year_offset = season_obj.year_offset
        start_date = date(year + year_offset, season_obj.start_month, season_obj.start_day)
        end_date = date(year, season_obj.end_month, season_obj.end_day)

        # Collect weather data
        weather_data = self.collect_weather_use_case.execute(
            province, start_date, end_date
        )

        if not weather_data:
            raise ValueError(f"No weather data available for {province} {start_date} to {end_date}")

        # Convert to DataFrame and discretize
        weather_df = pd.DataFrame(
            [
                {
                    "date": w.date,
                    "max_temp": w.max_temp,
                    "min_temp": w.min_temp,
                    "mean_temp": w.mean_temp,
                    "precipitation_sum": w.precipitation_sum,
                    "humidity_mean": w.humidity_mean,
                    "et0_mm": w.et0_mm,
                }
                for w in weather_data
            ]
        )

        # Create aligned data structure
        aligned_data = [
            {
                "id_vụ": f"{province}_{year}_{season}",
                "year": year,
                "yield_class": "Unknown",  # Not needed for prediction
                "daily_weather_sequence": weather_df,
            }
        ]

        # Discretize
        df_agg, df_sequences = self.discretize_use_case.execute(aligned_data)

        if df_agg is None or df_sequences is None:
            raise ValueError("Failed to process weather data")

        # Build features using same patterns from training
        if self.trained_patterns is None:
            raise ValueError(
                "Patterns not available. Model must be trained before prediction."
            )

        # Build feature matrix using stored patterns
        X, _, _, _ = self.build_features_use_case.execute(
            df_agg, df_sequences, self.trained_patterns
        )

        # Make prediction
        result = self.predict_use_case.execute(X, top_n_features=5, use_shap=True)

        # Add context information
        result["province"] = province
        result["season"] = season
        result["year"] = year

        return result

