"""Main service orchestrating the rice yield prediction workflow (2025 optimized)."""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from ...domain.entities.season import Season
from ...domain.entities.growth_stage import GrowthStage
from ...domain.repositories.rice_yield_repository import RiceYieldRepository
from ...domain.repositories.weather_repository import WeatherRepository
from ...domain.repositories.model_repository import ModelRepository

# Use cases
from ...domain.use_cases.collect_rice_yield_data import CollectRiceYieldDataUseCase
from ...domain.use_cases.collect_weather_data import CollectWeatherDataUseCase
from ...domain.use_cases.detrend_and_label_yield import DetrendAndLabelYieldUseCase
from ...domain.use_cases.discretize_weather import DiscretizeWeatherUseCase
from ...domain.use_cases.mine_sequential_patterns import MineSequentialPatternsUseCase
from ...domain.use_cases.mine_contrast_patterns import MineContrastPatternsUseCase
from ...domain.use_cases.build_feature_matrix import BuildFeatureMatrixUseCase
from ...domain.use_cases.train_model import TrainModelUseCase
from ...domain.use_cases.predict_and_explain import PredictAndExplainUseCase

logger = logging.getLogger(__name__)


class RiceYieldPredictorService:
    """Orchestrates the full rice yield prediction pipeline with contrast pattern mining."""

    EXPORT_DIR = Path("data/exports")
    PATTERN_DIR = Path("output/latest_run")

    def __init__(
        self,
        rice_yield_repo: RiceYieldRepository,
        weather_repo: WeatherRepository,
        model_repo: ModelRepository,
        season_definitions: Dict[str, Dict[str, Any]],
        growth_stage_definitions: Dict[str, Tuple[int, int]],
    ):
        self.rice_yield_repo = rice_yield_repo
        self.weather_repo = weather_repo
        self.model_repo = model_repo

        # Convert definitions to domain entities
        self.seasons = {
            name: Season.from_dict(name, definition)
            for name, definition in season_definitions.items()
        }
        self.growth_stages = {
            name: GrowthStage(name, start_day, end_day)
            for name, (start_day, end_day) in growth_stage_definitions.items()
        }

        # Use cases (lazy-init where possible)
        self.collect_yield_uc = CollectRiceYieldDataUseCase(rice_yield_repo)
        self.collect_weather_uc = CollectWeatherDataUseCase(weather_repo)
        self.detrend_uc = DetrendAndLabelYieldUseCase()
        self.discretize_uc = DiscretizeWeatherUseCase(self.growth_stages)

        # Pattern mining (new 2025 standard)
        self.frequent_miner = MineSequentialPatternsUseCase(min_support=0.12)
        self.contrast_miner = MineContrastPatternsUseCase(
            growth_high=3.0, growth_low=4.0, min_support_target=0.1
        )

        self.build_features_uc = BuildFeatureMatrixUseCase()
        self.train_model_uc = TrainModelUseCase()

        # Runtime state
        self.predict_use_case: Optional[PredictAndExplainUseCase] = None
        self.trained_contrast_patterns: Optional[set] = None
        self.feature_names: Optional[List[str]] = None
        self.class_labels: Optional[List[str]] = None

        # Create export directories
        self.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.PATTERN_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full preprocessing pipeline and return feature-ready data."""
        logger.info("=== Starting training data preparation ===")

        # Step 1: Yield data
        yield_records = self.collect_yield_uc.execute()
        labeled_records = self.detrend_uc.execute(yield_records)

        # Export labeled yield
        labeled_df = pd.DataFrame(
            [
                {
                    "province": r.province,
                    "year": r.year,
                    "season": r.season,
                    "rice_yield": r.rice_yield,
                    "yield_class": r.yield_class.value if r.yield_class else None,
                    "expected_yield": r.expected_yield,
                    "residual": r.residual,
                }
                for r in labeled_records
            ]
        )
        labeled_df.to_csv(self.EXPORT_DIR / "01_labeled_yield.csv", index=False)

        # Step 2: Align weather
        aligned_data = []
        for rec in labeled_records:
            if rec.season not in self.seasons:
                continue
            season = self.seasons[rec.season]
            start = date(rec.year + season.year_offset, season.start_month, season.start_day)
            end = date(rec.year, season.end_month, season.end_day)

            weather = self.collect_weather_uc.execute(rec.province, start, end)
            if not weather:
                continue

            weather_df = pd.DataFrame([w.to_dict() for w in weather])
            aligned_data.append(
                {
                    "id_vụ": f"{rec.province}_{rec.year}_{rec.season}",
                    "year": rec.year,
                    "yield_class": rec.yield_class.value,
                    "daily_weather_sequence": weather_df,
                }
            )

        # Export aligned weather
        if aligned_data:
            flat_records = []
            for item in aligned_data:
                for _, row in item["daily_weather_sequence"].iterrows():
                    flat_records.append(
                        {
                            "id_vụ": item["id_vụ"],
                            "year": item["year"],
                            "yield_class": item["yield_class"],
                            **row.to_dict(),
                        }
                    )
            pd.DataFrame(flat_records).to_csv(
                self.EXPORT_DIR / "02_aligned_weather.csv", index=False
            )

        # Step 3: Discretize
        df_agg, df_sequences = self.discretize_uc.execute(aligned_data)

        # Export final features
        df_agg.to_csv(self.EXPORT_DIR / "03_aggregated_features.csv", index=False)
        df_sequences.to_csv(self.EXPORT_DIR / "04_event_sequences.csv", index=False)

        logger.info("=== Training data preparation completed ===")
        return df_agg, df_sequences

    def train_model(
        self, df_agg: pd.DataFrame, df_sequences: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train model using contrast patterns (2025 best practice)."""
        logger.info("=== Starting model training with contrast patterns ===")

        # Step 1: Mine frequent patterns
        frequent_patterns = self.frequent_miner.execute(
            df_sequences, output_dir=str(self.PATTERN_DIR / "frequent")
        )

        # Step 2: Mine contrast patterns
        contrast_df = self.contrast_miner.execute(
            df_sequences,
            frequent_patterns=frequent_patterns,
            output_dir=str(self.PATTERN_DIR / "contrast"),
        )

        if contrast_df.empty:
            logger.error("NO CONTRAST PATTERNS FOUND — THIS SHOULD NOT HAPPEN")
            raise ValueError("Contrast pattern mining failed — check thresholds")

        patterns_to_use = contrast_df["events"].tolist()

        logger.info(f"Using {len(patterns_to_use)} contrast patterns as features")

        # Step 3: Build features
        X, y, feature_names, class_labels = self.build_features_uc.execute(
            df_agg=df_agg, df_sequences=df_sequences, patterns=patterns_to_use
        )

        # Step 4: Train model
        model, metrics = self.train_model_uc.execute(X, y, class_labels)

        # Step 5: Save model with full symbolic knowledge
        model_path = self.model_repo.save_model(
            model,
            metadata={
                "training_date": pd.Timestamp.now().isoformat(),
                "n_seasons": len(df_sequences),
                "n_features": X.shape[1],
                "feature_names": feature_names,
                "class_labels": class_labels.tolist(),
                "n_contrast_patterns": len(patterns_to_use),
                "contrast_patterns": [list(p) for p in patterns_to_use],
                "metrics": metrics,
            },
        )

        # Step 6: Initialize predictor with explanation
        self.predict_use_case = PredictAndExplainUseCase(
            model=model,
            feature_names=feature_names,
            class_labels=class_labels,
            contrast_patterns=patterns_to_use,
            contrast_report_df=contrast_df if not contrast_df.empty else None,
        )

        # Store state
        self.trained_contrast_patterns = patterns_to_use
        self.feature_names = feature_names
        self.class_labels = class_labels

        logger.info(f"Model trained and saved: {model_path}")
        logger.info(
            f"Accuracy: {metrics.get('accuracy', 0):.3f} | Patterns used: {len(patterns_to_use)}"
        )
        return model, metrics

    def predict(
        self,
        province: str,
        season: str,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict yield class for current or future season."""
        if self.predict_use_case is None:
            raise RuntimeError("Model not trained. Run train_model() first.")

        year = year or date.today().year
        if season not in self.seasons:
            raise ValueError(f"Unknown season: {season}")

        season_obj = self.seasons[season]
        start = date(year + season_obj.year_offset, season_obj.start_month, season_obj.start_day)
        end = date(year, season_obj.end_month, season_obj.end_day)

        weather = self.collect_weather_uc.execute(province, start, end)
        if not weather:
            raise ValueError(f"No weather data for {province} {start.date()} to {end.date()}")

        weather_df = pd.DataFrame([w.to_dict() for w in weather])
        aligned = [
            {
                "id_vụ": f"{province}_{year}_{season}",
                "year": year,
                "yield_class": "Unknown",
                "daily_weather_sequence": weather_df,
            }
        ]

        df_agg, df_sequences = self.discretize_uc.execute(aligned)
        if df_agg is None:
            raise RuntimeError("Weather discretization failed")

        X, _, _, _ = self.build_features_uc.execute(
            df_agg=df_agg, df_sequences=df_sequences, patterns=self.trained_contrast_patterns
        )

        result = self.predict_use_case.execute(X, top_n_features=6, use_shap=True)
        result.update(
            {
                "province": province,
                "season": season,
                "year": year,
                "prediction_date": date.today().isoformat(),
            }
        )

        return result
