"""Main service orchestrating the rice yield prediction workflow (2025 optimized)."""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

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
from ...domain.use_cases.mine_low_yield_patterns import MineLowYieldPatternsUseCase
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
        self.frequent_miner = MineSequentialPatternsUseCase(min_support=0.12, minlen=2, maxlen=3)
        self.low_yield_miner = MineLowYieldPatternsUseCase()

        self.build_features_uc = BuildFeatureMatrixUseCase()
        self.train_model_uc = TrainModelUseCase()

        # Runtime state
        self.predict_use_case: Optional[PredictAndExplainUseCase] = None
        self.trained_contrast_patterns: Optional[Set[Tuple[str, ...]]] = None
        self.feature_names: Optional[List[str]] = None
        self.class_labels: Optional[np.ndarray] = None
        self.contrast_report_df: Optional[pd.DataFrame] = None

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
        labeled_df.to_csv(self.EXPORT_DIR / "01_labeled_yield_v2.csv", index=False)

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

            weather_df = pd.DataFrame(weather)

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
                self.EXPORT_DIR / "02_aligned_weather_v2.csv", index=False
            )

        # Step 3: Discretize
        df_agg, df_sequences = self.discretize_uc.execute(aligned_data)

        # Export final features
        df_agg.to_csv(self.EXPORT_DIR / "03_aggregated_features_v2.csv", index=False)
        df_sequences.to_csv(self.EXPORT_DIR / "04_event_sequences_v2.csv", index=False)

        logger.info("=== Training data preparation completed ===")
        return df_agg, df_sequences

    def train_model(
        self, df_agg: Optional[pd.DataFrame] = None, df_sequences: Optional[pd.DataFrame] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model using contrast patterns (2025 best practice).

        Args:
            df_agg: Aggregated features (if None, will call prepare_training_data)
            df_sequences: Event sequences (if None, will call prepare_training_data)

        Returns:
            Tuple of (model, metrics)
        """
        logger.info("=== Starting model training with contrast patterns ===")

        # If data not provided, prepare it
        if df_agg is None or df_sequences is None:
            logger.info("Training data not provided, preparing from scratch...")
            df_agg, df_sequences = self.prepare_training_data()

        # Step 1: Mine frequent patterns (high-yield patterns)
        high_yield_patterns: Set[Tuple[str, ...]] = self.frequent_miner.execute(
            df_sequences, output_dir=str(self.PATTERN_DIR / "frequent")
        )

        if not high_yield_patterns:
            raise ValueError("No high-yield patterns found by frequent miner!")

        logger.info(
            f"Discovered {len(high_yield_patterns)} high-yield pattern(s) from frequent miner"
        )

        # Step 2: Mine low-yield destructive patterns
        logger.info("MINING LOW-YIELD DESTRUCTIVE MECHANISMS (contrast + rare + breakers)...")
        low_report = self.low_yield_miner.execute(
            df_sequences=df_sequences,
            high_golden_patterns=high_yield_patterns,
            output_dir=str(self.PATTERN_DIR / "destructive"),
        )

        # Extract low-yield patterns
        contrast_events = low_report.get("contrast_events", set())
        destructive_patterns = low_report.get("destructive_patterns", set())
        breaker_events = low_report.get("breaker_events", set())

        logger.info(f"   • {len(contrast_events)} Contrast Events")
        logger.info(f"   • {len(destructive_patterns)} Rare Catastrophic Patterns")
        logger.info(f"   • {len(breaker_events)} Golden Sequence Breakers")

        # Step 3: Combine all patterns
        all_candidate_patterns = set()
        all_candidate_patterns.update(high_yield_patterns)
        all_candidate_patterns.update(contrast_events)
        all_candidate_patterns.update(destructive_patterns)
        all_candidate_patterns.update(breaker_events)

        logger.info(f"Selected {len(all_candidate_patterns)} highest-impact patterns")

        # Step 4: Build feature matrix (numerical + pattern features)
        X, y, feature_names, class_labels = self.build_features_uc.execute(
            df_agg=df_agg, df_sequences=df_sequences, patterns=all_candidate_patterns
        )

        logger.info(f"Feature matrix built: {X.shape[0]} samples × {X.shape[1]} features")

        # Step 5: Train model
        model, metrics = self.train_model_uc.execute(X, y, class_labels)

        # Step 6: Create contrast report DataFrame for explanations
        # This should ideally come from your mining use cases
        contrast_report_df = self._create_contrast_report(
            high_yield_patterns, contrast_events, destructive_patterns, breaker_events
        )

        # Step 7: Save model with full metadata (including contrast report)
        model_path = self.model_repo.save_model(
            model,
            metadata={
                "training_date": pd.Timestamp.now().isoformat(),
                "n_seasons": len(df_sequences),
                "n_features": X.shape[1],
                "feature_names": feature_names,
                "class_labels": class_labels.tolist(),
                "n_contrast_patterns": len(all_candidate_patterns),
                "contrast_patterns": [list(p) for p in all_candidate_patterns],
                "metrics": metrics,
                # Save contrast report as serializable dict
                "contrast_report": (
                    contrast_report_df.to_dict(orient="records")
                    if contrast_report_df is not None
                    else None
                ),
            },
        )

        # Step 8: Initialize predictor with ALL features
        self.predict_use_case = PredictAndExplainUseCase(
            model=model,
            feature_names=feature_names,  # ✅ ALL features (numerical + patterns)
            class_labels=class_labels,
            contrast_patterns=all_candidate_patterns,
            contrast_report_df=contrast_report_df,
        )

        # Step 9: Store state for prediction
        self.trained_contrast_patterns = all_candidate_patterns
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.contrast_report_df = contrast_report_df

        logger.info(f"✅ Model trained and saved: {model_path}")
        logger.info(
            f"   Accuracy: {metrics.get('accuracy', 0):.3f} | "
            f"Patterns: {len(all_candidate_patterns)} | "
            f"Features: {len(feature_names)}"
        )

        return model, metrics

    def _create_contrast_report(
        self,
        high_patterns: Set[Tuple[str, ...]],
        contrast_events: Set[Tuple[str, ...]],
        destructive_patterns: Set[Tuple[str, ...]],
        breaker_events: Set[Tuple[str, ...]],
    ) -> pd.DataFrame:
        """
        Create a contrast report DataFrame for pattern explanations.

        This is a simplified version - ideally this should come from your mining use cases.
        """
        records = []

        # High-yield patterns
        for pattern in high_patterns:
            records.append(
                {
                    "events": list(pattern),
                    "type": "High",
                    "growth_rate": 2.5,  # Placeholder - should come from actual mining
                    "strength": "strong",
                }
            )

        # Contrast events (appear more in Low yield)
        for pattern in contrast_events:
            records.append(
                {"events": list(pattern), "type": "Low", "growth_rate": 2.0, "strength": "moderate"}
            )

        # Destructive patterns
        for pattern in destructive_patterns:
            records.append(
                {"events": list(pattern), "type": "Low", "growth_rate": 3.0, "strength": "strong"}
            )

        # Breaker events
        for pattern in breaker_events:
            records.append(
                {"events": list(pattern), "type": "Low", "growth_rate": 1.8, "strength": "moderate"}
            )

        return pd.DataFrame(records)

    def load_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a previously trained model from disk.

        Args:
            model_path: Path to model file. If None, loads the latest model.

        Returns:
            Dictionary with model metadata
        """
        logger.info("=== Loading trained model ===")

        # Load model and metadata
        loaded_data = self.model_repo.load_model(model_path)

        model = loaded_data["model"]
        metadata = loaded_data["metadata"]

        # Reconstruct patterns as tuples
        contrast_patterns = {tuple(p) for p in metadata["contrast_patterns"]}

        # Load contrast report from metadata
        contrast_report_df = None
        if "contrast_report" in metadata and metadata["contrast_report"] is not None:
            contrast_report_df = pd.DataFrame(metadata["contrast_report"])
            # Convert events back to lists if they were saved as strings
            if "events" in contrast_report_df.columns:
                contrast_report_df["events"] = contrast_report_df["events"].apply(
                    lambda x: x if isinstance(x, list) else list(x)
                )
        else:
            # Fallback: create a basic contrast report
            logger.warning("No contrast report in metadata, creating basic report")
            contrast_report_df = self._create_contrast_report(
                high_patterns=contrast_patterns,
                contrast_events=set(),
                destructive_patterns=set(),
                breaker_events=set(),
            )

        # Reconstruct the PredictAndExplainUseCase
        self.predict_use_case = PredictAndExplainUseCase(
            model=model,
            feature_names=metadata["feature_names"],
            class_labels=np.array(metadata["class_labels"]),
            contrast_patterns=contrast_patterns,
            contrast_report_df=contrast_report_df,
        )

        # Store state
        self.trained_contrast_patterns = contrast_patterns
        self.feature_names = metadata["feature_names"]
        self.class_labels = np.array(metadata["class_labels"])
        self.contrast_report_df = contrast_report_df

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Features: {len(self.feature_names)}")
        logger.info(f"   Patterns: {len(self.trained_contrast_patterns)}")
        logger.info(f"   Classes: {self.class_labels.tolist()}")
        logger.info(f"   Training date: {metadata.get('training_date', 'unknown')}")
        logger.info(f"   Accuracy: {metadata.get('metrics', {}).get('accuracy', 'N/A')}")

        return metadata

    def predict(
        self,
        province: str,
        season: str,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict yield class for current or future season.

        Args:
            province: Province name
            season: Season name
            year: Year to predict (defaults to current year)

        Returns:
            Dictionary with prediction and explanation
        """
        if self.predict_use_case is None:
            raise RuntimeError(
                "Model not trained or loaded. Run train_model() or load_model() first."
            )

        year = year or date.today().year

        if season not in self.seasons:
            raise ValueError(f"Unknown season: {season}. Available: {list(self.seasons.keys())}")

        logger.info(f"=== Predicting for {province} - {season} {year} ===")

        # Step 1: Get season boundaries
        season_obj = self.seasons[season]
        start = date(year + season_obj.year_offset, season_obj.start_month, season_obj.start_day)
        end = date(year, season_obj.end_month, season_obj.end_day)

        # Step 2: Fetch weather data
        weather = self.collect_weather_uc.execute(province, start, end)
        if not weather:
            raise ValueError(
                f"No weather data available for {province} from {start} to {end}. "
                "Weather data may not be available yet for future dates."
            )

        logger.info(f"   Retrieved {len(weather)} days of weather data")

        # Step 3: Prepare data for prediction
        weather_df = pd.DataFrame(weather)
        aligned = [
            {
                "id_vụ": f"{province}_{year}_{season}",
                "year": year,
                "yield_class": "Unknown",  # Placeholder for prediction
                "daily_weather_sequence": weather_df,
            }
        ]

        # Step 4: Discretize weather
        df_agg, df_sequences = self.discretize_uc.execute(aligned)
        if df_agg.empty:
            raise RuntimeError("Weather discretization failed - no features generated")

        # Step 5: Build feature matrix (must match training features)
        X, _, _, _ = self.build_features_uc.execute(
            df_agg=df_agg, df_sequences=df_sequences, patterns=self.trained_contrast_patterns
        )

        # Verify feature alignment
        if X.shape[1] != len(self.feature_names):
            logger.warning(
                f"Feature mismatch: got {X.shape[1]} features, expected {len(self.feature_names)}"
            )

        # Step 6: Make prediction with explanation
        result = self.predict_use_case.execute(X, top_n_features=10, use_shap=True)

        # Step 7: Add metadata
        result.update(
            {
                "province": province,
                "season": season,
                "year": year,
                "prediction_date": date.today().isoformat(),
                "weather_days": len(weather),
            }
        )

        logger.info(
            f"✅ Prediction: {result['prediction']} " f"(confidence: {result['confidence']:.2%})"
        )

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current loaded model."""
        if self.predict_use_case is None:
            return {"status": "No model loaded"}

        return {
            "status": "Model ready",
            "n_features": len(self.feature_names),
            "n_patterns": len(self.trained_contrast_patterns),
            "classes": self.class_labels.tolist(),
            "feature_breakdown": {
                "numerical": sum(1 for f in self.feature_names if not f.startswith("pat_")),
                "patterns": sum(1 for f in self.feature_names if f.startswith("pat_")),
            },
        }
