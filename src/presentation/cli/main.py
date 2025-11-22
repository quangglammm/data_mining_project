"""CLI interface for rice yield prediction (2025 optimized with data/train separation)."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ...application.services.rice_yield_predictor_service import RiceYieldPredictorService
from ...infrastructure.repositories.gso_rice_yield_repository import GSORiceYieldRepository
from ...infrastructure.repositories.nasa_weather_repository import NASAWeatherRepository
from ...infrastructure.repositories.file_model_repository import FileModelRepository

from config.settings import (
    RICE_DATA_FILE,
    WEATHER_DATA_FILE,
    MODEL_DIR,
    SEASON_DEFINITIONS,
    GROWTH_STAGE_DEFINITIONS,
)

# === Constants ===
EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Required exported files
REQUIRED_FILES = [
    EXPORT_DIR / "03_aggregated_features.csv",
    EXPORT_DIR / "04_event_sequences.csv",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def check_data_exists() -> bool:
    """Check if preprocessed data exists."""
    missing = [f for f in REQUIRED_FILES if not f.exists()]
    if missing:
        logger.error("Missing preprocessed data files:")
        for f in missing:
            logger.error(f"  → {f}")
        logger.info("Run: python -m src.cli.main prepare-data")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Rice Yield Prediction System (2025)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === prepare-data: Full preprocessing + export CSVs ===
    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Run full data pipeline: collect → detrend → align weather → discretize → export CSVs",
    )

    # === train: Load CSVs → mine contrast patterns → train model ===
    train_parser = subparsers.add_parser(
        "train", help="Train model using preprocessed data (fast, repeatable)"
    )
    train_parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR), help="Directory to save trained model"
    )

    # === predict: Fast inference ===
    predict_parser = subparsers.add_parser("predict", help="Predict yield for a province/season")
    predict_parser.add_argument("--province", type=str, required=True, help="e.g. 'An Giang'")
    predict_parser.add_argument(
        "--season",
        type=str,
        required=True,
        choices=["winter_spring", "summer_autumn", "main_season"],
        help="Season name",
    )
    predict_parser.add_argument("--year", type=int, default=None, help="Year (default: current)")

    args = parser.parse_args()

    # === Initialize repositories ===
    try:
        rice_yield_repo = GSORiceYieldRepository(str(RICE_DATA_FILE))
        weather_repo = NASAWeatherRepository(str(WEATHER_DATA_FILE), use_api=False)
        model_repo = FileModelRepository(
            args.model_dir if args.command == "train" else str(MODEL_DIR)
        )

        service = RiceYieldPredictorService(
            rice_yield_repo=rice_yield_repo,
            weather_repo=weather_repo,
            model_repo=model_repo,
            season_definitions=SEASON_DEFINITIONS,
            growth_stage_definitions=GROWTH_STAGE_DEFINITIONS,
        )
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        sys.exit(1)

    # === Command: prepare-data ===
    if args.command == "prepare-data":
        logger.info("Starting full data preparation pipeline...")
        try:
            df_agg, df_sequences = service.prepare_training_data()
            logger.info("Data preparation completed!")
            logger.info(f"   → Aggregated features: {len(df_agg)} seasons")
            logger.info(f"   → Event sequences:     {len(df_sequences)} seasons")
            logger.info(f"   → All files saved in: {EXPORT_DIR.resolve()}")
            print("\nData ready for training!")
            print("Next: python -m src.cli.main train")
        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            sys.exit(1)

    # === Command: train ===
    elif args.command == "train":
        if not check_data_exists():
            sys.exit(1)

        logger.info("Loading preprocessed data...")
        try:
            df_agg = pd.read_csv(EXPORT_DIR / "03_aggregated_features.csv")
            df_sequences = pd.read_csv(EXPORT_DIR / "04_event_sequences.csv")

            # Fix: event_sequence column is stored as string → convert back to list
            import ast

            df_sequences["event_sequence"] = df_sequences["event_sequence"].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )

            logger.info(f"Loaded {len(df_agg)} seasons for training")
            logger.info("Starting training with contrast pattern mining...")

            model, metrics = service.train_model(df_agg, df_sequences)

            print("\n" + "=" * 60)
            print(" TRAINING COMPLETED SUCCESSFULLY ")
            print("=" * 60)
            print(f" Accuracy:      {metrics.get('avg_accuracy', 0):.4f}")
            print(f" F1-macro:      {metrics.get('avg_f1_macro', 0):.4f}")
            print(f" Model saved to: {model_repo.model_dir}")
            print(f" Contrast patterns report: output/latest_run/contrast/contrast_patterns.txt")
            print("=" * 60)

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

    # === Command: predict ===
    elif args.command == "predict":
        if not check_data_exists():
            logger.warning("No preprocessed data found, but trying prediction anyway...")
        # Prediction doesn't require training data files

        try:
            result = service.predict(
                province=args.province,
                season=args.season,
                year=args.year,
            )

            print("\n" + "=" * 50)
            print(" RICE YIELD PREDICTION ")
            print("=" * 50)
            print(
                f" Location:   {result['province']} | {result['season'].replace('_', ' ').title()} {result['year']}"
            )
            print(f" Prediction: {result['prediction']} Yield")
            print("-" * 50)

            if result.get("explanation"):
                print("Explanation:")
                print(result["explanation"])

            if result.get("top_features"):
                print("\nKey Weather Patterns Driving Prediction:")
                for feat, val in list(result["top_features"].items())[:6]:
                    print(f"  • {feat}: {val:+.4f}")

            print("=" * 50)

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
