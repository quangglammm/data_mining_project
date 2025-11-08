"""CLI interface for rice yield prediction."""

import argparse
import logging
import sys
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Rice Yield Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the ML model")
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory to save trained model",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict rice yield")
    predict_parser.add_argument(
        "--province", type=str, required=True, help="Province name (e.g., 'An Giang')"
    )
    predict_parser.add_argument(
        "--season",
        type=str,
        required=True,
        choices=["winter_spring", "summer_autumn", "main_season"],
        help="Season name",
    )
    predict_parser.add_argument(
        "--year", type=int, default=None, help="Year (defaults to current year)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize service
    try:
        rice_yield_repo = GSORiceYieldRepository(str(RICE_DATA_FILE))
        weather_repo = NASAWeatherRepository(str(WEATHER_DATA_FILE), use_api=False)
        model_repo = FileModelRepository(args.output_dir if args.command == "train" else str(MODEL_DIR))

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

    # Execute command
    if args.command == "train":
        try:
            logger.info("Starting model training...")
            df_agg, df_sequences = service.prepare_training_data()
            model, metrics = service.train_model(df_agg, df_sequences)

            print("\n=== Training Completed ===")
            print(f"Average F1-Score (macro): {metrics['avg_f1_macro']:.4f}")
            print(f"Average Accuracy: {metrics['avg_accuracy']:.4f}")
            print("\nModel saved successfully!")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == "predict":
        try:
            logger.info(f"Predicting for {args.province} {args.season} {args.year or 'current year'}...")
            result = service.predict(
                province=args.province,
                season=args.season,
                year=args.year,
            )

            print("\n=== Prediction Result ===")
            print(f"Province: {result['province']}")
            print(f"Season: {result['season']}")
            print(f"Year: {result['year']}")
            print(f"Prediction: {result['prediction']}")

            if result.get("explanation"):
                print(f"\nExplanation:\n{result['explanation']}")

            if result.get("top_features"):
                print("\nTop Features:")
                for feature, importance in result["top_features"].items():
                    print(f"  {feature}: {importance:.4f}")

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()

