"""CLI interface for rice yield prediction (2025 optimized with data/train separation)."""

import argparse
import logging
import sys
from pathlib import Path
import ast

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
    EXPORT_DIR / "03_aggregated_features_v2.csv",
    EXPORT_DIR / "04_event_sequences_v2.csv",
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
        logger.error("❌ Missing preprocessed data files:")
        for f in missing:
            logger.error(f"  → {f}")
        logger.info("💡 Run: python -m src.presentation.cli.main prepare-data")
        return False
    return True


def initialize_service(model_dir: str = None) -> RiceYieldPredictorService:
    """Initialize the service with all dependencies."""
    rice_yield_repo = GSORiceYieldRepository(str(RICE_DATA_FILE))
    weather_repo = NASAWeatherRepository(str(WEATHER_DATA_FILE), use_api=False)
    model_repo = FileModelRepository(model_dir or str(MODEL_DIR))

    service = RiceYieldPredictorService(
        rice_yield_repo=rice_yield_repo,
        weather_repo=weather_repo,
        model_repo=model_repo,
        season_definitions=SEASON_DEFINITIONS,
        growth_stage_definitions=GROWTH_STAGE_DEFINITIONS,
    )

    return service


def command_prepare_data(args):
    """Handle the 'prepare-data' command."""
    logger.info("=" * 70)
    logger.info("STARTING FULL DATA PREPARATION PIPELINE")
    logger.info("=" * 70)

    try:
        service = initialize_service()
        df_agg, df_sequences = service.prepare_training_data()

        logger.info("=" * 70)
        logger.info("✅ DATA PREPARATION COMPLETED!")
        logger.info("=" * 70)
        logger.info(f"   📊 Aggregated features: {len(df_agg)} seasons")
        logger.info(f"   🔄 Event sequences:     {len(df_sequences)} seasons")
        logger.info(f"   📁 Files saved in:      {EXPORT_DIR.resolve()}")
        logger.info("")
        logger.info("💡 Next step: python -m src.presentation.cli.main train")

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}", exc_info=True)
        sys.exit(1)


def command_train(args):
    """Handle the 'train' command."""
    if not check_data_exists():
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING WITH CONTRAST PATTERNS")
    logger.info("=" * 70)

    try:
        # Initialize service
        service = initialize_service(model_dir=args.model_dir)

        # Load preprocessed data
        logger.info("📂 Loading preprocessed data...")
        df_agg = pd.read_csv(EXPORT_DIR / "03_aggregated_features_v2.csv")
        df_sequences = pd.read_csv(EXPORT_DIR / "04_event_sequences_v2.csv")

        # Fix: event_sequence column is stored as string → convert back to list
        df_sequences["event_sequence"] = df_sequences["event_sequence"].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else []
        )

        logger.info(f"   Loaded {len(df_agg)} seasons for training")
        logger.info("")

        # Train model
        model, metrics = service.train_model(df_agg, df_sequences)

        # Display results
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"📊 Performance Metrics:")
        logger.info(f"   Accuracy:  {metrics.get('avg_accuracy', metrics.get('accuracy', 0)):.4f}")
        logger.info(
            f"   Precision: {metrics.get('avg_precision', metrics.get('precision', 0)):.4f}"
        )
        logger.info(f"   Recall:    {metrics.get('avg_recall', metrics.get('recall', 0)):.4f}")
        logger.info(f"   F1-macro:  {metrics.get('avg_f1_macro', metrics.get('f1', 0)):.4f}")

        # Show model info
        info = service.get_model_info()
        logger.info("")
        logger.info(f"🔧 Model Configuration:")
        logger.info(f"   Total Features:     {info['n_features']}")
        logger.info(f"     • Numerical:      {info['feature_breakdown']['numerical']}")
        logger.info(f"     • Patterns:       {info['feature_breakdown']['patterns']}")
        logger.info(f"   Contrast Patterns:  {info['n_patterns']}")
        logger.info(f"   Classes:            {', '.join(info['classes'])}")

        logger.info("")
        logger.info(f"💾 Model saved to:      {args.model_dir}")
        logger.info(f"📋 Pattern report:      output/latest_run/")
        logger.info("=" * 70)
        logger.info("")
        logger.info(
            "💡 Next step: python -m src.presentation.cli.main predict --province 'An Giang' --season 'winter_spring' --year 2020"
        )

    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


def command_predict(args):
    """Handle the 'predict' command."""
    logger.info("=" * 70)
    logger.info("STARTING PREDICTION")
    logger.info("=" * 70)

    try:
        # Initialize service
        service = initialize_service(
            model_dir=args.model_dir if hasattr(args, "model_dir") else None
        )

        # Load model (CRITICAL for CLI workflow)
        model_path = args.model_path if hasattr(args, "model_path") and args.model_path else None
        logger.info(f"📂 Loading model from: {model_path or 'latest in ' + str(MODEL_DIR)}")
        service.load_model(model_path)
        logger.info("✅ Model loaded successfully")
        logger.info("")

        # Make prediction
        logger.info(
            f"🔮 Predicting for {args.province} | {args.season.replace('_', ' ').title()} {args.year or 'current year'}..."
        )
        result = service.predict(
            province=args.province,
            season=args.season,
            year=args.year,
        )

        # Display results
        logger.info("")
        logger.info("=" * 70)
        logger.info("🎯 RICE YIELD PREDICTION RESULTS")
        logger.info("=" * 70)
        logger.info(f"📍 Location:    {result['province']}")
        logger.info(
            f"📅 Season:      {result['season'].replace('_', ' ').title()} {result['year']}"
        )
        logger.info(f"🗓️  Predicted:   {result['prediction_date']}")
        logger.info(f"🌾 Weather Days: {result.get('weather_days', 'N/A')}")
        logger.info("")
        logger.info(f"🎯 PREDICTION:  {result['prediction']} Yield")
        logger.info(f"📊 Confidence:  {result['confidence']:.1%} ({result['confidence_level']})")

        # Warnings
        if result.get("warnings"):
            logger.info("")
            logger.warning("⚠️  WARNINGS:")
            for warning in result["warnings"]:
                logger.warning(f"   {warning}")

        # Probabilities
        logger.info("")
        logger.info("📈 Class Probabilities:")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob * 50)
            logger.info(f"   {cls:12s}: {prob:6.1%} {bar}")

        # Explanation
        logger.info("")
        logger.info("=" * 70)
        logger.info("💡 EXPLANATION")
        logger.info("=" * 70)
        logger.info(result["explanation"])

        # Triggered patterns
        if result.get("triggered_patterns"):
            logger.info("")
            logger.info(f"🔍 Triggered Weather Patterns ({len(result['triggered_patterns'])}):")
            for i, pattern in enumerate(result["triggered_patterns"][:5], 1):
                logger.info(f"   {i}. {pattern['pattern']}")
                logger.info(
                    f"      Type: {pattern['type']} | Growth Rate: {pattern['growth_rate']}× | Strength: {pattern['strength']}"
                )

            if len(result["triggered_patterns"]) > 5:
                logger.info(f"   ... and {len(result['triggered_patterns']) - 5} more patterns")

        # Top features (SHAP)
        if result.get("top_features"):
            logger.info("")
            logger.info("🎯 Top Contributing Features (SHAP Values):")
            for feat, value in list(result["top_features"].items())[:8]:
                direction = "↑" if value > 0 else "↓"
                impact = "High" if abs(value) > 0.1 else "Moderate" if abs(value) > 0.05 else "Low"
                logger.info(f"   {direction} {feat:50s}: {value:+.4f} ({impact})")

        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        sys.exit(1)


def command_info(args):
    """Handle the 'info' command - show model information."""
    logger.info("=" * 70)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 70)

    try:
        service = initialize_service(
            model_dir=args.model_dir if hasattr(args, "model_dir") else None
        )

        # Load model
        model_path = args.model_path if hasattr(args, "model_path") and args.model_path else None
        logger.info(f"📂 Loading model from: {model_path or 'latest in ' + str(MODEL_DIR)}")
        metadata = service.load_model(model_path)

        # Display info
        info = service.get_model_info()

        logger.info("")
        logger.info(f"✅ Status:         {info['status']}")
        logger.info(f"📅 Training Date:  {metadata.get('training_date', 'unknown')}")
        logger.info(f"📊 Training Data:  {metadata.get('n_seasons', 'unknown')} seasons")
        logger.info("")
        logger.info(f"🔧 Model Configuration:")
        logger.info(f"   Total Features:     {info['n_features']}")
        logger.info(f"     • Numerical:      {info['feature_breakdown']['numerical']}")
        logger.info(f"     • Patterns:       {info['feature_breakdown']['patterns']}")
        logger.info(f"   Contrast Patterns:  {info['n_patterns']}")
        logger.info(f"   Classes:            {', '.join(info['classes'])}")
        logger.info("")
        logger.info(f"📈 Performance Metrics:")
        metrics = metadata.get("metrics", {})
        logger.info(f"   Accuracy:  {metrics.get('avg_accuracy', metrics.get('accuracy', 0)):.4f}")
        logger.info(
            f"   Precision: {metrics.get('avg_precision', metrics.get('precision', 0)):.4f}"
        )
        logger.info(f"   Recall:    {metrics.get('avg_recall', metrics.get('recall', 0)):.4f}")
        logger.info(f"   F1 Score:  {metrics.get('avg_f1_macro', metrics.get('f1', 0)):.4f}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Failed to load model info: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rice Yield Prediction System (2025 - Contrast Pattern Mining)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Prepare training data (run once or when data updates)
  python -m src.presentation.cli.main prepare-data
  
  # Step 2: Train model with contrast patterns
  python -m src.presentation.cli.main train
  
  # Step 3: Make predictions
  python -m src.presentation.cli.main predict --province "An Giang" --season "winter_spring" --year 2020
  
  # Check model information
  python -m src.presentation.cli.main info
  
  # Use specific model file
  python -m src.presentation.cli.main predict --province "An Giang" --season "winter_spring" --year 2020 --model-path "models/model_20241124.pkl"

Available Seasons:
  - winter_spring  (Đông Xuân)
  - summer_autumn  (Hè Thu)
  - main_season    (Vụ Chính - for Northern Vietnam)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === prepare-data: Full preprocessing + export CSVs ===
    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Run full data pipeline: collect → detrend → align weather → discretize → export CSVs",
    )
    prepare_parser.set_defaults(func=command_prepare_data)

    # === train: Load CSVs → mine contrast patterns → train model ===
    train_parser = subparsers.add_parser(
        "train", help="Train model using preprocessed data (fast, repeatable)"
    )
    train_parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR), help="Directory to save trained model"
    )
    train_parser.set_defaults(func=command_train)

    # === predict: Fast inference ===
    predict_parser = subparsers.add_parser("predict", help="Predict yield for a province/season")
    predict_parser.add_argument(
        "--province", type=str, required=True, help="Province name (e.g. 'An Giang', 'Đồng Tháp')"
    )
    predict_parser.add_argument(
        "--season",
        type=str,
        required=True,
        choices=["winter_spring", "summer_autumn", "main_season"],
        help="Season name",
    )
    predict_parser.add_argument(
        "--year", type=int, default=None, help="Year to predict (default: current year)"
    )
    predict_parser.add_argument(
        "--model-path",
        type=str,
        help="Path to specific model file (optional, uses latest if not specified)",
    )
    predict_parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory containing models (default: models/)",
    )
    predict_parser.set_defaults(func=command_predict)

    # === info: Show model information ===
    info_parser = subparsers.add_parser("info", help="Show detailed model information and metrics")
    info_parser.add_argument(
        "--model-path", type=str, help="Path to specific model file (optional)"
    )
    info_parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory containing models (default: models/)",
    )
    info_parser.set_defaults(func=command_info)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
