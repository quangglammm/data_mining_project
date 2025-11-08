"""Example usage of the rice yield prediction system."""

import logging
from src.application.services.rice_yield_predictor_service import RiceYieldPredictorService
from src.infrastructure.repositories.gso_rice_yield_repository import GSORiceYieldRepository
from src.infrastructure.repositories.nasa_weather_repository import NASAWeatherRepository
from src.infrastructure.repositories.file_model_repository import FileModelRepository
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
    """Example usage."""
    # Initialize repositories
    rice_yield_repo = GSORiceYieldRepository(str(RICE_DATA_FILE))
    weather_repo = NASAWeatherRepository(str(WEATHER_DATA_FILE), use_api=False)
    model_repo = FileModelRepository(str(MODEL_DIR))

    # Initialize service
    service = RiceYieldPredictorService(
        rice_yield_repo=rice_yield_repo,
        weather_repo=weather_repo,
        model_repo=model_repo,
        season_definitions=SEASON_DEFINITIONS,
        growth_stage_definitions=GROWTH_STAGE_DEFINITIONS,
    )

    # Example 1: Train the model
    print("=" * 60)
    print("Example 1: Training the model")
    print("=" * 60)
    try:
        df_agg, df_sequences = service.prepare_training_data()
        model, metrics = service.train_model(df_agg, df_sequences)

        print(f"\nTraining completed!")
        print(f"Average F1-Score (macro): {metrics['avg_f1_macro']:.4f}")
        print(f"Average Accuracy: {metrics['avg_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return

    # Example 2: Make a prediction
    print("\n" + "=" * 60)
    print("Example 2: Making a prediction")
    print("=" * 60)
    try:
        result = service.predict(
            province="An Giang",
            season="winter_spring",
            year=2020,
        )

        print(f"\nPrediction Result:")
        print(f"  Province: {result['province']}")
        print(f"  Season: {result['season']}")
        print(f"  Year: {result['year']}")
        print(f"  Predicted Yield Class: {result['prediction']}")

        if result.get("top_features"):
            print(f"\n  Top Features:")
            for feature, importance in list(result["top_features"].items())[:5]:
                print(f"    {feature}: {importance:.4f}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

