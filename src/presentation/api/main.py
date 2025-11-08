"""FastAPI main application."""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    API_SETTINGS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_SETTINGS["title"],
    description=API_SETTINGS["description"],
    version=API_SETTINGS["version"],
)

# Initialize repositories and service
rice_yield_repo = GSORiceYieldRepository(str(RICE_DATA_FILE))
weather_repo = NASAWeatherRepository(str(WEATHER_DATA_FILE), use_api=False)
model_repo = FileModelRepository(str(MODEL_DIR))

service = RiceYieldPredictorService(
    rice_yield_repo=rice_yield_repo,
    weather_repo=weather_repo,
    model_repo=model_repo,
    season_definitions=SEASON_DEFINITIONS,
    growth_stage_definitions=GROWTH_STAGE_DEFINITIONS,
)


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction."""

    province: str = Field(..., description="Province name (e.g., 'An Giang')")
    season: str = Field(
        ..., description="Season name (e.g., 'winter_spring', 'summer_autumn', 'main_season')"
    )
    year: Optional[int] = Field(None, description="Year (defaults to current year)")


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    province: str
    season: str
    year: int
    prediction: str
    prediction_vietnamese: str
    explanation: Optional[str] = None
    top_features: Optional[Dict[str, float]] = None


class TrainingResponse(BaseModel):
    """Response model for training."""

    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Rice Yield Prediction API",
        "version": API_SETTINGS["version"],
        "endpoints": {
            "predict": "/predict",
            "train": "/train",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict rice yield class for a given province and season.

    Args:
        request: Prediction request with province, season, and optional year

    Returns:
        Prediction response with yield class and explanation
    """
    try:
        result = service.predict(
            province=request.province,
            season=request.season,
            year=request.year,
        )

        # Convert prediction to Vietnamese
        from ...domain.entities.yield_class import YieldClass

        try:
            # Handle both single prediction and list
            pred_value = result["prediction"]
            if isinstance(pred_value, list):
                pred_value = pred_value[0]
            yield_class = YieldClass(pred_value)
            prediction_vn = yield_class.to_vietnamese()
        except (ValueError, KeyError, TypeError):
            prediction_vn = str(result.get("prediction", "Unknown"))

        return PredictionResponse(
            province=result["province"],
            season=result["season"],
            year=result["year"],
            prediction=result["prediction"],
            prediction_vietnamese=prediction_vn,
            explanation=result.get("explanation"),
            top_features=result.get("top_features"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainingResponse)
async def train() -> TrainingResponse:
    """
    Train the ML model.

    Returns:
        Training response with status and metrics
    """
    try:
        logger.info("Starting model training via API")
        df_agg, df_sequences = service.prepare_training_data()
        model, metrics = service.train_model(df_agg, df_sequences)

        return TrainingResponse(
            status="success",
            message="Model trained successfully",
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

