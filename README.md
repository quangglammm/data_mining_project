# Rice Yield Prediction System - Mekong Delta (ĐBSCL)

A complete machine learning system for predicting rice yield classification in the Mekong Delta region of Vietnam, built with clean architecture principles.

## Overview

This project implements a comprehensive rice yield prediction system that:

- Collects rice yield and weather data for 13 provinces in the Mekong Delta
- Processes and labels yield data using detrending techniques
- Discretizes weather data into event sequences based on growth stages
- Mines sequential patterns from weather events
- Trains ML models (XGBoost/Random Forest) with time series cross-validation
- Provides predictions with SHAP-based explanations
- Supports both CLI and REST API interfaces

## Architecture

The project follows **Clean Architecture** principles with clear separation of concerns:

```
src/
├── domain/              # Core business logic (independent of external concerns)
│   ├── entities/        # Domain entities (Province, Season, RiceYieldData, etc.)
│   ├── repositories/    # Repository interfaces (abstractions)
│   └── use_cases/       # Business operations (8 use cases)
├── application/         # Application services (orchestrates use cases)
│   └── services/        # RiceYieldPredictorService
├── infrastructure/      # External adapters and implementations
│   ├── repositories/    # Concrete repository implementations
│   ├── adapters/        # ML adapters (SHAP, etc.)
│   └── external/        # API clients
└── presentation/        # User interfaces
    ├── api/             # FastAPI REST API
    └── cli/             # Command-line interface
```

## Features

### Data Processing
- **Rice Yield Data**: Loads from GSO (General Statistics Office) CSV files
- **Weather Data**: Loads from NASA POWER API or local Excel files
- **Detrending**: Removes temporal trends using linear regression
- **Labeling**: Classifies yield into High/Medium/Low using quantile-based labeling

### Weather Processing
- **Season Alignment**: Maps daily weather to growing seasons
- **Growth Stage Analysis**: Divides seasons into 5 growth stages
- **Discretization**: Converts continuous weather into discrete events (e.g., "Nóng-Khô", "Mát-Ướt")

### Pattern Mining
- **Sequential Pattern Mining**: Uses PrefixSpan algorithm to find frequent patterns
- **Pattern Features**: Creates binary features indicating pattern presence

### Machine Learning
- **Time Series Cross-Validation**: Uses TimeSeriesSplit to respect temporal order
- **Model Training**: Supports XGBoost and Random Forest classifiers
- **Evaluation**: Provides F1-macro, accuracy, and confusion matrix metrics

### Explainability
- **SHAP Integration**: Generates feature importance explanations
- **Top Features**: Identifies most influential features for predictions

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or poetry

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Ensure data files are present**:
   - `data/DBSCL_agriculture_1995_2024.csv` - Rice yield data
   - `data/DBSCL_weather_1995_2024_FULL.xlsx` - Weather data

## Usage

### Command-Line Interface (CLI)

#### Train the Model

```bash
python main.py train
```

This will:
1. Load and process rice yield data
2. Detrend and label the data
3. Align weather data with seasons
4. Discretize weather into events
5. Mine sequential patterns
6. Build feature matrix
7. Train model with time series cross-validation
8. Save the trained model to `models/` directory

#### Make Predictions

```bash
python main.py predict --province "An Giang" --season "winter_spring" --year 2024
```

Options:
- `--province`: Province name (e.g., "An Giang", "Can Tho")
- `--season`: Season name (`winter_spring`, `summer_autumn`, `main_season`)
- `--year`: Year (optional, defaults to current year)

### REST API

#### Start the API Server

```bash
uvicorn src.presentation.api.main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**1. Health Check**
```bash
GET /health
```

**2. Predict Yield**
```bash
POST /predict
Content-Type: application/json

{
  "province": "An Giang",
  "season": "winter_spring",
  "year": 2024
}
```

Response:
```json
{
  "province": "An Giang",
  "season": "winter_spring",
  "year": 2024,
  "prediction": "High",
  "prediction_vietnamese": "Năng suất Cao",
  "explanation": "...",
  "top_features": {
    "pat_stage_2_Nóng-Khô": 0.15,
    "stage_2_avg_temp": 0.12,
    ...
  }
}
```

**3. Train Model**
```bash
POST /train
```

**4. API Documentation**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Configuration is managed in `config/settings.py`. Key settings:

- **Data Paths**: Locations of rice yield and weather data files
- **Season Definitions**: Start/end dates for each growing season
- **Growth Stage Definitions**: Day ranges for each growth stage
- **Model Settings**: Cross-validation splits, model type, pattern mining parameters

### Environment Variables

- `LLM_ENABLED`: Enable LLM-based explanations (default: false)
- `LLM_API_URL`: URL for LLM API
- `LLM_API_KEY`: API key for LLM service

## Project Structure

```
.
├── config/                  # Configuration files
│   └── settings.py
├── data/                    # Data files (CSV, Excel)
├── models/                  # Trained models (generated)
├── src/
│   ├── domain/              # Domain layer
│   │   ├── entities/        # Domain entities
│   │   ├── repositories/    # Repository interfaces
│   │   └── use_cases/       # Use cases
│   ├── application/         # Application layer
│   │   └── services/        # Application services
│   ├── infrastructure/      # Infrastructure layer
│   │   └── repositories/    # Repository implementations
│   └── presentation/        # Presentation layer
│       ├── api/             # FastAPI
│       └── cli/             # CLI
├── tests/                   # Unit tests
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Methodology

The system implements the following methodology:

1. **Data Collection**: Fetch rice yield and weather data for 13 provinces, 3 seasons, 1995-2024
2. **Detrending and Labeling**: Remove temporal trends, label residuals into High/Medium/Low
3. **Weather Processing**: Align daily weather to seasons, compute stage-wise aggregates
4. **Discretization**: Convert continuous weather into discrete events
5. **Pattern Mining**: Mine sequential patterns using PrefixSpan
6. **Feature Engineering**: Combine numeric features with pattern features
7. **Model Training**: Train with time series cross-validation
8. **Evaluation**: Compute metrics and generate explanations

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Development

### Code Style

The project uses:
- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking

Format code:
```bash
black src/ tests/
```

Lint:
```bash
flake8 src/ tests/
```

Type check:
```bash
mypy src/
```

## Dependencies

Key dependencies:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **xgboost**: Gradient boosting classifier
- **shap**: Model explainability
- **prefixspan**: Sequential pattern mining
- **fastapi**: REST API framework
- **pydantic**: Data validation

See `requirements.txt` for complete list.

## Limitations and Future Work

### Current Limitations
- Prediction requires stored patterns from training (needs refinement)
- NASA POWER API integration is placeholder (currently uses local files)
- LLM explanation integration is optional and not fully implemented

### Future Enhancements
- Complete NASA POWER API integration for real-time weather data
- Store and reuse patterns for prediction
- Full LLM integration for natural language explanations
- Database persistence (SQLite/PostgreSQL)
- Web UI for interactive predictions
- Model versioning and A/B testing

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

- Data sources: GSO (General Statistics Office), NASA POWER API
- Methodology based on sequential pattern mining for agricultural yield prediction

