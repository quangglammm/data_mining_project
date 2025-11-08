# Quick Start Guide

## Installation

1. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify data files exist**:
   - `data/DBSCL_agriculture_1995_2024.csv`
   - `data/DBSCL_weather_1995_2024_FULL.xlsx`

## Quick Examples

### Train Model (CLI)
```bash
python main.py train
```

### Make Prediction (CLI)
```bash
python main.py predict --province "An Giang" --season "winter_spring" --year 2020
```

### Start API Server
```bash
uvicorn src.presentation.api.main:app --reload
```

Then visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Make Prediction via API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "province": "An Giang",
    "season": "winter_spring",
    "year": 2020
  }'
```

### Run Example Script
```bash
python example_usage.py
```

## Project Structure Overview

```
src/
├── domain/          # Business logic (entities, use cases, interfaces)
├── application/     # Application services (orchestration)
├── infrastructure/  # External adapters (repositories, APIs)
└── presentation/    # User interfaces (CLI, FastAPI)
```

## Key Components

- **Domain Layer**: Core business logic, independent of external systems
- **Application Layer**: Orchestrates use cases
- **Infrastructure Layer**: Implements repository interfaces, handles external APIs
- **Presentation Layer**: Provides CLI and REST API interfaces

## Next Steps

1. Review `README.md` for detailed documentation
2. Check `config/settings.py` for configuration options
3. Run tests: `pytest tests/ -v`
4. Explore the codebase starting from `src/domain/entities/`

