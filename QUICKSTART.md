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
### Train (saves model + metadata to disk)
```
python main.py train
```

### Predict (loads model from disk, then predicts)
```
python main.py predict --province "An Giang" --season "winter_spring" --year 2020
```

### Check model info
```
python main.py info
```

### Use specific model file
```
python main.py predict --province "An Giang" --season "winter_spring" --year 2020 --model-path "models/model_20241124.pkl"
```
