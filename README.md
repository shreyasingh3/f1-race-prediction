## Overview

This project predicts Formula 1 race finishing positions using historical race data from the FastF1 API. I built an end-to-end ML pipeline that handles data ingestion, feature engineering, time-aware training, and evaluation.

The focus is on modeling real-world constraints like data leakage and temporal splits, while exploring how driver form, constructor performance, and qualifying results impact race outcomes.

## F1 race prediction

This repository implements an end-to-end, reproducible machine learning pipeline that uses the FastF1 API to predict Formula 1 race finishing positions at the driver level. It is designed to be recruiter-grade: clear structure, scripts instead of notebooks only, and a focus on time-aware evaluation to avoid data leakage.

The target is each driver's final race finishing position (1–20), which is inherently **ordinal**: the order matters and the distance between ranks is not strictly linear, but we begin with regression-style baselines and tree models for simplicity.

### Problem statement

- **Task**: Predict each driver's race finishing position for a given Grand Prix.
- **Target**: `finish_position` in \[1, 20\], one row per (season, round, driver).
- **Why ordinal?**:
  - Positions have a strict order (P1 is better than P2, etc.).
  - The difference between P1 and P2 is not the same as between P10 and P11.
  - Rank-aware metrics (e.g. Spearman correlation) are therefore important.

### Data source

All data is built from the **FastF1** API, which wraps the official F1 timing data. The project uses the FastF1 cache to avoid repeatedly downloading raw timing data and keeps the repository lightweight by ignoring large raw/processed files by default.

### Feature groups

- **Driver form (rolling, prior races only)**:
  - `driver_avg_finish_last_3`, `driver_avg_finish_last_5`
  - `driver_std_finish_last_5`
  - `driver_dnf_rate_last_10`
- **Constructor form (rolling)**:
  - `constructor_avg_finish_last_5`
  - `constructor_dnf_rate_last_10`
- **Track history (prior years only)**:
  - `driver_avg_finish_at_circuit`
  - `driver_avg_quali_at_circuit`
- **Qualifying / grid features**:
  - `grid_position`
  - `quali_position`
  - `quali_gap_to_pole`
  - `quali_gap_to_teammate`

All rolling and history features are built in a **strictly time-aware** way: only races prior to the current (season, round) are used when computing features. This is critical to avoid peeking into the future.

### Evaluation setup

We use a **forward-chaining, time-aware split**:

- **Train**: seasons 2018–2022
- **Validation**: season 2023
- **Test**: season 2024

This mimics a real-world deployment scenario where we train on past years, tune on the most recent complete season, and then evaluate on held-out future data.

### Models

- **Baseline**:
  - Predict `finish_position = grid_position` (simple but surprisingly strong).
- **Model A (linear)**:
  - Ridge regression with standardized numeric features.
- **Model B (tree)**:
  - RandomForestRegressor (or similar gradient boosting model).

Each model is evaluated with:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Spearman rank correlation (to respect the ordinal nature of positions)

### Results (placeholder)

| Model                | Split | MAE | RMSE | Spearman |
|----------------------|-------|-----|------|----------|
| Baseline (grid)      | Test  | TBD | TBD  | TBD      |
| Ridge Regression     | Test  | TBD | TBD  | TBD      |
| Random Forest        | Test  | TBD | TBD  | TBD      |

## Example Predictions

| Driver        | Predicted Position | Actual Position |
|---------------|-------------------|-----------------|
| Verstappen    | 1                 | 1               |
| Leclerc       | 3                 | 2               |
| Norris        | 4                 | 5               |

### Repository structure

```text
fastf1-position-predictor/
  README.md
  requirements.txt
  .gitignore
  notebooks/
    01_eda.ipynb              # EDA using processed datasets (to be added)
  data/
    raw/                      # Raw FastF1 cache / exports (ignored by git)
    processed/                # Processed tabular data (ignored by git)
  src/
    data/
      make_dataset.py         # Build race-driver-level dataset from FastF1
    features/
      build_features.py       # Rolling form, track history, qualifying features
    models/
      train.py                # Train baseline + models, save best
      evaluate.py             # Evaluate models, error analysis & plots
      predict.py              # CLI prediction for specific race
    utils/
      config.py               # Shared paths, constants, and seasons
  reports/
    figures/                  # Saved plots from evaluation
  models/                     # Serialized model pipelines (joblib)
```

### How to run

From the project root `fastf1-position-predictor/`:

1. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Build the dataset from FastF1**

```bash
python -m src.data.make_dataset --start-year 2018 --end-year 2024
```

4. **Build features**

```bash
python -m src.features.build_features
```

5. **Train models**

```bash
python -m src.models.train
```

6. **Evaluate and generate plots**

```bash
python -m src.models.evaluate
```

7. **Predict for a specific race**

```bash
python -m src.models.predict --season 2024 --round 10
```

### Next steps

- Upgrade to **ordinal regression** or ranking models (e.g. pairwise or listwise ranking, ordinal logistic regression).
- Incorporate **weather, pit strategy, tyre usage**, and safety car information as additional features.
- Experiment with **per-track models**, **driver-specific calibration**, and **uncertainty estimates** for predictions.

