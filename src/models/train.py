from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    FEATURES_DATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    ensure_directories,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


FEATURE_COLUMNS: List[str] = [
    # Qualifying / grid
    "grid_position",
    "quali_position",
    "quali_gap_to_pole",
    "quali_gap_to_teammate",
    # Driver form
    "driver_avg_finish_last_3",
    "driver_avg_finish_last_5",
    "driver_std_finish_last_5",
    "driver_dnf_rate_last_10",
    # Constructor form
    "constructor_avg_finish_last_5",
    "constructor_dnf_rate_last_10",
    # Track history
    "driver_avg_finish_at_circuit",
    "driver_avg_quali_at_circuit",
]


def _time_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_YEARS)].copy()
    val = df[df["season"].isin(VAL_YEARS)].copy()
    test = df[df["season"].isin(TEST_YEARS)].copy()
    return {"train": train, "val": val, "test": test}


def _round_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    """Split by (season, round) when year-based splits leave train/val empty."""
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    keys = df[["season", "round"]].drop_duplicates().reset_index(drop=True)
    n = len(keys)
    if n == 0:
        return {"train": df.iloc[0:0], "val": df.iloc[0:0], "test": df}
    n_train = max(1, int(n * train_frac))
    n_val = max(0, int(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    train_keys = keys.iloc[:n_train]
    val_keys = keys.iloc[n_train : n_train + n_val]
    test_keys = keys.iloc[n_train + n_val :]
    train = df.merge(train_keys, on=["season", "round"], how="inner")
    val = df.merge(val_keys, on=["season", "round"], how="inner")
    test = df.merge(test_keys, on=["season", "round"], how="inner")
    return {"train": train, "val": val, "test": test}


def _baseline_metrics(splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            metrics[split_name] = {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
            continue
        y_true = split_df["finish_position"].values
        y_pred = split_df["grid_position"].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rho, _ = spearmanr(y_true, y_pred)
        metrics[split_name] = {"mae": float(mae), "rmse": float(rmse), "spearman": float(rho)}
    return metrics


def _evaluate_model(
    model: Pipeline, splits: Dict[str, pd.DataFrame]
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            metrics[split_name] = {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
            continue
        X = split_df[FEATURE_COLUMNS].values
        y_true = split_df["finish_position"].values
        y_pred = model.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rho, _ = spearmanr(y_true, y_pred)
        metrics[split_name] = {"mae": float(mae), "rmse": float(rmse), "spearman": float(rho)}
    return metrics


def train_models() -> Dict[str, Dict[str, Dict[str, float]]]:
    LOGGER.info("Loading features from %s", FEATURES_DATA_PATH)
    df = pd.read_parquet(FEATURES_DATA_PATH)

    missing = [c for c in FEATURE_COLUMNS + ["finish_position", "season"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in features dataframe: {missing}")

    splits = _time_split(df)
    if len(splits["train"]) == 0:
        LOGGER.warning(
            "No data in configured train years %s; using round-based split on available data.",
            TRAIN_YEARS,
        )
        splits = _round_based_split(df)
    for name, split_df in splits.items():
        LOGGER.info("Split %s has %d rows", name, len(split_df))

    if len(splits["train"]) == 0:
        raise ValueError(
            "No training data available. Run make_dataset with --start-year/--end-year "
            "covering the configured TRAIN_YEARS (e.g. 2018–2022), or ensure features.parquet "
            "contains those seasons."
        )

    # Baseline
    baseline = _baseline_metrics(splits)

    # Prepare train data for models
    X_train = splits["train"][FEATURE_COLUMNS].values
    y_train = splits["train"]["finish_position"].values

    # Model A: Ridge regression with imputation and standardization
    ridge_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )
    LOGGER.info("Training Ridge regression model")
    ridge_pipeline.fit(X_train, y_train)
    ridge_metrics = _evaluate_model(ridge_pipeline, splits)

    # Model B: RandomForestRegressor with imputation
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    LOGGER.info("Training RandomForestRegressor model")
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = _evaluate_model(rf_pipeline, splits)

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "baseline_grid": baseline,
        "ridge": ridge_metrics,
        "random_forest": rf_metrics,
    }

    # Select best model by validation MAE (fall back to test MAE if val is empty)
    val_mae_ridge = ridge_metrics["val"]["mae"]
    val_mae_rf = rf_metrics["val"]["mae"]
    if not (np.isnan(val_mae_ridge) or np.isnan(val_mae_rf)):
        best_name = "ridge" if val_mae_ridge <= val_mae_rf else "random_forest"
    else:
        test_mae_ridge = ridge_metrics["test"]["mae"]
        test_mae_rf = rf_metrics["test"]["mae"]
        best_name = "ridge" if test_mae_ridge <= test_mae_rf else "random_forest"
    best_model = ridge_pipeline if best_name == "ridge" else rf_pipeline

    LOGGER.info("Best model by validation MAE: %s", best_name)

    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    LOGGER.info("Saving best model pipeline to %s", model_path)
    dump(best_model, model_path)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    LOGGER.info("Saving model metadata to %s", metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    LOGGER.info("Saving metrics to %s", METRICS_PATH)
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))

    return all_metrics


def main() -> None:
    ensure_directories()
    metrics = train_models()

    LOGGER.info("Training complete. Summary (validation/test MAE):")
    for model_name, model_metrics in metrics.items():
        val_mae = model_metrics.get("val", {}).get("mae")
        test_mae = model_metrics.get("test", {}).get("mae")
        LOGGER.info("  %s: val MAE=%.3f, test MAE=%.3f", model_name, val_mae, test_mae)


if __name__ == "__main__":
    main()

