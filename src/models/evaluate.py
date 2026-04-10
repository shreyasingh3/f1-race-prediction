from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.config import (
    FEATURES_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
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


def _load_best_model_and_metadata():
    model_path = MODELS_DIR / "best_model.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    model = load(model_path)
    metadata = json.loads(metadata_path.read_text())
    feature_columns = metadata["feature_columns"]
    return model, feature_columns


def _time_split(df: pd.DataFrame):
    train = df[df["season"].isin(TRAIN_YEARS)].copy()
    val = df[df["season"].isin(VAL_YEARS)].copy()
    test = df[df["season"].isin(TEST_YEARS)].copy()
    return {"train": train, "val": val, "test": test}


def _round_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "spearman": float(rho)}


def _make_plots(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    feature_importances: dict | None,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Predicted vs actual scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_test, y_pred_test, alpha=0.5)
    max_pos = max(y_true_test.max(), y_pred_test.max())
    min_pos = min(y_true_test.min(), y_pred_test.min())
    plt.plot([min_pos, max_pos], [min_pos, max_pos], "r--", label="y=x")
    plt.xlabel("Actual finish position")
    plt.ylabel("Predicted finish position")
    plt.title("Predicted vs Actual Finish Position (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pred_vs_actual_test.png")
    plt.close()

    # Residual distribution
    residuals = y_true_test - y_pred_test
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Residual Distribution (Test)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_hist_test.png")
    plt.close()

    # Feature importance (if available)
    if feature_importances:
        names = list(feature_importances.keys())
        values = np.array(list(feature_importances.values()))
        order = np.argsort(values)[::-1]
        names = [names[i] for i in order]
        values = values[order]

        plt.figure(figsize=(8, 6))
        plt.barh(names, values)
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png")
        plt.close()

    # Confusion-style heatmap of binned positions
    bins = [1, 6, 11, 16, 21]
    labels = ["1-5", "6-10", "11-15", "16-20"]
    true_bins = pd.cut(y_true_test, bins=bins, labels=labels, right=False)
    pred_bins = pd.cut(y_pred_test, bins=bins, labels=labels, right=False)

    conf = pd.crosstab(true_bins, pred_bins, rownames=["Actual"], colnames=["Predicted"])
    conf = conf.reindex(index=labels, columns=labels, fill_value=0)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf.values, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted position bin")
    plt.ylabel("Actual position bin")
    plt.title("Binned Position Confusion Heatmap (Test)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, int(conf.values[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_heatmap.png")
    plt.close()


def evaluate() -> dict:
    LOGGER.info("Loading features from %s", FEATURES_DATA_PATH)
    df = pd.read_parquet(FEATURES_DATA_PATH)

    model, feature_columns = _load_best_model_and_metadata()
    splits = _time_split(df)
    if len(splits["train"]) == 0:
        LOGGER.warning(
            "No data in configured train years %s; using round-based split for evaluation.",
            TRAIN_YEARS,
        )
        splits = _round_based_split(df)

    evaluation = {}
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            evaluation[split_name] = {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
            continue
        X = split_df[feature_columns].values
        y_true = split_df["finish_position"].values
        y_pred = model.predict(X)
        evaluation[split_name] = _compute_metrics(y_true, y_pred)

    # Error analysis on test set
    test_df = splits["test"].copy()
    if len(test_df) == 0:
        LOGGER.warning("No test data; skipping error analysis and plots.")
        return evaluation
    X_test = test_df[feature_columns].values
    y_true_test = test_df["finish_position"].values
    y_pred_test = model.predict(X_test)

    test_df["abs_error"] = np.abs(y_true_test - y_pred_test)

    # MAE by constructor (skip if column missing, e.g. older features.parquet)
    if "constructor_id" in test_df.columns:
        mae_by_constructor = (
            test_df.groupby("constructor_id")["abs_error"]
            .mean()
            .sort_values()
            .reset_index()
        )
        mae_by_constructor.to_csv(REPORTS_DIR / "mae_by_constructor.csv", index=False)
    else:
        LOGGER.warning("No 'constructor_id' in test data; skipping mae_by_constructor.")

    # MAE by grid bucket (skip if grid_position missing)
    if "grid_position" in test_df.columns:
        bins = [1, 6, 11, 16, 21]
        labels = ["1-5", "6-10", "11-15", "16-20"]
        test_df["grid_bucket"] = pd.cut(
            test_df["grid_position"], bins=bins, labels=labels, right=False
        )
        mae_by_grid = (
            test_df.groupby("grid_bucket", observed=True)["abs_error"]
            .mean()
            .reindex(labels)
            .reset_index()
        )
        mae_by_grid.to_csv(REPORTS_DIR / "mae_by_grid_bucket.csv", index=False)
    else:
        LOGGER.warning("No 'grid_position' in test data; skipping mae_by_grid_bucket.")

    # Feature importances
    feature_importances = None
    estimator = getattr(model, "named_steps", {}).get("model")
    if estimator is not None:
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            feature_importances = dict(zip(feature_columns, importances))
        elif hasattr(estimator, "coef_"):
            coefs = np.abs(estimator.coef_)
            if coefs.ndim > 1:
                coefs = coefs.mean(axis=0)
            feature_importances = dict(zip(feature_columns, coefs))

    _make_plots(y_true_test, y_pred_test, feature_importances)

    # Save evaluation summary
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(evaluation, indent=2))

    return evaluation


def main() -> None:
    ensure_directories()
    evaluation = evaluate()
    LOGGER.info("Evaluation complete.")
    for split, metrics in evaluation.items():
        LOGGER.info(
            "%s -> MAE=%.3f, RMSE=%.3f, Spearman=%.3f",
            split,
            metrics["mae"],
            metrics["rmse"],
            metrics["spearman"],
        )


if __name__ == "__main__":
    main()

