from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from joblib import load

from src.utils.config import FEATURES_DATA_PATH, MODELS_DIR, REPORTS_DIR, ensure_directories


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict driver finishing positions for a given F1 race."
    )
    parser.add_argument("--season", type=int, required=True, help="Season (e.g. 2024)")
    parser.add_argument("--round", type=int, required=True, help="Round number (e.g. 10)")
    parser.add_argument(
        "--features-path",
        type=str,
        default=str(FEATURES_DATA_PATH),
        help="Path to features parquet file. Default: %(default)s",
    )
    return parser.parse_args()


def load_model_and_metadata():
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


def predict_for_race(season: int, round_number: int, features_path: str) -> pd.DataFrame:
    LOGGER.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)

    mask = (df["season"] == season) & (df["round"] == round_number)
    race_df = df.loc[mask].copy()
    if race_df.empty:
        raise ValueError(
            f"No feature rows found for season={season}, round={round_number}. "
            "Ensure you have run make_dataset and build_features including this race."
        )

    model, feature_columns = load_model_and_metadata()

    missing = [c for c in feature_columns if c not in race_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for prediction: {missing}")

    X = race_df[feature_columns].values
    preds = model.predict(X)

    race_df["predicted_finish_position"] = preds

    result_cols = [
        "season",
        "round",
        "race_name",
        "circuit_name",
        "driver_id",
        "constructor_id",
        "grid_position",
        "predicted_finish_position",
    ]
    existing_cols = [c for c in result_cols if c in race_df.columns]
    result = race_df[existing_cols].copy()
    result = result.sort_values("predicted_finish_position").reset_index(drop=True)

    return result


def main() -> None:
    ensure_directories()
    args = parse_args()

    predictions = predict_for_race(args.season, args.round, args.features_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        REPORTS_DIR
        / f"predictions_season{args.season}_round{args.round}.csv"
    )
    LOGGER.info("Saving predictions to %s", out_path)
    predictions.to_csv(out_path, index=False)

    # Print nicely to stdout
    pd.set_option("display.max_rows", None)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()

