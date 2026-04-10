from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURES_DATA_PATH,
    RACE_DRIVER_DATA_PATH,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
    ensure_directories,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "season", "round"]).copy()

    def _driver_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["season", "round"])

        finish_shifted = g["finish_position"].shift(1)
        dnf_shifted = g["dnf"].astype(int).shift(1)

        g["driver_avg_finish_last_3"] = finish_shifted.rolling(3, min_periods=1).mean()
        g["driver_avg_finish_last_5"] = finish_shifted.rolling(5, min_periods=1).mean()
        g["driver_std_finish_last_5"] = finish_shifted.rolling(5, min_periods=1).std()
        g["driver_dnf_rate_last_10"] = dnf_shifted.rolling(10, min_periods=1).mean()

        return g

    return df.groupby("driver_id", group_keys=False).apply(_driver_group, include_groups=False)


def _add_constructor_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["constructor_id", "season", "round"]).copy()

    def _constructor_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["season", "round"])

        finish_shifted = g["finish_position"].shift(1)
        dnf_shifted = g["dnf"].astype(int).shift(1)

        g["constructor_avg_finish_last_5"] = (
            finish_shifted.rolling(5, min_periods=1).mean()
        )
        g["constructor_dnf_rate_last_10"] = (
            dnf_shifted.rolling(10, min_periods=1).mean()
        )
        return g

    return df.groupby("constructor_id", group_keys=False).apply(_constructor_group, include_groups=False)


def _add_track_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round"]).copy()

    seasons = sorted(df["season"].unique())

    df["driver_avg_finish_at_circuit"] = np.nan
    df["driver_avg_quali_at_circuit"] = np.nan

    for season in seasons:
        prior_seasons = [s for s in seasons if s < season]
        if not prior_seasons:
            continue

        prior_mask = df["season"].isin(prior_seasons)
        prior = df.loc[prior_mask]

        hist = (
            prior.groupby(["driver_id", "circuit_id"])
            .agg(
                driver_avg_finish_at_circuit=("finish_position", "mean"),
                driver_avg_quali_at_circuit=("quali_position", "mean"),
            )
            .reset_index()
        )

        season_mask = df["season"] == season
        season_df = df.loc[season_mask]
        merged = season_df.merge(
            hist,
            on=["driver_id", "circuit_id"],
            how="left",
        )

        df.loc[season_mask, "driver_avg_finish_at_circuit"] = merged[
            "driver_avg_finish_at_circuit"
        ].values
        df.loc[season_mask, "driver_avg_quali_at_circuit"] = merged[
            "driver_avg_quali_at_circuit"
        ].values

    return df


def build_features() -> pd.DataFrame:
    LOGGER.info("Loading base dataset from %s", RACE_DRIVER_DATA_PATH)
    df = pd.read_parquet(RACE_DRIVER_DATA_PATH)

    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)

    # Ensure identifiers exist
    required_cols = [
        "season",
        "round",
        "race_date",
        "driver_id",
        "constructor_id",
        "circuit_id",
        "finish_position",
        "grid_position",
        "quali_position",
        "quali_gap_to_pole",
        "quali_gap_to_teammate",
        "dnf",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in base dataset: {missing}")

    df = df.sort_values(["season", "round", "race_date", "driver_id"]).reset_index(
        drop=True
    )

    LOGGER.info("Adding driver form features")
    df = _add_driver_form_features(df)

    LOGGER.info("Adding constructor form features")
    df = _add_constructor_form_features(df)

    LOGGER.info("Adding track history features (prior seasons only)")
    df = _add_track_history_features(df)

    # Handle missing history with global averages
    global_avg_finish = df["finish_position"].mean()
    global_dnf_rate = df["dnf"].astype(int).mean()
    global_quali_position = df["quali_position"].mean()

    fill_map = {
        "driver_avg_finish_last_3": global_avg_finish,
        "driver_avg_finish_last_5": global_avg_finish,
        "driver_std_finish_last_5": 0.0,
        "driver_dnf_rate_last_10": global_dnf_rate,
        "constructor_avg_finish_last_5": global_avg_finish,
        "constructor_dnf_rate_last_10": global_dnf_rate,
        "driver_avg_finish_at_circuit": global_avg_finish,
        "driver_avg_quali_at_circuit": global_quali_position,
    }

    for col, value in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    LOGGER.info("Built features dataframe with shape %s", df.shape)
    return df


def main() -> None:
    ensure_directories()
    features = build_features()

    LOGGER.info("Saving features to %s", FEATURES_DATA_PATH)
    features.to_parquet(FEATURES_DATA_PATH, index=False)


if __name__ == "__main__":
    main()

