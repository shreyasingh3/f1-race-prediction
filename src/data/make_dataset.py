from __future__ import annotations

import argparse
import logging
from typing import List

import fastf1
import numpy as np
import pandas as pd

from src.utils.config import (
    DEFAULT_END_SEASON,
    DEFAULT_START_SEASON,
    FASTF1_CACHE_DIR,
    RACE_DRIVER_DATA_PATH,
    ensure_directories,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _enable_fastf1_cache() -> None:
    """Enable FastF1 cache in the configured directory."""
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    LOGGER.info("FastF1 cache enabled at %s", FASTF1_CACHE_DIR)


def _get_event_metadata(
    session: fastf1.core.Session,
    season: int,
    round_number: int,
) -> dict:
    """Extract basic metadata for a race event from a FastF1 session."""
    event = session.event
    # Event can be dict-like (event['EventDate']) or attribute-based (getattr).
    try:
        date = event["EventDate"] if hasattr(event, "__getitem__") else getattr(event, "EventDate", None)
    except (KeyError, TypeError):
        date = getattr(event, "EventDate", None)
    name = getattr(event, "EventName", None) or getattr(event, "OfficialEventName", None)
    circuit = getattr(event, "Location", None) or getattr(event, "EventName", None)
    return {
        "season": season,
        "round": round_number,
        "race_name": name,
        "circuit_name": circuit,
        "race_date": pd.to_datetime(date),
    }


def _build_race_results(session: fastf1.core.Session) -> pd.DataFrame:
    """Return a race results DataFrame at driver level."""
    results = session.results
    if results is None or results.empty:
        raise RuntimeError("Race session results are empty.")

    df = results.copy()

    # FastF1 3.8 uses FullName (not Driver); map to our canonical names.
    rename_map = {
        "DriverNumber": "driver_number",
        "Abbreviation": "driver_id",
        "FullName": "driver_name",
        "TeamName": "constructor_id",
        "Position": "finish_position",
        "GridPosition": "grid_position",
        "Status": "status",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Fallback if older API used "Driver" for name
    if "driver_name" not in df.columns and "Driver" in df.columns:
        df = df.rename(columns={"Driver": "driver_name"})

    needed_cols = [
        "driver_number",
        "driver_id",
        "driver_name",
        "constructor_id",
        "finish_position",
        "grid_position",
        "status",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in race results: {missing}")

    # DNF flag – anything that is not a standard finished status.
    finished_statuses = {"Finished", "Finished "}
    df["dnf"] = ~df["status"].isin(finished_statuses)

    return df[needed_cols + ["dnf"]]


def _build_quali_features(quali_session: fastf1.core.Session) -> pd.DataFrame:
    """Compute qualifying-based features per driver."""
    laps = quali_session.laps
    if laps is None or laps.empty:
        raise RuntimeError("Qualifying session laps are empty.")

    df = laps[["DriverNumber", "Driver", "Team", "LapTime"]].copy()
    df = df.rename(
        columns={
            "DriverNumber": "driver_number",
            "Driver": "driver_name",
            "Team": "constructor_id",
        }
    )

    # LapTime is a pandas Timedelta; convert to seconds for easy math.
    df["lap_time_seconds"] = df["LapTime"].dt.total_seconds()
    df = df.dropna(subset=["lap_time_seconds"])

    best_laps = (
        df.sort_values("lap_time_seconds")
        .groupby("driver_number", as_index=False)
        .first()
    )

    # Rank drivers by best lap time (1 = pole)
    best_laps["quali_position"] = best_laps["lap_time_seconds"].rank(
        method="min"
    ).astype(int)

    pole_time = best_laps["lap_time_seconds"].min()
    best_laps["quali_gap_to_pole"] = best_laps["lap_time_seconds"] - pole_time

    # Gap to teammate: difference between driver and teammate best laps within a constructor.
    def _gap_to_teammate(group: pd.DataFrame) -> pd.Series:
        if len(group) < 2:
            return pd.Series(np.nan, index=group.index)
        # For each driver: lap_time_seconds - teammate_best
        best_times = group["lap_time_seconds"]
        teammate_best = best_times.mean()
        return best_times - teammate_best

    best_laps["quali_gap_to_teammate"] = (
        best_laps.groupby("constructor_id", group_keys=False)
        .apply(_gap_to_teammate, include_groups=False)
        .reset_index(drop=True)
    )

    return best_laps[
        [
            "driver_number",
            "constructor_id",
            "quali_position",
            "quali_gap_to_pole",
            "quali_gap_to_teammate",
        ]
    ]


def _build_dataset_for_season(year: int) -> List[pd.DataFrame]:
    """Build race-driver rows for all race rounds in a given season."""
    season_dfs: List[pd.DataFrame] = []

    # TODO: If FastF1's event listing API changes, consider using fastf1.get_event_schedule
    # or other helpers instead of assuming that round numbers map 1:1 to races.
    schedule = fastf1.get_event_schedule(year)
    race_events = schedule[schedule["EventFormat"].isin(["conventional", "sprint"])]

    for _, event in race_events.iterrows():
        round_number = int(event["RoundNumber"])
        try:
            race_session = fastf1.get_session(year, round_number, "R")
            quali_session = fastf1.get_session(year, round_number, "Q")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Skipping year=%s round=%s due to session creation error: %s",
                year,
                round_number,
                exc,
            )
            continue

        LOGGER.info("Loading sessions for %s round %s", year, round_number)
        try:
            race_session.load()
            quali_session.load()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Skipping year=%s round=%s due to load error: %s",
                year,
                round_number,
                exc,
            )
            continue

        meta = _get_event_metadata(race_session, season=year, round_number=round_number)
        race_results = _build_race_results(race_session)
        quali_features = _build_quali_features(quali_session)

        df = race_results.merge(
            quali_features,
            on=["driver_number", "constructor_id"],
            how="left",
        )

        df["season"] = meta["season"]
        df["round"] = meta["round"]
        df["race_name"] = meta["race_name"]
        df["circuit_name"] = meta["circuit_name"]
        df["race_date"] = meta["race_date"]

        # Stable identifiers for downstream feature engineering
        df["circuit_id"] = df["circuit_name"]

        season_dfs.append(df)

    return season_dfs


def build_race_driver_dataset(start_year: int, end_year: int) -> pd.DataFrame:
    """Build the race-driver-level dataset for a range of seasons."""
    _enable_fastf1_cache()

    all_dfs: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        LOGGER.info("Processing season %s", year)
        season_dfs = _build_dataset_for_season(year)
        if not season_dfs:
            LOGGER.warning("No races processed for season %s", year)
            continue
        all_dfs.extend(season_dfs)

    if not all_dfs:
        raise RuntimeError("No race data collected; check FastF1 configuration.")

    dataset = pd.concat(all_dfs, ignore_index=True)

    # Ensure consistent dtypes
    dataset["season"] = dataset["season"].astype(int)
    dataset["round"] = dataset["round"].astype(int)
    dataset["finish_position"] = dataset["finish_position"].astype(int)
    dataset["grid_position"] = dataset["grid_position"].astype(int)
    dataset["dnf"] = dataset["dnf"].astype(bool)

    # Quali-based features may have missing values for drivers without a time; leave as NaN.
    LOGGER.info(
        "Built dataset with %d rows across seasons %s–%s",
        len(dataset),
        dataset["season"].min(),
        dataset["season"].max(),
    )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build race-driver-level dataset from FastF1."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_SEASON,
        help="Start season (inclusive). Default: %(default)s",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_SEASON,
        help="End season (inclusive). Default: %(default)s",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(RACE_DRIVER_DATA_PATH),
        help="Output parquet path. Default: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    dataset = build_race_driver_dataset(args.start_year, args.end_year)

    output_path = args.output_path
    LOGGER.info("Saving dataset to %s", output_path)
    dataset.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()

