from __future__ import annotations

import os
from pathlib import Path
from typing import List


# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Data files
RACE_DRIVER_DATA_PATH: Path = PROCESSED_DATA_DIR / "race_driver.parquet"
FEATURES_DATA_PATH: Path = PROCESSED_DATA_DIR / "features.parquet"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"

# FastF1 cache
FASTF1_CACHE_DIR: Path = PROJECT_ROOT / "fastf1_cache"

# Seasons and splits
DEFAULT_START_SEASON: int = 2018
DEFAULT_END_SEASON: int = 2024

TRAIN_YEARS: List[int] = list(range(2018, 2023))  # 2018–2022
VAL_YEARS: List[int] = [2023]
TEST_YEARS: List[int] = [2024]

# Reproducibility
RANDOM_STATE: int = 42


def ensure_directories() -> None:
    """Create all required directories if they do not exist."""
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        FASTF1_CACHE_DIR,
    ):
        os.makedirs(path, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "MODELS_DIR",
    "RACE_DRIVER_DATA_PATH",
    "FEATURES_DATA_PATH",
    "METRICS_PATH",
    "FASTF1_CACHE_DIR",
    "DEFAULT_START_SEASON",
    "DEFAULT_END_SEASON",
    "TRAIN_YEARS",
    "VAL_YEARS",
    "TEST_YEARS",
    "RANDOM_STATE",
    "ensure_directories",
]

