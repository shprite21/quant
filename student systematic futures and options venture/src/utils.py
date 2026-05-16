"""Shared utility helpers for the trading research repository."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
import pandas as pd
import yaml

PathLike = str | Path


def ensure_directory(path: PathLike) -> Path:
    """Create a directory if it does not exist and return its path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml_config(path: PathLike) -> dict:
    """Load a YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config or {}


def save_dataframe(data: pd.DataFrame, path: PathLike) -> None:
    """Persist a data frame to CSV, creating parent directories as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible research."""

    random.seed(seed)
    np.random.seed(seed)


def safe_divide(
    numerator: float | pd.Series | np.ndarray,
    denominator: float | pd.Series | np.ndarray,
) -> float | pd.Series | np.ndarray:
    """Safely divide while avoiding divide-by-zero warnings."""

    if isinstance(denominator, (pd.Series, np.ndarray)):
        denominator = np.where(np.asarray(denominator) == 0, np.nan, denominator)
    elif denominator == 0:
        denominator = np.nan
    return numerator / denominator


def infer_numeric_feature_columns(
    data: pd.DataFrame,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Infer model-ready numeric feature columns from a data frame."""

    exclusions = set(exclude or [])
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if column not in exclusions]


def walk_forward_windows(
    n_samples: int,
    train_window: int,
    test_window: int,
    step_size: int | None = None,
    expanding: bool = False,
) -> Generator[tuple[slice, slice], None, None]:
    """Yield rolling train/test slices for walk-forward validation."""

    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive integers.")
    step = step_size or test_window
    train_start = 0
    train_end = train_window
    while train_end + test_window <= n_samples:
        test_end = train_end + test_window
        yield slice(train_start, train_end), slice(train_end, test_end)
        if expanding:
            train_end += step
        else:
            train_start += step
            train_end += step


def annualization_factor_from_frequency(frequency: str) -> int:
    """Map a data frequency string to an annualization factor."""

    lookup = {
        "1d": 252,
        "1wk": 52,
        "1mo": 12,
    }
    return lookup.get(frequency.lower(), 252)
