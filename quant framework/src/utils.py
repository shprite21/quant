from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> Path:
    """Build an absolute path relative to the repository root."""

    return PROJECT_ROOT.joinpath(*parts)


def ensure_directories(paths: Iterable[str | Path]) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    """Synchronize the standard-library and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)


def annualization_factor(freq: str = "daily") -> int:
    """Return the standard annualization factor for common data frequencies."""

    mapping = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "hourly": 252 * 24,
    }

    if freq not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported frequency '{freq}'. Use one of: {supported}.")

    return mapping[freq]


def safe_divide(numerator, denominator, fill_value: float = 0.0) -> np.ndarray:
    """Safely divide arrays while avoiding divisions by zero."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = np.abs(denominator) > 1e-12
    np.divide(numerator, denominator, out=result, where=mask)
    return result

