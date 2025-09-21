"""Utility functions shared across the Fantasy Manager application."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

_FIRST_N_PREFIX = 3
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def american_to_probability(odds: Optional[float]) -> float:
    """Convert American odds to implied probability.

    Parameters
    ----------
    odds:
        American odds expressed as an integer (e.g. ``-110``).

    Returns
    -------
    float
        The implied probability between 0 and 1. ``np.nan`` is returned when the
        input is missing.
    """

    if odds is None:
        return math.nan
    try:
        odds = int(odds)
    except (TypeError, ValueError):
        return math.nan

    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return math.nan


def _strip_accents(name: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def _alnum(token: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]", "", token)


def _normalize_order(name: str) -> str:
    if isinstance(name, str) and "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        return f"{first} {last}".strip()
    return name


def make_merge_key(name: Optional[str], n_prefix: int = _FIRST_N_PREFIX) -> str:
    """Create a fuzzy key that aligns players across disparate data sources."""

    if not isinstance(name, str) or not name.strip():
        return ""

    normalized = _normalize_order(_strip_accents(name).lower().strip())
    tokens = [tok for tok in normalized.replace(".", " ").split() if tok]

    while tokens and _alnum(tokens[-1]) in _SUFFIXES:
        tokens.pop()

    if not tokens:
        return ""

    first, *last_tokens = tokens
    first_clean = _alnum(first.replace("-", ""))
    last_clean = _alnum("".join(tok.replace("-", "") for tok in last_tokens))

    if not last_clean and len(tokens) == 1:
        last_clean = first_clean

    return f"{first_clean[:n_prefix]}{last_clean}"


@dataclass(frozen=True)
class DataPaths:
    """Resolve the canonical locations for raw and processed artifacts."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    metadata_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "DataPaths":
        raw = root / "data" / "raw"
        processed = root / "data" / "processed"
        metadata = root / "data" / "state.json"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        return cls(root=root, raw_dir=raw, processed_dir=processed, metadata_path=metadata)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def coerce_commence_time(ts: Optional[str]) -> Optional[pd.Timestamp]:
    if not ts:
        return None
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return None


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    if pd is None:
        raise ImportError("pandas is required to write parquet files")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to read parquet files")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def get_known_projection_columns() -> Iterable[str]:
    return (
        "passing_yards",
        "passing_tds",
        "interceptions",
        "pass_completions",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "fantasy_ppr",
        "fantasy_half_ppr",
        "fantasy_ppr_te",
        "fantasy_half_ppr_te",
        "fantasy_dk",
        "fantasy_cbs_offense",
    )
