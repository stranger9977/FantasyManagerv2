"""Utilities for collecting FantasyPros projection data."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

import requests

from app.utils import make_merge_key

DEFAULT_POSITIONS = ("qb", "rb", "wr", "te")


@dataclass
class FantasyProsQuery:
    scoring: str = "ppr"
    week: Optional[int] = None
    season: Optional[int] = None

    @property
    def params(self) -> dict:
        params: dict = {}
        if self.week is not None:
            params["week"] = self.week
        if self.season is not None:
            params["season"] = self.season
        return params


POSITION_COLUMN_MAP = {
    "QB": {
        "Pass Yds": "passing_yards",
        "Pass TD": "passing_tds",
        "Pass INT": "interceptions",
        "Pass Comp": "pass_completions",
        "Rush Yds": "rushing_yards",
        "Rush TD": "rushing_tds",
    },
    "RB": {
        "Rush Yds": "rushing_yards",
        "Rush TD": "rushing_tds",
        "Rec": "receptions",
        "Rec Yds": "receiving_yards",
        "Rec TD": "receiving_tds",
    },
    "WR": {
        "Rec": "receptions",
        "Rec Yds": "receiving_yards",
        "Rec TD": "receiving_tds",
        "Rush Yds": "rushing_yards",
        "Rush TD": "rushing_tds",
    },
    "TE": {
        "Rec": "receptions",
        "Rec Yds": "receiving_yards",
        "Rec TD": "receiving_tds",
    },
}


def _build_url(base_url: str, position: str, scoring: str) -> str:
    return f"{base_url}/{position}.php" if scoring == "ppr" else f"{base_url}/{position}-{scoring}.php"


def _clean_player_column(series: pd.Series) -> pd.DataFrame:
    name_team = series.str.extract(r"(?P<name>[^\(]+)(?:\((?P<meta>[^\)]+)\))?")
    name_team["name"] = name_team["name"].str.strip()
    meta_split = name_team["meta"].str.split(" - ", expand=True)
    name_team["team"] = meta_split[0].str.upper().str.strip()
    name_team["position"] = meta_split[1].str.upper().str.strip()
    return name_team.drop(columns=["meta"])


def fetch_position(
    base_url: str,
    position: str,
    *,
    session: Optional[requests.Session] = None,
    query: Optional[FantasyProsQuery] = None,
) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to parse FantasyPros projections")
    query = query or FantasyProsQuery()
    position = position.lower()
    url = _build_url(base_url, position, query.scoring)
    session = session or requests.Session()
    resp = session.get(url, params=query.params, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    if not tables:
        return pd.DataFrame()
    df = tables[0].copy()
    df.columns = [re.sub(r"\s+", " ", col).strip() for col in df.columns]
    parsed = _clean_player_column(df.iloc[:, 0])
    df = pd.concat([parsed, df.drop(columns=df.columns[0])], axis=1)
    df["merge_name"] = df["name"].map(make_merge_key)
    df.rename(columns={df.columns[0]: "player_name"}, inplace=True)

    # Standardize numeric columns
    for col in df.columns:
        if col in {"player_name", "team", "position", "merge_name"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["bookmaker"] = "fantasypros"
    return df


def fetch_fantasypros(
    base_url: str,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    *,
    session: Optional[requests.Session] = None,
    query: Optional[FantasyProsQuery] = None,
) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to parse FantasyPros projections")
    frames: List[pd.DataFrame] = []
    for pos in positions:
        data = fetch_position(base_url, pos, session=session, query=query)
        if data.empty:
            continue
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["position"] = df["position"].fillna(df["position"].str.upper())
    df["position"] = df["position"].str.upper()
    return df


def project_columns(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to project FantasyPros columns")
    if df.empty:
        return df
    df = df.copy()
    stat_columns = {}
    for pos, mapping in POSITION_COLUMN_MAP.items():
        mask = df["position"].eq(pos)
        if not mask.any():
            continue
        for source_col, target_col in mapping.items():
            if source_col in df.columns:
                stat_columns.setdefault(target_col, 0.0)
                df.loc[mask, target_col] = df.loc[mask, source_col]

    default_cols = {
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "pass_completions": 0.0,
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_tds": 0.0,
    }
    for col, default in default_cols.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    return df
