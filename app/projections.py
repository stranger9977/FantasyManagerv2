"""Projection assembly utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

try:  # pragma: no cover - optional dependencies
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

from app.utils import get_known_projection_columns, make_merge_key

STAT_MAP = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_pass_interceptions": "interceptions",
    "player_pass_completions": "pass_completions",
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_rush_yds": "rushing_yards",
    "player_anytime_td": "anytime_td_prob",
}


@dataclass
class ProjectionConfig:
    rec_points: float = 1.0
    te_bonus: float = 0.0


def _pts_offense_generic(row: pd.Series, *, rec_pts: float, te_bonus: float) -> float:
    pos = str(row.get("position", "")).upper()
    bonus = te_bonus if pos == "TE" else 0.0
    rec_mult = rec_pts + bonus

    pts = 0.0
    pts += 4 * row.get("passing_tds", 0.0) + row.get("passing_yards", 0.0) / 25.0 - 2 * row.get("interceptions", 0.0)
    pts += 2 * row.get("passing_2pt", 0.0)
    pts += 6 * row.get("rushing_tds", 0.0) + 0.1 * row.get("rushing_yards", 0.0) + 2 * row.get("rushing_2pt", 0.0)
    pts += 6 * row.get("receiving_tds", 0.0) + 0.1 * row.get("receiving_yards", 0.0) + 2 * row.get("receiving_2pt", 0.0)
    pts += rec_mult * row.get("receptions", 0.0)
    pts -= 2 * row.get("fumbles_lost", 0.0)
    return float(pts)


def _pts_draftkings(row: pd.Series) -> float:
    pts = 0.0
    pts += 4 * row.get("passing_tds", 0.0) + 0.04 * row.get("passing_yards", 0.0) - row.get("interceptions", 0.0)
    pts += 6 * row.get("rushing_tds", 0.0) + 0.1 * row.get("rushing_yards", 0.0)
    pts += 6 * row.get("receiving_tds", 0.0) + 0.1 * row.get("receiving_yards", 0.0) + row.get("receptions", 0.0)
    pts += 3 if row.get("passing_yards", 0.0) >= 300 else 0
    pts += 3 if row.get("rushing_yards", 0.0) >= 100 else 0
    pts += 3 if row.get("receiving_yards", 0.0) >= 100 else 0
    return float(pts)


_CBS_REC_BONUS = [
    (4, 5, 3),
    (6, 7, 4),
    (8, 9, 5),
    (10, 11, 6),
    (12, 13, 7),
    (14, 15, 8),
    (16, 17, 9),
    (18, 19, 10),
    (20, 21, 11),
    (22, 23, 12),
    (24, 25, 13),
    (26, 999, 14),
]
_CBS_COMP_BONUS = [
    (20, 24, 3),
    (25, 29, 4),
    (30, 34, 5),
    (35, 39, 6),
    (40, 999, 7),
]


def _cbs_rec_bonus(recs: float) -> float:
    for lo, hi, pts in _CBS_REC_BONUS:
        if lo <= recs <= hi:
            return pts
    return 0.0


def _cbs_comp_bonus(cmps: float) -> float:
    for lo, hi, pts in _CBS_COMP_BONUS:
        if lo <= cmps <= hi:
            return pts
    return 0.0


def _pts_cbs(row: pd.Series) -> float:
    pts = 0.0
    pts += 4 * row.get("passing_tds", 0.0) + row.get("passing_yards", 0.0) / 25.0 - 2 * row.get("interceptions", 0.0)
    pts += _cbs_comp_bonus(row.get("pass_completions", 0.0))
    pts += 2 * row.get("passing_2pt", 0.0)
    pts += 3 if row.get("passing_yards", 0.0) >= 300 else 0
    pts += 6 * row.get("rushing_tds", 0.0) + 0.1 * row.get("rushing_yards", 0.0)
    pts += 3 if row.get("rushing_yards", 0.0) >= 100 else 0
    pts += 6 * row.get("receiving_tds", 0.0) + 0.1 * row.get("receiving_yards", 0.0)
    pts += 3 if row.get("receiving_yards", 0.0) >= 100 else 0
    pts += _cbs_rec_bonus(row.get("receptions", 0.0))
    pts -= 2 * row.get("fumbles_lost", 0.0)
    pts += 6 * row.get("fumble_recovery_td", 0.0)
    pts += 6 * row.get("kick_return_td", 0.0)
    pts += 6 * row.get("punt_return_td", 0.0)
    return float(pts)


VALUE_COLUMNS = [
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
]


def _split_anytime_td(row: pd.Series) -> tuple[float, float]:
    prob = row.get("anytime_td_prob", 0.0) or 0.0
    pos = str(row.get("position", "")).upper()
    if pos in {"QB", "RB"}:
        return prob, 0.0
    if pos in {"WR", "TE"}:
        return 0.0, prob
    return prob * 0.5, prob * 0.5


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "passing_yards",
        "passing_tds",
        "interceptions",
        "pass_completions",
        "passing_2pt",
        "rushing_yards",
        "rushing_tds",
        "rushing_2pt",
        "receiving_yards",
        "receiving_tds",
        "receiving_2pt",
        "receptions",
        "fumbles_lost",
        "fumble_recovery_td",
        "kick_return_td",
        "punt_return_td",
    ):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df


def score_fantasy(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to score fantasy projections")
    df = _ensure_base_columns(df)
    df = df.copy()
    df["fantasy_ppr"] = df.apply(lambda r: _pts_offense_generic(r, rec_pts=1.0, te_bonus=0.0), axis=1)
    df["fantasy_half_ppr"] = df.apply(lambda r: _pts_offense_generic(r, rec_pts=0.5, te_bonus=0.0), axis=1)
    df["fantasy_ppr_te"] = df.apply(lambda r: _pts_offense_generic(r, rec_pts=1.0, te_bonus=0.5), axis=1)
    df["fantasy_half_ppr_te"] = df.apply(lambda r: _pts_offense_generic(r, rec_pts=0.5, te_bonus=0.5), axis=1)
    df["fantasy_dk"] = df.apply(_pts_draftkings, axis=1)
    df["fantasy_cbs_offense"] = df.apply(_pts_cbs, axis=1)
    return df


def map_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to map stat columns")
    df = df.copy()
    df["stat_name"] = df["market"].map(STAT_MAP)
    return df


def compute_projection_values(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to compute projection values")
    df = df.copy()
    df["proj_value"] = df["point"]
    mask_int = (
        df["stat_name"].eq("interceptions")
        & df["point"].eq(0.5)
        & df.get("over_prob").notna()
    )
    df.loc[mask_int, "proj_value"] = df.loc[mask_int, "over_prob"]
    td_mask = df["stat_name"].eq("anytime_td_prob")
    df.loc[td_mask, "proj_value"] = df.loc[td_mask, "implied_prob"].clip(0, 0.99).map(
        lambda p: -math.log(1 - p) if pd.notna(p) else math.nan
    )
    return df


def pivot_projection_stats(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to pivot projection stats")
    if df.empty:
        return df
    df = df.copy()
    stats = df.pivot_table(
        index=["merge_name", "entity", "position", "bookmaker", "commence_time"],
        columns="stat_name",
        values="proj_value",
        aggfunc="first",
        observed=True,
    ).reset_index()
    stats = stats.rename(columns={"entity": "player_name"})
    rush_rec = stats.apply(lambda row: pd.Series(_split_anytime_td(row)), axis=1)
    stats["rushing_tds"] = rush_rec.iloc[:, 0]
    stats["receiving_tds"] = rush_rec.iloc[:, 1]
    stats = score_fantasy(stats)
    return stats


def aggregate_books(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to aggregate projection books")
    if df.empty:
        return df
    value_cols = [col for col in VALUE_COLUMNS if col in df.columns]
    grouped = (
        df.groupby(["merge_name", "player_name", "position", "commence_time"], dropna=False)[value_cols]
        .median()
        .reset_index()
        .assign(bookmaker="AGG")
    )
    ordered = (
        pd.concat([df[value_cols + ["merge_name", "player_name", "position", "bookmaker", "commence_time"]], grouped],
                  ignore_index=True)
        .sort_values(["player_name", "commence_time", "bookmaker"])
        .reset_index(drop=True)
    )
    cols = ["merge_name", "player_name", "position", "bookmaker", "commence_time"] + value_cols
    return ordered[cols]


def merge_roster_metadata(df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to merge roster metadata")
    if df.empty or roster.empty:
        return df
    roster = roster.copy()
    if "merge_name" not in roster.columns:
        name_col = next(
            (col for col in ("player_name", "gsis_name", "full_name") if col in roster.columns),
            None,
        )
        if name_col:
            roster["merge_name"] = roster[name_col].map(make_merge_key)
    meta_cols = [col for col in ("merge_name", "position", "team", "recent_team") if col in roster.columns]
    roster_meta = roster[meta_cols].drop_duplicates("merge_name")
    merged = df.merge(roster_meta, on="merge_name", how="left", suffixes=("", "_roster"))
    merged["position"] = merged["position"].fillna(merged.get("position_roster"))
    merged.drop(columns=[col for col in merged.columns if col.endswith("_roster")], inplace=True)
    return merged
