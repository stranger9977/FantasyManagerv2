"""Data orchestration for the Fantasy Manager application."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

from app.config import AppConfig, load_metadata, save_metadata
from app.data_sources.fantasypros import FantasyProsQuery, fetch_fantasypros, project_columns
from app.data_sources.nfl_data import import_weekly_rosters
from app.data_sources.odds_api import (
    OddsAPIClient,
    OddsWindow,
    best_line_selector,
    flatten_odds_payload,
    select_books_lenient,
)
from app.projections import (
    VALUE_COLUMNS,
    aggregate_books,
    compute_projection_values,
    map_stat_columns,
    merge_roster_metadata,
    pivot_projection_stats,
    score_fantasy,
)
from app.utils import DataPaths, coerce_commence_time, safe_read_parquet, safe_to_parquet, utcnow


@dataclass
class UpdateWindow:
    commence_from: datetime
    commence_to: datetime

    def to_metadata(self) -> dict:
        return {
            "commence_from": self.commence_from.isoformat(),
            "commence_to": self.commence_to.isoformat(),
        }


class DataManager:
    """Owns the local cache of projections and supporting data."""

    MIN_INTERVAL = timedelta(hours=1)

    def __init__(self, config: Optional[AppConfig] = None, *, root: Optional[Path] = None):
        self.config = config or AppConfig.load()
        self.paths = DataPaths.from_root(root or Path.cwd())
        self.metadata = load_metadata(self.paths.metadata_path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @property
    def last_updated(self) -> Optional[datetime]:
        stamp = self.metadata.get("last_updated")
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            return None

    def can_update(self, *, now: Optional[datetime] = None) -> bool:
        now = now or utcnow()
        last = self.last_updated
        if last is None:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return now - last >= self.MIN_INTERVAL

    # ------------------------------------------------------------------
    def update_data(self, window: UpdateWindow, *, force: bool = False) -> dict:
        if pd is None:
            raise ImportError("pandas is required to update projection data")
        now = utcnow()
        if not force and not self.can_update(now=now):
            return {"status": "skipped", "reason": "rate_limited", "last_updated": self.last_updated}

        per_book_df = self._build_odds_based_projections(window)
        fantasypros_df = self._build_fantasypros_projection(window)

        combined = pd.concat([per_book_df, fantasypros_df], ignore_index=True, sort=False)
        numeric_cols = combined.select_dtypes(include=["number"]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(0.0)
        combined = combined.sort_values(["player_name", "bookmaker"]).reset_index(drop=True)

        aggregated = aggregate_books(combined)

        safe_to_parquet(combined, self.paths.processed_dir / "projections_per_book.parquet")
        safe_to_parquet(aggregated, self.paths.processed_dir / "projections_all.parquet")

        metadata = {
            "last_updated": now.isoformat(),
            "last_window": window.to_metadata(),
            "records": int(len(combined)),
        }
        save_metadata(self.paths.metadata_path, metadata)
        self.metadata = metadata
        return {"status": "updated", "records": int(len(combined)), "aggregated": int(len(aggregated))}

    # ------------------------------------------------------------------
    def _build_odds_based_projections(self, window: UpdateWindow) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required to build odds-based projections")
        api_key = self.config.odds_api_key
        if not api_key:
            return pd.DataFrame(columns=["merge_name", "player_name", "position", "bookmaker", "commence_time"])

        client = OddsAPIClient(api_key)
        odds_window = OddsWindow(start=window.commence_from, end=window.commence_to)
        events = client.fetch_events(odds_window)
        payloads = []
        for event in events:
            event_id = event.get("id")
            if not event_id:
                continue
            markets = client.fetch_event_markets(event_id)
            payloads.append({"event_id": event_id, "odds": markets})
        if not payloads:
            return pd.DataFrame(columns=["merge_name", "player_name", "position", "bookmaker", "commence_time"])

        flattened = flatten_odds_payload(payloads)
        if flattened.empty:
            return pd.DataFrame(columns=["merge_name", "player_name", "position", "bookmaker", "commence_time"])
        keep_books = select_books_lenient(flattened)
        if keep_books:
            flattened = flattened[flattened["bookmaker"].isin(keep_books)].copy()
        best_lines = best_line_selector(flattened)
        if best_lines.empty:
            return pd.DataFrame(columns=["merge_name", "player_name", "position", "bookmaker", "commence_time"])

        best_lines = map_stat_columns(best_lines)
        best_lines = compute_projection_values(best_lines)

        rosters = import_weekly_rosters([window.commence_from.year])
        merged = merge_roster_metadata(best_lines, rosters)
        merged["commence_time"] = merged["commence_time"].map(coerce_commence_time)

        stat_wide = pivot_projection_stats(merged)
        stat_wide["commence_time"] = stat_wide["commence_time"].fillna(pd.Timestamp(window.commence_from, tz="UTC"))
        stat_wide["bookmaker"] = stat_wide["bookmaker"].fillna(stat_wide.get("book_title"))
        stat_wide.drop(columns=[col for col in stat_wide.columns if col.endswith("book_title")], errors="ignore", inplace=True)
        numeric_cols = stat_wide.select_dtypes(include=["number"]).columns
        stat_wide[numeric_cols] = stat_wide[numeric_cols].fillna(0.0)
        return stat_wide

    def _build_fantasypros_projection(self, window: UpdateWindow) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required to incorporate FantasyPros projections")
        query = FantasyProsQuery(scoring="ppr")
        data = fetch_fantasypros(self.config.fantasypros_base_url, query=query)
        if data.empty:
            return pd.DataFrame(columns=["merge_name", "player_name", "position", "bookmaker", "commence_time"])

        data = project_columns(data)
        data = score_fantasy(data)
        data.rename(columns={"name": "player_name"}, inplace=True)
        data["commence_time"] = pd.Timestamp(window.commence_from, tz="UTC")
        required = ["merge_name", "player_name", "position", "bookmaker", "commence_time"]
        for col in required:
            if col not in data.columns:
                data[col] = None
        for col in VALUE_COLUMNS:
            if col not in data.columns:
                data[col] = 0.0
        value_cols = [col for col in VALUE_COLUMNS if col in data.columns]
        extra_cols = [col for col in data.columns if col not in required + value_cols]
        return data[required + value_cols + extra_cols]

    # ------------------------------------------------------------------
    def load_projections(self) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required to load projection data")
        path = self.paths.processed_dir / "projections_all.parquet"
        df = safe_read_parquet(path)
        if df.empty:
            sample_path = Path("data/sample_data/projections_all_sample.parquet")
            if sample_path.exists():
                return pd.read_json(sample_path)
        return df

    def load_per_book(self) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required to load projection data")
        path = self.paths.processed_dir / "projections_per_book.parquet"
        df = safe_read_parquet(path)
        if df.empty:
            sample_path = Path("data/sample_data/projections_per_book_sample.parquet")
            if sample_path.exists():
                return pd.read_json(sample_path)
        return df
