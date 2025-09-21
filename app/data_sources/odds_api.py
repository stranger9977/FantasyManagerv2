"""Wrappers around The Odds API for player prop markets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore
import requests

from app.utils import american_to_probability, make_merge_key

DEFAULT_MARKETS = (
    "player_anytime_td",
    "player_pass_completions",
    "player_pass_interceptions",
    "player_pass_tds",
    "player_pass_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_yds",
)

DEFAULT_REGIONS = ("us", "us2")


@dataclass
class OddsWindow:
    start: datetime
    end: datetime

    @property
    def params(self) -> dict:
        return {
            "commenceTimeFrom": self.start.replace(microsecond=0).isoformat() + "Z",
            "commenceTimeTo": self.end.replace(microsecond=0).isoformat() + "Z",
        }


class OddsAPIClient:
    def __init__(self, api_key: str, *, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def _request(self, url: str, params: Optional[dict] = None) -> dict:
        params = {**(params or {}), "apiKey": self.api_key}
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_events(self, window: OddsWindow, *, regions: Iterable[str] = DEFAULT_REGIONS) -> List[dict]:
        params = {
            **window.params,
            "regions": ",".join(regions),
            "dateFormat": "iso",
        }
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
        data = self._request(url, params=params)
        return list(data)

    def fetch_event_markets(
        self,
        event_id: str,
        *,
        regions: Iterable[str] = DEFAULT_REGIONS,
        markets: Iterable[str] = DEFAULT_MARKETS,
        odds_format: str = "american",
    ) -> dict:
        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
        return self._request(url, params=params)


TEAMISH_TERMS = ("defense", "d/st", "dst", "team")


def flatten_odds_payload(records: List[dict]) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to flatten odds payloads")
    rows = []
    for payload in records:
        event_meta = {
            "game_id": payload.get("event_id") or payload.get("id"),
        }
        odds_block = payload.get("odds") or payload
        meta = {
            **event_meta,
            "commence_time": odds_block.get("commence_time"),
            "home_team": odds_block.get("home_team"),
            "away_team": odds_block.get("away_team"),
        }
        for bookmaker in odds_block.get("bookmakers", []) or []:
            book_key = bookmaker.get("key")
            book_title = bookmaker.get("title")
            for market in bookmaker.get("markets", []) or []:
                market_key = market.get("key")
                last_update = market.get("last_update")
                for outcome in market.get("outcomes", []) or []:
                    rows.append(
                        {
                            **meta,
                            "bookmaker": book_key,
                            "book_title": book_title,
                            "market": market_key,
                            "last_update": last_update,
                            "label": outcome.get("name"),
                            "entity": outcome.get("description"),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df["implied_prob"] = df["price"].map(american_to_probability)
    df["merge_name"] = df["entity"].map(make_merge_key)

    if {"entity", "home_team", "away_team"}.issubset(df.columns):
        entity = df["entity"].fillna("").str.lower()
        home = df["home_team"].fillna("").str.lower()
        away = df["away_team"].fillna("").str.lower()
        teamish = entity.isin({*home.unique(), *away.unique()})
        for term in TEAMISH_TERMS:
            teamish |= entity.str.contains(term, na=False)
        df = df.loc[~teamish].copy()
    return df


def select_books_lenient(df: pd.DataFrame) -> List[str]:
    if pd is None:
        raise ImportError("pandas is required to analyse bookmaker coverage")
    if df.empty:
        return []
    subset = df.dropna(subset=["entity"]).drop_duplicates(subset=["market", "bookmaker", "entity"])
    pivot = subset.groupby(["market", "bookmaker"]).size().unstack("bookmaker", fill_value=0)
    best = pivot.max(axis=1).replace(0, pd.NA)
    coverage = (pivot > 0).sum(axis=0)
    global_ratio = pivot.sum(axis=0) / best.fillna(0).sum()
    keep_mask = (coverage >= 4) & (global_ratio >= 0.40)
    return sorted(pivot.columns[keep_mask])


def best_line_selector(df: pd.DataFrame) -> pd.DataFrame:
    if pd is None:
        raise ImportError("pandas is required to select best odds lines")
    if df.empty:
        return df
    df = df.copy()
    over_under = df[df["label"].isin(["Over", "Under"])]
    over = over_under[over_under["label"] == "Over"].copy()
    over["over_prob"] = over["implied_prob"]
    over["dist"] = (over["over_prob"] - 0.5).abs()
    over = over.sort_values(["merge_name", "bookmaker", "market", "dist", "commence_time"])
    ou_best = over.drop_duplicates(subset=["merge_name", "bookmaker", "market"], keep="first")

    td = df[(df["market"] == "player_anytime_td") & (df["label"] == "Yes")]\
        .sort_values(["merge_name", "bookmaker", "commence_time"])
    td_best = td.drop_duplicates(subset=["merge_name", "bookmaker", "market"], keep="last")

    combined = pd.concat([ou_best, td_best], ignore_index=True, sort=False)
    combined["over_prob"] = pd.to_numeric(combined.get("over_prob"), errors="coerce")
    return combined
