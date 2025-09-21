from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from app.data_manager import DataManager, UpdateWindow
from app.projections import VALUE_COLUMNS

st.set_page_config(page_title="Fantasy Manager Projections", layout="wide")

DEFAULT_START = datetime(2025, 9, 21, tzinfo=timezone.utc)
DEFAULT_END = datetime(2025, 9, 25, tzinfo=timezone.utc)

def _combine_date_and_time(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=timezone.utc)

def _filter_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty or "commence_time" not in df.columns:
        return df
    ts = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    mask = (ts >= start) & (ts <= end)
    return df.loc[mask].copy()

def _get_metric_options(df: pd.DataFrame) -> List[str]:
    metrics = [col for col in VALUE_COLUMNS if col in df.columns]
    if "fantasy_ppr" in metrics:
        metrics.remove("fantasy_ppr")
        metrics.insert(0, "fantasy_ppr")
    return metrics

def _prepare_player_choices(df: pd.DataFrame, metric: str, top_n: int) -> List[str]:
    if df.empty or metric not in df.columns:
        return []
    subset = df.sort_values(metric, ascending=False).head(top_n)
    return subset["player_name"].dropna().unique().tolist()

def _render_leaderboard(df: pd.DataFrame, metric: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = ["player_name", "position", "team", "commence_time", metric]
    available = [c for c in cols if c in df.columns]
    table = df.loc[df["bookmaker"] == source, available].copy()
    if "commence_time" in table.columns:
        table["commence_time"] = pd.to_datetime(table["commence_time"], utc=True).dt.tz_localize(None)
    table = table.sort_values(metric, ascending=False)
    return table

def _render_player_comparison(df: pd.DataFrame, players: List[str], metric: str) -> pd.DataFrame:
    if df.empty or not players:
        return pd.DataFrame()
    subset = df[df["player_name"].isin(players)].copy()
    pivot = subset.pivot_table(
        index=["player_name", "position", "team"],
        columns="bookmaker",
        values=metric,
        aggfunc="first",
    )
    pivot = pivot.reset_index().sort_values(metric if metric in pivot.columns else pivot.columns[-1], ascending=False)
    return pivot

def _player_bar_chart(df: pd.DataFrame, metric: str) -> alt.Chart:
    melted = df.melt(id_vars=["player_name", "position", "team"], var_name="source", value_name=metric)
    return (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("player_name", sort="-y"),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
            color="source",
            tooltip=["player_name", "source", metric]
        )
    )

def main() -> None:
    st.title("Fantasy Manager — Projection Workbench")

    manager = DataManager()

    with st.sidebar:
        st.header("Update window")
        start_date = st.date_input("Commence from", value=DEFAULT_START.date())
        end_date = st.date_input("Commence to", value=DEFAULT_END.date())
        force_update = st.checkbox("Force refresh", value=False, help="Overrides the 1 hour rate limit")
        if st.button("Update data", type="primary"):
            commence_from = _combine_date_and_time(start_date, time.min)
            commence_to = _combine_date_and_time(end_date, time.max)
            window = UpdateWindow(commence_from=commence_from, commence_to=commence_to)
            result = manager.update_data(window, force=force_update)
            if result.get("status") == "updated":
                st.success(f"Updated {result['records']} records")
            else:
                st.info("Update skipped — data was refreshed less than an hour ago.")

        last_updated = manager.last_updated
        if last_updated:
            st.caption(f"Last updated: {pd.to_datetime(last_updated).tz_localize(None)} UTC")

    per_book = manager.load_per_book()
    all_proj = manager.load_projections()

    commence_from = _combine_date_and_time(start_date, time.min)
    commence_to = _combine_date_and_time(end_date, time.max)

    per_book = _filter_window(per_book, commence_from, commence_to)
    all_proj = _filter_window(all_proj, commence_from, commence_to)

    if all_proj.empty and per_book.empty:
        st.warning("No projection data available. Use the update control to fetch the latest slate.")
        return

    metric_options = _get_metric_options(all_proj if not all_proj.empty else per_book)
    metric = st.selectbox("Projection metric", metric_options, index=0 if metric_options else 0)

    sources = sorted(per_book["bookmaker"].dropna().unique().tolist()) if not per_book.empty else []
    agg_mask = all_proj["bookmaker"].eq("AGG") if "bookmaker" in all_proj.columns else []
    if agg_mask.any():
        sources.insert(0, "AGG")
    source = st.selectbox("Projection source", sources, index=0 if sources else 0)

    st.subheader("Top players")
    leaderboard_df = _render_leaderboard(all_proj if source == "AGG" else per_book, metric, source)
    if leaderboard_df.empty:
        st.info("No players found for the selected filters.")
    else:
        top_n = st.slider("Display top N players", min_value=5, max_value=50, value=15, step=5)
        st.dataframe(leaderboard_df.head(top_n), use_container_width=True)

    st.subheader("Player comparison")
    default_choices = _prepare_player_choices(leaderboard_df, metric, top_n=10)
    players = st.multiselect("Select players to compare", options=leaderboard_df["player_name"].unique() if not leaderboard_df.empty else [], default=default_choices[:5])

    comparison_source = per_book if source != "AGG" else all_proj
    comparison_df = comparison_source.copy()
    if source == "AGG":
        comparison_df = comparison_df[comparison_df["bookmaker"].eq("AGG")]
    table = _render_player_comparison(comparison_df, players, metric)
    if table.empty:
        st.info("Select players to see per-book comparisons.")
    else:
        st.dataframe(table, use_container_width=True)
        chart = _player_bar_chart(table, metric)
        st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
