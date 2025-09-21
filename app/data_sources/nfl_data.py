"""Thin wrappers around :mod:`nfl_data_py` that guard imports and provide caching."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import pandas as pd


@lru_cache(maxsize=8)
def import_weekly_rosters(years: Iterable[int]) -> pd.DataFrame:
    from nfl_data_py import import_weekly_rosters as _import

    return _import(list(years))


@lru_cache(maxsize=8)
def import_weekly_data(years: Iterable[int]) -> pd.DataFrame:
    from nfl_data_py import import_weekly_data as _import

    return _import(list(years))


@lru_cache(maxsize=8)
def import_schedules(years: Iterable[int]) -> pd.DataFrame:
    from nfl_data_py import import_schedules as _import

    return _import(list(years))


@lru_cache(maxsize=1)
def import_team_desc() -> pd.DataFrame:
    from nfl_data_py import import_team_desc as _import

    return _import()
