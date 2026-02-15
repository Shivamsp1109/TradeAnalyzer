from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestRunResult:
    strategy_name: str
    universe_name: str
    period_start: date
    period_end: date
    benchmark_symbol: str
    summary: dict[str, Any]
    daily_performance: pd.DataFrame
    trades: pd.DataFrame
    windows: pd.DataFrame


@dataclass(frozen=True)
class BacktestInputs:
    data: pd.DataFrame
    feature_columns: list[str]
    benchmark_prices: pd.DataFrame | None = None
    strategy_name: str = "xgboost_rule_strategy"
    universe_name: str = "nse_filtered"
