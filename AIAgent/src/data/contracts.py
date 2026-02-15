from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SymbolIngestionResult:
    symbol: str
    ohlcv: pd.DataFrame
    fundamentals: dict[str, Any]


@dataclass(frozen=True)
class UniverseIngestionResult:
    ohlcv: pd.DataFrame
    fundamentals: pd.DataFrame
    filter_report: pd.DataFrame


@dataclass(frozen=True)
class IngestionRequest:
    symbols: list[str]
    start_date: date
    end_date: date
