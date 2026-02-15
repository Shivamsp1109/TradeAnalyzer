from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from nsepy import get_history


def to_yahoo_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.endswith(".NS"):
        return s
    return f"{s}.NS"


class NsePriceClient:
    """Fetches daily OHLCV from NSEpy."""

    def fetch_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        raw = get_history(symbol=symbol.upper(), start=start_date, end=end_date)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        expected = ["date", "open", "high", "low", "close", "volume"]
        df = df[[col for col in expected if col in df.columns]]
        df["symbol"] = symbol.upper()
        df["source"] = "nsepy"
        return df


class YahooFinanceClient:
    """Fetches fallback OHLCV and fundamentals from Yahoo Finance."""

    def fetch_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        yf_symbol = to_yahoo_symbol(symbol)
        raw = yf.download(
            yf_symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        expected = ["date", "open", "high", "low", "close", "volume"]
        df = df[[col for col in expected if col in df.columns]]
        df["symbol"] = symbol.upper()
        df["source"] = "yahoo"
        return df

    def fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        ticker = yf.Ticker(to_yahoo_symbol(symbol))
        info = ticker.info or {}

        return {
            "symbol": symbol.upper(),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "roe": info.get("returnOnEquity"),
            "revenue_growth": info.get("revenueGrowth"),
            "eps_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "market_cap": info.get("marketCap"),
            "fetched_at": datetime.utcnow(),
            "source": "yahoo",
        }
