from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "symbol", "close"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def validate_price_frame(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []

    for col in REQUIRED_PRICE_COLUMNS:
        if col not in df.columns:
            errors.append(f"missing_column:{col}")

    if not df.empty and (df["close"] <= 0).any():
        errors.append("non_positive_close")

    if not df.empty and (df["volume"] < 0).any():
        errors.append("negative_volume")

    return len(errors) == 0, errors


def add_turnover(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["turnover"] = out["close"] * out["volume"]
    out["turnover"] = np.where(np.isfinite(out["turnover"]), out["turnover"], np.nan)
    return out
