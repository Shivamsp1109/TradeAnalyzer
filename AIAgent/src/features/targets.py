from __future__ import annotations

import pandas as pd


def add_return_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    for horizon in horizons:
        col = f"target_ret_{horizon}d"
        out[col] = out.groupby("symbol")["close"].shift(-horizon) / out["close"] - 1

    return out


def add_probability_label(df: pd.DataFrame, threshold_return: float, label_horizon: int = 60) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    ret_col = f"target_ret_{label_horizon}d"
    if ret_col not in out.columns:
        raise ValueError(f"Missing return target column: {ret_col}")

    out["target_prob_gt_8pct"] = (out[ret_col] > threshold_return).astype("Int64")
    return out
