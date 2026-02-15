from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.fillna(0.0)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    n = len(equity)
    if n == 0:
        return 0.0
    years = max(n / periods_per_year, 1e-9)
    total_return = float(equity.iloc[-1])
    return float(total_return ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float, periods_per_year: int = 252) -> float:
    returns = returns.fillna(0.0)
    if returns.empty:
        return 0.0
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    std = float(excess.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.fillna(0.0)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def win_rate(trade_returns: pd.Series) -> float:
    trade_returns = trade_returns.dropna()
    if trade_returns.empty:
        return 0.0
    return float((trade_returns > 0).mean())
