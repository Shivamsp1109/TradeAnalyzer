from __future__ import annotations

from sqlalchemy.orm import Session

from src.backtest.contracts import BacktestRunResult
from src.db.models import BacktestReport


def persist_backtest_report(db: Session, result: BacktestRunResult) -> BacktestReport:
    record = BacktestReport(
        strategy_name=result.strategy_name,
        universe_name=result.universe_name,
        period_start=result.period_start,
        period_end=result.period_end,
        benchmark_symbol=result.benchmark_symbol,
        cagr=float(result.summary.get("cagr", 0.0)),
        sharpe=float(result.summary.get("sharpe", 0.0)),
        max_drawdown=float(result.summary.get("max_drawdown", 0.0)),
        win_rate=float(result.summary.get("win_rate", 0.0)),
        benchmark_cagr=float(result.summary.get("benchmark_cagr", 0.0)),
        benchmark_sharpe=float(result.summary.get("benchmark_sharpe", 0.0)),
        config_snapshot={
            "strategy_name": result.strategy_name,
            "universe_name": result.universe_name,
            "benchmark_symbol": result.benchmark_symbol,
            "windows": int(len(result.windows)),
        },
        metrics_snapshot={
            "summary": result.summary,
            "daily_rows": int(len(result.daily_performance)),
            "trade_rows": int(len(result.trades)),
        },
    )
    db.add(record)
    db.flush()
    return record
