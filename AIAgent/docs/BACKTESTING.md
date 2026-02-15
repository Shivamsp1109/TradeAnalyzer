# Backtesting (Step 7)

## Implemented
- Walk-forward validation with strict out-of-sample evaluation
- Rolling retraining on each walk-forward window
- Strategy simulation using Decision Engine outputs
- Benchmark comparison against NIFTY 50 (`^NSEI` by default)
- Report persistence hook into `backtest_reports`

## Modules
- `src/backtest/contracts.py`
  - `BacktestInputs`, `BacktestRunResult`
- `src/backtest/metrics.py`
  - CAGR, Sharpe, Max Drawdown, Win Rate
- `src/backtest/engine.py`
  - `WalkForwardBacktester.run(inputs)`
- `src/backtest/reporting.py`
  - `persist_backtest_report(db, result)`

## Input expectation
Backtester expects a labeled dataset from Step 4 that includes:
- model features (`feature_columns`)
- targets: `target_ret_30d`, `target_ret_60d`, `target_ret_90d`, `target_prob_gt_8pct`

## Output summary metrics
- `cagr`
- `sharpe`
- `max_drawdown`
- `win_rate`
- `benchmark_cagr`
- `benchmark_sharpe`
- total trade and window stats

## Example usage
```python
from src.backtest import BacktestInputs, WalkForwardBacktester

backtester = WalkForwardBacktester()
result = backtester.run(
    BacktestInputs(
        data=labeled_dataset.data,
        feature_columns=labeled_dataset.feature_columns,
        benchmark_prices=benchmark_df,
        strategy_name="xgboost_rule_strategy",
        universe_name="nse_filtered",
    )
)
print(result.summary)
```
