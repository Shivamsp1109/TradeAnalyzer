# Feature Engineering and Labeling (Step 4)

## What this layer does
- Computes technical features from OHLCV:
  - MA20, MA50, MA200
  - RSI(14)
  - MACD, signal, histogram
  - 3-month return (63 trading days)
  - 6-month return (126 trading days)
  - Rolling 63-day annualized volatility
- Merges fundamentals from Yahoo data:
  - PE, ROE, revenue growth, EPS growth, debt/equity, free cash flow, market cap
- Creates model targets:
  - `target_ret_30d`
  - `target_ret_60d`
  - `target_ret_90d`
  - `target_prob_gt_8pct` (binary label from 60-day return threshold)

## Main entrypoint
- `src/features/dataset.py`
  - `TrainingDatasetBuilder.build(ohlcv, fundamentals)` returns:
    - `data`: model-ready DataFrame
    - `feature_columns`
    - `target_columns`

## Notes
- Rows without target labels (tail region by horizon) are dropped.
- Numeric NA values are median-imputed in this step for training readiness.
