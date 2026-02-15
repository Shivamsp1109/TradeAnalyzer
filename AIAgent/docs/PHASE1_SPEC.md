# Phase 1 Technical Specification (EOD Only)

## 1) Scope
Phase 1 delivers only end-of-day predictions and recommendation API for Indian stocks.

In scope:
- Daily batch data ingestion
- Stock universe filtering
- Feature computation
- XGBoost model training/inference
- Rule-based decision engine
- Explanation templates
- Backtesting framework (walk-forward + rolling retraining + OOS)
- FastAPI endpoints for prediction/recommendation/backtest summary
- PostgreSQL schema for core artifacts
- Legal and risk disclaimers in API response

Out of scope:
- Intraday signals
- Portfolio optimizer
- Regime classifier
- Sector-relative scoring

## 2) Data Sources
- NSEpy: Indian market historical data
- Yahoo Finance: price/fundamental feature support

## 3) Prediction Targets
For each eligible stock/date:
- `ret_30d`: future 30-day return
- `ret_60d`: future 60-day return
- `ret_90d`: future 90-day return
- `prob_gt_8pct`: probability that future return > 8% (default on 60-day horizon)

## 4) Universe Filters (Risk Controls)
Must exclude:
- low-liquidity stocks (configurable median daily turnover threshold)
- penny stocks (configurable minimum close price)
- low market-cap stocks (configurable minimum market cap)

## 5) Feature Set
Technical:
- MA20, MA50, MA200
- RSI
- MACD
- 3-month return
- 6-month return
- rolling volatility (std)

Fundamental:
- PE ratio
- ROE
- Revenue growth
- EPS growth
- Debt/Equity
- Free cash flow trend

## 6) Decision Engine Rules
BUY if all hold:
- expected 60-day return > 10%
- probability > 60%
- volatility < configured threshold
- fundamentals not deteriorating

Else: DO_NOT_BUY

Fail-safe:
- If model confidence < confidence_floor -> DO_NOT_BUY

## 7) Entry, Target, Time Window Logic
- Entry:
  - default: current close
  - optional signal: near MA20 pullback
- Target:
  - `target_price = current_price * (1 + selected_predicted_return)`
- Suggested window:
  - compare 30/60/90-day risk-adjusted expected return
  - choose highest score

## 8) Backtesting Requirements
- Walk-forward validation
- Rolling retraining
- Strict out-of-sample testing

Metrics:
- CAGR
- Sharpe ratio
- Max drawdown
- Win rate
- Benchmark comparison vs NIFTY 50

## 9) API Output Includes
- stock symbol
- prediction block (30/60/90 + probability)
- decision (BUY / DO_NOT_BUY)
- entry price and target price
- suggested horizon
- explanation text
- disclaimers:
  - Educational purposes only
  - Not investment advice
  - Markets involve risk

## 10) Definition of Done (Phase 1)
- Reproducible EOD pipeline from data -> recommendation
- Backtest report generated and stored
- Model version stored with metadata
- API endpoint returns recommendation + explanation + disclaimers
- Containerized service runs locally via Docker
