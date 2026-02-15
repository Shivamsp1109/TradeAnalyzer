# System Architecture

## High-Level Flow
1. Java Frontend sends request for stock recommendation.
2. FastAPI receives request and orchestrates processing.
3. Prediction Engine generates:
   - 30-day return forecast
   - 60-day return forecast
   - 90-day return forecast
   - Probability(return > 8%)
4. Decision Engine applies rule-based filters to generate BUY / DO NOT BUY.
5. Explanation Layer returns templated explanation and risk/legal disclaimer.

## Components
- `src/api`: FastAPI routes, request/response schemas.
- `src/data`: NSEpy + Yahoo Finance ingestion, cleaning, market filters.
- `src/features`: technical + fundamental feature engineering.
- `src/model`: XGBoost training, inference, calibration, version metadata.
- `src/decision`: policy rules, fail-safe checks, entry/target logic.
- `src/explanation`: template-driven explanation assembly.
- `src/backtest`: walk-forward validation, rolling retraining, OOS metrics.
- `src/db`: PostgreSQL persistence (predictions, models, backtests, live performance).
- `infra`: Docker, deployment config for Render.

## Deployment Target
- FastAPI + Uvicorn in Docker
- PostgreSQL for persistence
- Hosted on Render

## Non-functional Requirements
- Weekly retraining
- Drift detection
- Model versioning
- Benchmark auto-update (NIFTY 50)
- Fail-safe: no recommendation when confidence < threshold
- Live recommendation performance tracking
