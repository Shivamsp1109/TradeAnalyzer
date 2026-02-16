# API (Step 8)

## Endpoints
- `GET /health`
  - Basic liveness check
- `POST /recommendation`
  - Request: `RecommendationRequest`
  - Query: `model_version` optional (defaults to `MODEL_DEFAULT_VERSION`)
  - Response: `RecommendationResponse`
- `GET /backtest/latest`
  - Returns latest `BacktestSummaryResponse`

## Notes
- Set `MODEL_DEFAULT_VERSION` in `.env` to the artifact folder name.
- Models are loaded from `MODEL_ARTIFACTS_DIR/{MODEL_DEFAULT_VERSION}`.
