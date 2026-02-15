# Model Training (Step 5)

## Scope
- Multi-output modeling with separate models:
  - XGBoost regressor for 30-day return
  - XGBoost regressor for 60-day return
  - XGBoost regressor for 90-day return
- Probability model:
  - XGBoost classifier (or Dummy fallback when class imbalance is extreme)
  - sigmoid calibration using validation data when possible

## Modules
- `src/model/trainer.py`
  - `XGBoostTrainingService`
  - `XGBoostPredictionEngine`
- `src/model/metrics.py`
  - regression and classification metric helpers
- `src/model/registry.py`
  - DB registration utilities for `model_versions`
- `src/model/contracts.py`
  - typed result containers

## Artifacts
Artifacts are stored at:
- `${MODEL_ARTIFACTS_DIR}/{model_version}/`

Each version directory contains:
- `regressor_30d.pkl`
- `regressor_60d.pkl`
- `regressor_90d.pkl`
- `classifier_prob_gt_8pct.pkl`
- `metadata.json`

## Metrics captured
- Regression: RMSE, MAE, R2 for each horizon
- Classification: accuracy, log-loss, brier, ROC-AUC (when both classes exist)

## Persistence hooks
- `register_model_version(db, result)` inserts into `model_versions`
- `deactivate_previous_versions(db, model_name)` disables old active versions
