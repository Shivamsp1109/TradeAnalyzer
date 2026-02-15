# Implementation Plan

## Step 1 (Current)
- Finalize architecture and Phase 1 scope
- Create repository structure and base documents

## Step 2
- Implement config system and shared schemas
- Define PostgreSQL schema (tables for predictions, signals, backtests, models)

## Step 3
- Build ingestion layer for NSEpy + Yahoo Finance
- Add quality checks and risk filters (liquidity, price, market cap)

## Step 4
- Implement feature engineering module (technical + fundamental)
- Build dataset assembly for 30/60/90-day targets

## Step 5
- Train XGBoost models and probability calibration
- Add model registry/version metadata

## Step 6
- Implement decision engine + fail-safe + horizon selection
- Implement explanation templates

## Step 7
- Implement walk-forward and rolling backtest engine
- Compute strategy and benchmark metrics

## Step 8
- Expose FastAPI endpoints and DTOs for Java frontend
- Add legal/risk disclaimer into every recommendation response

## Step 9
- Dockerize, add Render deployment files
- Add weekly retraining + drift detection jobs

## Step 10
- Integrate Java frontend and track live performance
