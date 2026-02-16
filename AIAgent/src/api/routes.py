from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.recommendation_service import RecommendationBuilderService
from src.api.schemas import BacktestSummaryResponse, RecommendationRequest, RecommendationResponse
from src.common.config import get_config
from src.data.ingestion import MarketDataIngestionService
from src.features.engineering import FeatureEngineeringService
from src.model.predictor import XGBoostPredictionEngine

try:
    from src.db.session import SessionLocal
    from src.db.models import BacktestReport
except ModuleNotFoundError:
    SessionLocal = None
    BacktestReport = None

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.post("/recommendation", response_model=RecommendationResponse)
def recommend(
    payload: RecommendationRequest,
    model_version: str | None = Query(default=None, description="Model version to load"),
) -> RecommendationResponse:
    config = get_config()
    as_of_date = payload.as_of_date or date.today()

    lookback_days = max(config.training.train_window_days * 2, 800)
    start_date = as_of_date - timedelta(days=lookback_days)

    ingestion = MarketDataIngestionService(config)
    symbol_result = ingestion.fetch_symbol_data(payload.symbol, start_date, as_of_date)

    if symbol_result.ohlcv.empty:
        raise HTTPException(status_code=404, detail="No price data returned for symbol")

    features = FeatureEngineeringService().build_features(symbol_result.ohlcv, pd.DataFrame([symbol_result.fundamentals]))
    feat_df = features.features

    if feat_df.empty:
        raise HTTPException(status_code=500, detail="Feature generation failed")

    feat_df["date"] = pd.to_datetime(feat_df["date"]).dt.date
    latest_row = feat_df[feat_df["date"] <= as_of_date].sort_values("date").tail(1)

    if latest_row.empty:
        raise HTTPException(status_code=404, detail="No features available as of requested date")

    selected_version = model_version or config.model.default_model_version
    if not selected_version:
        raise HTTPException(status_code=400, detail="MODEL_DEFAULT_VERSION not set and no model_version provided")
    artifacts_dir = config.model.artifacts_dir

    engine = XGBoostPredictionEngine(artifacts_dir)
    try:
        engine.load(selected_version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model version not found: {selected_version}") from exc

    row = latest_row.iloc[0]
    inference = engine.predict_row(payload.symbol.upper(), pd.Timestamp(row["date"]), row)
    response = RecommendationBuilderService().build(inference=inference, latest_row=row)
    return response


@router.get("/backtest/latest", response_model=BacktestSummaryResponse)
def latest_backtest() -> BacktestSummaryResponse:
    if SessionLocal is None or BacktestReport is None:
        raise HTTPException(status_code=503, detail="Database integration not available")

    with SessionLocal() as db:
        report = db.query(BacktestReport).order_by(BacktestReport.generated_at.desc()).first()

    if report is None:
        raise HTTPException(status_code=404, detail="No backtest reports found")

    return BacktestSummaryResponse(
        report_id=str(report.id),
        strategy_name=report.strategy_name,
        universe_name=report.universe_name,
        period_start=report.period_start,
        period_end=report.period_end,
        cagr=report.cagr,
        sharpe=report.sharpe,
        max_drawdown=report.max_drawdown,
        win_rate=report.win_rate,
        benchmark_symbol=report.benchmark_symbol,
        benchmark_cagr=report.benchmark_cagr,
        generated_at=report.generated_at,
    )
