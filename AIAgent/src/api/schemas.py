from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendationDecision(str, Enum):
    BUY = "BUY"
    DO_NOT_BUY = "DO_NOT_BUY"


class Horizon(str, Enum):
    D30 = "30D"
    D60 = "60D"
    D90 = "90D"


class PredictionBlock(BaseModel):
    expected_return_30d: float = Field(..., description="Predicted 30-day forward return")
    expected_return_60d: float = Field(..., description="Predicted 60-day forward return")
    expected_return_90d: float = Field(..., description="Predicted 90-day forward return")
    probability_return_gt_8pct: float = Field(..., ge=0.0, le=1.0)
    model_confidence: float = Field(..., ge=0.0, le=1.0)


class RecommendationRequest(BaseModel):
    symbol: str = Field(..., description="NSE symbol")
    as_of_date: Optional[date] = Field(default=None, description="EOD date in YYYY-MM-DD")


class EntryTargetBlock(BaseModel):
    current_price: float
    entry_price: float
    target_price: float
    entry_logic: str


class FeatureSummary(BaseModel):
    momentum_3m: Optional[float] = None
    momentum_6m: Optional[float] = None
    volatility: Optional[float] = None
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None


class RecommendationResponse(BaseModel):
    symbol: str
    as_of_date: date
    decision: RecommendationDecision
    suggested_horizon: Horizon
    prediction: PredictionBlock
    pricing: EntryTargetBlock
    explanation: str
    feature_summary: FeatureSummary
    disclaimers: List[str]
    generated_at: datetime


class ModelVersionPayload(BaseModel):
    model_name: str
    model_version: str
    trained_until: date
    training_start: date
    training_end: date
    feature_set_version: str
    hyperparams: dict
    metrics: dict
    artifact_uri: str


class BacktestSummaryResponse(BaseModel):
    report_id: str
    strategy_name: str
    universe_name: str
    period_start: date
    period_end: date
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    benchmark_symbol: str
    benchmark_cagr: float
    generated_at: datetime
