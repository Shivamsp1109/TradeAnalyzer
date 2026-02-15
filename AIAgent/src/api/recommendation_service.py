from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.api.schemas import (
    EntryTargetBlock,
    FeatureSummary,
    Horizon,
    PredictionBlock,
    RecommendationDecision,
    RecommendationResponse,
)
from src.decision.engine import DecisionEngine
from src.explanation.templates import ExplanationTemplateEngine
from src.model.contracts import InferenceResult

DEFAULT_DISCLAIMERS = [
    "Educational purposes only",
    "Not investment advice",
    "Markets involve risk",
]


class RecommendationBuilderService:
    def __init__(self) -> None:
        self.decision_engine = DecisionEngine()
        self.explainer = ExplanationTemplateEngine()

    def build(self, inference: InferenceResult, latest_row: pd.Series | dict[str, Any]) -> RecommendationResponse:
        row = latest_row if isinstance(latest_row, dict) else latest_row.to_dict()
        decision_result = self.decision_engine.evaluate(inference=inference, latest_row=row)
        explanation = self.explainer.render(inference=inference, decision=decision_result, latest_row=row)

        horizon_enum = self._to_horizon(decision_result.suggested_horizon_days)
        decision_enum = RecommendationDecision(decision_result.decision)

        return RecommendationResponse(
            symbol=inference.symbol,
            as_of_date=inference.as_of_date,
            decision=decision_enum,
            suggested_horizon=horizon_enum,
            prediction=PredictionBlock(
                expected_return_30d=inference.expected_return_30d,
                expected_return_60d=inference.expected_return_60d,
                expected_return_90d=inference.expected_return_90d,
                probability_return_gt_8pct=inference.probability_return_gt_8pct,
                model_confidence=inference.model_confidence,
            ),
            pricing=EntryTargetBlock(
                current_price=decision_result.entry_target.current_price,
                entry_price=decision_result.entry_target.entry_price,
                target_price=decision_result.entry_target.target_price,
                entry_logic=decision_result.entry_target.entry_logic,
            ),
            explanation=explanation,
            feature_summary=FeatureSummary(
                momentum_3m=self._as_float(row.get("return_3m")),
                momentum_6m=self._as_float(row.get("return_6m")),
                volatility=self._as_float(row.get("volatility_63d")),
                pe_ratio=self._as_float(row.get("pe_ratio")),
                roe=self._as_float(row.get("roe")),
                revenue_growth=self._as_float(row.get("revenue_growth")),
                eps_growth=self._as_float(row.get("eps_growth")),
                debt_to_equity=self._as_float(row.get("debt_to_equity")),
            ),
            disclaimers=DEFAULT_DISCLAIMERS,
            generated_at=datetime.utcnow(),
        )

    @staticmethod
    def _to_horizon(days: int) -> Horizon:
        return {
            30: Horizon.D30,
            60: Horizon.D60,
            90: Horizon.D90,
        }.get(days, Horizon.D60)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
