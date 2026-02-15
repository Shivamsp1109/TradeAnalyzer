from __future__ import annotations

from typing import Any

from src.decision.contracts import DecisionResult
from src.model.contracts import InferenceResult


class ExplanationTemplateEngine:
    def render(self, inference: InferenceResult, decision: DecisionResult, latest_row: dict[str, Any]) -> str:
        momentum_3m = self._pct(latest_row.get("return_3m"))
        momentum_6m = self._pct(latest_row.get("return_6m"))
        expected_60d = self._pct(inference.expected_return_60d)
        prob = self._pct(inference.probability_return_gt_8pct)

        pe_ratio = latest_row.get("pe_ratio")
        rev_growth = self._pct(latest_row.get("revenue_growth"))
        eps_growth = self._pct(latest_row.get("eps_growth"))

        decision_phrase = "meets BUY criteria" if decision.decision == "BUY" else "does not meet BUY criteria"
        horizon = f"{decision.suggested_horizon_days}-day"

        pe_text = f"PE around {pe_ratio:.2f}" if self._is_number(pe_ratio) else "PE unavailable"
        return (
            f"Stock shows 3-month momentum {momentum_3m} and 6-month momentum {momentum_6m}. "
            f"Fundamentals indicate {pe_text}, revenue growth {rev_growth}, and EPS growth {eps_growth}. "
            f"Model predicts {expected_60d} upside over 60 days with {prob} confidence for return above 8%. "
            f"Based on risk-adjusted expectations, the {horizon} window is preferred and the signal {decision_phrase}."
        )

    @staticmethod
    def _pct(value: Any) -> str:
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
