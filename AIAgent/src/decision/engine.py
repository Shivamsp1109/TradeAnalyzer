from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import AppConfig, get_config
from src.decision.contracts import DecisionResult, EntryTargetPlan
from src.model.contracts import InferenceResult


class DecisionEngine:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def evaluate(self, inference: InferenceResult, latest_row: pd.Series | dict[str, Any]) -> DecisionResult:
        row = latest_row if isinstance(latest_row, dict) else latest_row.to_dict()

        current_price = float(row.get("close", 0.0))
        ma20 = self._to_float_or_nan(row.get("ma_20"))
        volatility = self._to_float_or_nan(row.get("volatility_63d"))

        expected_returns = {
            30: float(inference.expected_return_30d),
            60: float(inference.expected_return_60d),
            90: float(inference.expected_return_90d),
        }
        risk_adjusted = self._risk_adjusted_returns(expected_returns, volatility)
        suggested_horizon = max(risk_adjusted, key=risk_adjusted.get)
        selected_expected_return = expected_returns[suggested_horizon]

        fundamentals_ok = self._fundamentals_not_deteriorating(row)

        checks = {
            "expected_60d_return": inference.expected_return_60d > self.config.decision.min_expected_60d_return,
            "probability": inference.probability_return_gt_8pct > self.config.decision.min_probability,
            "volatility": np.isfinite(volatility) and volatility < self.config.decision.max_volatility,
            "fundamentals": fundamentals_ok,
            "model_confidence": inference.model_confidence >= self.config.decision.confidence_floor,
        }

        decision = "BUY" if all(checks.values()) else "DO_NOT_BUY"
        entry_target = self._build_entry_target(current_price, ma20, selected_expected_return)

        return DecisionResult(
            symbol=inference.symbol,
            as_of_date=inference.as_of_date,
            decision=decision,
            suggested_horizon_days=suggested_horizon,
            selected_expected_return=selected_expected_return,
            entry_target=entry_target,
            rule_checks=checks,
            risk_adjusted_returns=risk_adjusted,
            meta={"evaluated_at_utc": datetime.utcnow().isoformat()},
        )

    def _risk_adjusted_returns(self, expected_returns: dict[int, float], volatility: float) -> dict[int, float]:
        vol = float(volatility) if np.isfinite(volatility) and volatility > 0 else 1e-6
        scores: dict[int, float] = {}
        for horizon, expected_return in expected_returns.items():
            horizon_scale = np.sqrt(max(horizon, 1) / 252.0)
            adjusted_vol = max(vol * horizon_scale, 1e-6)
            scores[horizon] = float(expected_return / adjusted_vol)
        return scores

    def _fundamentals_not_deteriorating(self, row: dict[str, Any]) -> bool:
        revenue_growth = self._to_float_or_nan(row.get("revenue_growth"))
        eps_growth = self._to_float_or_nan(row.get("eps_growth"))
        free_cash_flow = self._to_float_or_nan(row.get("free_cash_flow"))

        checks = []
        if np.isfinite(revenue_growth):
            checks.append(revenue_growth >= 0)
        if np.isfinite(eps_growth):
            checks.append(eps_growth >= 0)
        if np.isfinite(free_cash_flow):
            checks.append(free_cash_flow >= 0)

        return all(checks) if checks else False

    def _build_entry_target(self, current_price: float, ma20: float, selected_return: float) -> EntryTargetPlan:
        entry_price = current_price
        entry_logic = "current_price"

        if np.isfinite(ma20) and ma20 > 0:
            distance = abs(current_price - ma20) / ma20
            if distance <= self.config.decision.ma20_pullback_tolerance:
                entry_price = ma20
                entry_logic = "near_ma20_pullback"

        target_price = float(entry_price * (1.0 + selected_return))
        return EntryTargetPlan(
            current_price=float(current_price),
            entry_price=float(entry_price),
            target_price=target_price,
            entry_logic=entry_logic,
        )

    @staticmethod
    def _to_float_or_nan(value: Any) -> float:
        try:
            if value is None:
                return float("nan")
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
