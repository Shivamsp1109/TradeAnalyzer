from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class EntryTargetPlan:
    current_price: float
    entry_price: float
    target_price: float
    entry_logic: str


@dataclass(frozen=True)
class DecisionResult:
    symbol: str
    as_of_date: date
    decision: str
    suggested_horizon_days: int
    selected_expected_return: float
    entry_target: EntryTargetPlan
    rule_checks: dict[str, bool]
    risk_adjusted_returns: dict[int, float]
    meta: dict[str, Any]
