from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.model.contracts import InferenceResult


class XGBoostPredictionEngine:
    def __init__(self, artifacts_dir: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)

    def load(self, model_version: str) -> None:
        model_dir = self.artifacts_dir / model_version
        self.reg_30 = self._load_pickle(model_dir / "regressor_30d.pkl")
        self.reg_60 = self._load_pickle(model_dir / "regressor_60d.pkl")
        self.reg_90 = self._load_pickle(model_dir / "regressor_90d.pkl")
        self.cls = self._load_pickle(model_dir / "classifier_prob_gt_8pct.pkl")

        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.feature_columns = meta["feature_columns"]

    def predict_row(self, symbol: str, as_of_date: pd.Timestamp, row: pd.Series) -> InferenceResult:
        X = row[self.feature_columns].to_numpy(dtype=float).reshape(1, -1)

        r30 = float(self.reg_30.predict(X)[0])
        r60 = float(self.reg_60.predict(X)[0])
        r90 = float(self.reg_90.predict(X)[0])
        p = float(self.cls.predict_proba(X)[:, 1][0])

        model_confidence = self._confidence_score([r30, r60, r90], p)

        return InferenceResult(
            symbol=symbol,
            as_of_date=pd.to_datetime(as_of_date).date(),
            expected_return_30d=r30,
            expected_return_60d=r60,
            expected_return_90d=r90,
            probability_return_gt_8pct=p,
            model_confidence=model_confidence,
            feature_snapshot={k: float(row[k]) if pd.notna(row[k]) else None for k in self.feature_columns},
        )

    def _confidence_score(self, returns: list[float], probability: float) -> float:
        scale = np.array([returns], dtype=float)
        anchor = np.array([[0.06, 0.10, 0.12]], dtype=float)
        trend_align = float((scale @ anchor.T) / (np.linalg.norm(scale) * np.linalg.norm(anchor) + 1e-9))
        prob_conf = 1.0 - abs(probability - 0.5) * 2
        score = 0.6 * max(0.0, trend_align) + 0.4 * (1.0 - prob_conf)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with path.open("rb") as f:
            return pickle.load(f)
