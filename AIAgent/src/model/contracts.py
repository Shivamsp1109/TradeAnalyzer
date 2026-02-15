from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelArtifacts:
    model_version: str
    model_dir: Path
    regressor_paths: dict[int, Path]
    classifier_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    model_version: str
    trained_until: date
    training_start: date
    training_end: date
    feature_set_version: str
    metrics: dict[str, Any]
    hyperparams: dict[str, Any]
    artifact_uri: str
    artifacts: ModelArtifacts


@dataclass(frozen=True)
class InferenceResult:
    symbol: str
    as_of_date: date
    expected_return_30d: float
    expected_return_60d: float
    expected_return_90d: float
    probability_return_gt_8pct: float
    model_confidence: float
    feature_snapshot: dict[str, Any]


@dataclass(frozen=True)
class PredictionBatch:
    predictions: pd.DataFrame
