from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from src.db.models import ModelVersion
from src.model.contracts import TrainingResult


def register_model_version(db: Session, result: TrainingResult, is_active: bool = True) -> ModelVersion:
    record = ModelVersion(
        model_name=result.model_name,
        model_version=result.model_version,
        feature_set_version=result.feature_set_version,
        training_start=result.training_start,
        training_end=result.training_end,
        trained_until=result.trained_until,
        hyperparams=result.hyperparams,
        metrics=result.metrics,
        artifact_uri=result.artifact_uri,
        is_active=is_active,
    )
    db.add(record)
    db.flush()
    return record


def deactivate_previous_versions(db: Session, model_name: str, as_of: date | None = None) -> int:
    query = db.query(ModelVersion).filter(ModelVersion.model_name == model_name, ModelVersion.is_active.is_(True))
    updated = 0
    for row in query.all():
        row.is_active = False
        updated += 1
    return updated
