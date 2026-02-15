from src.model.contracts import InferenceResult, ModelArtifacts, PredictionBatch, TrainingResult
from src.model.trainer import XGBoostPredictionEngine, XGBoostTrainingService

try:
    from src.model.registry import deactivate_previous_versions, register_model_version
except ModuleNotFoundError:  # Allows training-only usage when SQLAlchemy isn't installed yet.
    deactivate_previous_versions = None
    register_model_version = None

__all__ = [
    "InferenceResult",
    "ModelArtifacts",
    "PredictionBatch",
    "TrainingResult",
    "XGBoostTrainingService",
    "XGBoostPredictionEngine",
    "register_model_version",
    "deactivate_previous_versions",
]
