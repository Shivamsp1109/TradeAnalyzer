from src.model.contracts import InferenceResult, ModelArtifacts, PredictionBatch, TrainingResult
from src.model.predictor import XGBoostPredictionEngine

try:
    from src.model.trainer import XGBoostTrainingService
except ModuleNotFoundError:  # Allows partial usage where ML dependencies are unavailable.
    XGBoostTrainingService = None

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
