from src.db.base import Base
from src.db.models import BacktestReport, LivePerformance, ModelVersion, Prediction, Signal  # noqa: F401
from src.db.session import engine


def create_all_tables() -> None:
    Base.metadata.create_all(bind=engine)
