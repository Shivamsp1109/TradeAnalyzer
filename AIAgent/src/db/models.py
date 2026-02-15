import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(64), nullable=False)
    training_start: Mapped[Date] = mapped_column(Date, nullable=False)
    training_end: Mapped[Date] = mapped_column(Date, nullable=False)
    trained_until: Mapped[Date] = mapped_column(Date, nullable=False)
    hyperparams: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_model_name_version"),
        Index("idx_model_versions_active", "model_name", "is_active"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    as_of_date: Mapped[Date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False
    )
    expected_return_30d: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return_60d: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return_90d: Mapped[float] = mapped_column(Float, nullable=False)
    probability_return_gt_8pct: Mapped[float] = mapped_column(Float, nullable=False)
    model_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    feature_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    signal: Mapped["Signal"] = relationship(back_populates="prediction", uselist=False)

    __table_args__ = (
        UniqueConstraint("as_of_date", "symbol", "model_version_id", name="uq_predictions_date_symbol_model"),
        CheckConstraint("probability_return_gt_8pct >= 0 AND probability_return_gt_8pct <= 1", name="ck_predictions_prob_range"),
        CheckConstraint("model_confidence >= 0 AND model_confidence <= 1", name="ck_predictions_confidence_range"),
        Index("idx_predictions_symbol_date", "symbol", "as_of_date"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False
    )
    as_of_date: Mapped[Date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    suggested_horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_logic: Mapped[str] = mapped_column(String(64), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    risk_checks: Mapped[dict] = mapped_column(JSON, nullable=False)
    disclaimers: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    prediction: Mapped["Prediction"] = relationship(back_populates="signal")

    __table_args__ = (
        UniqueConstraint("as_of_date", "symbol", name="uq_signals_date_symbol"),
        CheckConstraint("decision IN ('BUY', 'DO_NOT_BUY')", name="ck_signals_decision"),
        CheckConstraint("suggested_horizon_days IN (30, 60, 90)", name="ck_signals_horizon"),
        Index("idx_signals_symbol_date", "symbol", "as_of_date"),
    )


class BacktestReport(Base):
    __tablename__ = "backtest_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    universe_name: Mapped[str] = mapped_column(String(64), nullable=False)
    period_start: Mapped[Date] = mapped_column(Date, nullable=False)
    period_end: Mapped[Date] = mapped_column(Date, nullable=False)
    benchmark_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    cagr: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_cagr: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_sharpe: Mapped[float] = mapped_column(Float, nullable=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    generated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_backtest_period", "period_start", "period_end"),
    )


class LivePerformance(Base):
    __tablename__ = "live_performance"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("signals.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_date: Mapped[Date] = mapped_column(Date, nullable=False)
    exit_date: Mapped[Date] = mapped_column(Date, nullable=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    realized_return: Mapped[float] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("horizon_days IN (30, 60, 90)", name="ck_live_perf_horizon"),
        CheckConstraint("status IN ('OPEN', 'CLOSED')", name="ck_live_perf_status"),
        Index("idx_live_perf_symbol_entry", "symbol", "entry_date"),
    )
