CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(64) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    feature_set_version VARCHAR(64) NOT NULL,
    training_start DATE NOT NULL,
    training_end DATE NOT NULL,
    trained_until DATE NOT NULL,
    hyperparams JSONB NOT NULL,
    metrics JSONB NOT NULL,
    artifact_uri VARCHAR(512) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_name_version UNIQUE (model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_active
    ON model_versions (model_name, is_active);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    as_of_date DATE NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    model_version_id UUID NOT NULL REFERENCES model_versions(id) ON DELETE RESTRICT,
    expected_return_30d DOUBLE PRECISION NOT NULL,
    expected_return_60d DOUBLE PRECISION NOT NULL,
    expected_return_90d DOUBLE PRECISION NOT NULL,
    probability_return_gt_8pct DOUBLE PRECISION NOT NULL CHECK (probability_return_gt_8pct >= 0 AND probability_return_gt_8pct <= 1),
    model_confidence DOUBLE PRECISION NOT NULL CHECK (model_confidence >= 0 AND model_confidence <= 1),
    feature_snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_predictions_date_symbol_model UNIQUE (as_of_date, symbol, model_version_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date
    ON predictions (symbol, as_of_date);

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    decision VARCHAR(16) NOT NULL CHECK (decision IN ('BUY', 'DO_NOT_BUY')),
    suggested_horizon_days INT NOT NULL CHECK (suggested_horizon_days IN (30, 60, 90)),
    entry_price DOUBLE PRECISION NOT NULL,
    target_price DOUBLE PRECISION NOT NULL,
    entry_logic VARCHAR(64) NOT NULL,
    explanation TEXT NOT NULL,
    risk_checks JSONB NOT NULL,
    disclaimers JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_signals_date_symbol UNIQUE (as_of_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_date
    ON signals (symbol, as_of_date);

CREATE TABLE IF NOT EXISTS backtest_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(64) NOT NULL,
    universe_name VARCHAR(64) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    benchmark_symbol VARCHAR(32) NOT NULL,
    cagr DOUBLE PRECISION NOT NULL,
    sharpe DOUBLE PRECISION NOT NULL,
    max_drawdown DOUBLE PRECISION NOT NULL,
    win_rate DOUBLE PRECISION NOT NULL,
    benchmark_cagr DOUBLE PRECISION NOT NULL,
    benchmark_sharpe DOUBLE PRECISION,
    config_snapshot JSONB NOT NULL,
    metrics_snapshot JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_period
    ON backtest_reports (period_start, period_end);

CREATE TABLE IF NOT EXISTS live_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL,
    horizon_days INT NOT NULL CHECK (horizon_days IN (30, 60, 90)),
    entry_date DATE NOT NULL,
    exit_date DATE,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    realized_return DOUBLE PRECISION,
    benchmark_return DOUBLE PRECISION,
    status VARCHAR(16) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_perf_symbol_entry
    ON live_performance (symbol, entry_date);
