import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    debug: bool


@dataclass(frozen=True)
class DatabaseConfig:
    url: str
    echo_sql: bool
    pool_size: int
    max_overflow: int


@dataclass(frozen=True)
class UniverseFilterConfig:
    min_close_price: float
    min_market_cap_inr: float
    min_median_daily_turnover_inr: float


@dataclass(frozen=True)
class DecisionConfig:
    min_expected_60d_return: float
    min_probability: float
    max_volatility: float
    confidence_floor: float
    ma20_pullback_tolerance: float


@dataclass(frozen=True)
class TrainingConfig:
    retrain_day_utc: str
    train_window_days: int
    prediction_horizons_days: List[int]
    probability_threshold_return: float


@dataclass(frozen=True)
class BacktestConfig:
    walk_forward_step_days: int
    min_training_days: int
    benchmark_symbol: str
    risk_free_rate_annual: float


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    feature_set_version: str
    artifacts_dir: str
    random_state: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float


@dataclass(frozen=True)
class AppConfig:
    env: str
    api: ApiConfig
    db: DatabaseConfig
    universe: UniverseFilterConfig
    decision: DecisionConfig
    training: TrainingConfig
    backtest: BacktestConfig
    model: ModelConfig


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    horizons = [int(value) for value in _get_env_list("PREDICTION_HORIZONS_DAYS", ["30", "60", "90"])]
    return AppConfig(
        env=_get_env_str("APP_ENV", "dev"),
        api=ApiConfig(
            host=_get_env_str("API_HOST", "0.0.0.0"),
            port=_get_env_int("API_PORT", 8000),
            debug=_get_env_bool("API_DEBUG", False),
        ),
        db=DatabaseConfig(
            url=_get_env_str(
                "DATABASE_URL",
                "postgresql+psycopg2://postgres:postgres@localhost:5432/aiagent",
            ),
            echo_sql=_get_env_bool("DB_ECHO_SQL", False),
            pool_size=_get_env_int("DB_POOL_SIZE", 5),
            max_overflow=_get_env_int("DB_MAX_OVERFLOW", 10),
        ),
        universe=UniverseFilterConfig(
            min_close_price=_get_env_float("MIN_CLOSE_PRICE", 20.0),
            min_market_cap_inr=_get_env_float("MIN_MARKET_CAP_INR", 5_000_000_000.0),
            min_median_daily_turnover_inr=_get_env_float(
                "MIN_MEDIAN_DAILY_TURNOVER_INR", 50_000_000.0
            ),
        ),
        decision=DecisionConfig(
            min_expected_60d_return=_get_env_float("MIN_EXPECTED_60D_RETURN", 0.10),
            min_probability=_get_env_float("MIN_PROBABILITY", 0.60),
            max_volatility=_get_env_float("MAX_VOLATILITY", 0.35),
            confidence_floor=_get_env_float("MODEL_CONFIDENCE_FLOOR", 0.55),
            ma20_pullback_tolerance=_get_env_float("MA20_PULLBACK_TOLERANCE", 0.02),
        ),
        training=TrainingConfig(
            retrain_day_utc=_get_env_str("RETRAIN_DAY_UTC", "SUN"),
            train_window_days=_get_env_int("TRAIN_WINDOW_DAYS", 720),
            prediction_horizons_days=horizons,
            probability_threshold_return=_get_env_float(
                "PROBABILITY_THRESHOLD_RETURN", 0.08
            ),
        ),
        backtest=BacktestConfig(
            walk_forward_step_days=_get_env_int("WALK_FORWARD_STEP_DAYS", 20),
            min_training_days=_get_env_int("MIN_TRAINING_DAYS", 252),
            benchmark_symbol=_get_env_str("BENCHMARK_SYMBOL", "^NSEI"),
            risk_free_rate_annual=_get_env_float("RISK_FREE_RATE_ANNUAL", 0.07),
        ),
        model=ModelConfig(
            model_name=_get_env_str("MODEL_NAME", "xgboost_multihorizon"),
            feature_set_version=_get_env_str("FEATURE_SET_VERSION", "v1"),
            artifacts_dir=_get_env_str("MODEL_ARTIFACTS_DIR", "artifacts/models"),
            random_state=_get_env_int("MODEL_RANDOM_STATE", 42),
            n_estimators=_get_env_int("MODEL_N_ESTIMATORS", 400),
            learning_rate=_get_env_float("MODEL_LEARNING_RATE", 0.03),
            max_depth=_get_env_int("MODEL_MAX_DEPTH", 4),
            subsample=_get_env_float("MODEL_SUBSAMPLE", 0.9),
            colsample_bytree=_get_env_float("MODEL_COLSAMPLE_BYTREE", 0.8),
        ),
    )
