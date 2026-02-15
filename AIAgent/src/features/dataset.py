from __future__ import annotations

import pandas as pd

from src.common.config import TrainingConfig
from src.features.contracts import LabeledDataset
from src.features.engineering import FeatureEngineeringService
from src.features.targets import add_probability_label, add_return_targets


class TrainingDatasetBuilder:
    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config
        self.feature_service = FeatureEngineeringService()

    def build(self, ohlcv: pd.DataFrame, fundamentals: pd.DataFrame) -> LabeledDataset:
        feature_result = self.feature_service.build_features(ohlcv=ohlcv, fundamentals=fundamentals)
        df = feature_result.features

        df = add_return_targets(df, horizons=self.training_config.prediction_horizons_days)
        df = add_probability_label(
            df,
            threshold_return=self.training_config.probability_threshold_return,
            label_horizon=60,
        )

        target_columns = [
            f"target_ret_{h}d" for h in self.training_config.prediction_horizons_days
        ] + ["target_prob_gt_8pct"]

        # Drop rows where no training target is available (tail rows by horizon).
        df = df.dropna(subset=target_columns)

        # Conservative NA handling for numeric model inputs.
        df = self._impute_numeric(df)

        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "ma_20",
            "ma_50",
            "ma_200",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "return_3m",
            "return_6m",
            "volatility_63d",
            "price_vs_ma20",
            "price_vs_ma50",
            "price_vs_ma200",
            "pe_ratio",
            "roe",
            "revenue_growth",
            "eps_growth",
            "debt_to_equity",
            "free_cash_flow",
            "market_cap",
        ]

        available_features = [col for col in feature_columns if col in df.columns]
        return LabeledDataset(data=df, feature_columns=available_features, target_columns=target_columns)

    def _impute_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            median = out[col].median()
            out[col] = out[col].fillna(median)
        return out
