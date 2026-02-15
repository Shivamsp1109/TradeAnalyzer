from __future__ import annotations

import pandas as pd

from src.features.contracts import FeatureBuildResult
from src.features.indicators import macd, rsi


class FeatureEngineeringService:
    def build_features(self, ohlcv: pd.DataFrame, fundamentals: pd.DataFrame) -> FeatureBuildResult:
        if ohlcv.empty:
            return FeatureBuildResult(features=pd.DataFrame())

        df = ohlcv.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        parts: list[pd.DataFrame] = []
        for _, sdf in df.groupby("symbol", sort=False):
            parts.append(self._build_symbol_features(sdf))
        features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        merged = self._merge_fundamentals(features, fundamentals)
        return FeatureBuildResult(features=merged)

    def _build_symbol_features(self, sdf: pd.DataFrame) -> pd.DataFrame:
        out = sdf.copy()

        out["ma_20"] = out["close"].rolling(20, min_periods=20).mean()
        out["ma_50"] = out["close"].rolling(50, min_periods=50).mean()
        out["ma_200"] = out["close"].rolling(200, min_periods=200).mean()

        out["rsi_14"] = rsi(out["close"], period=14)

        macd_df = macd(out["close"])
        out["macd"] = macd_df["macd"].values
        out["macd_signal"] = macd_df["macd_signal"].values
        out["macd_hist"] = macd_df["macd_hist"].values

        out["return_3m"] = out["close"].pct_change(periods=63)
        out["return_6m"] = out["close"].pct_change(periods=126)

        daily_ret = out["close"].pct_change()
        out["volatility_63d"] = daily_ret.rolling(63, min_periods=20).std() * (252**0.5)

        out["price_vs_ma20"] = (out["close"] / out["ma_20"]) - 1
        out["price_vs_ma50"] = (out["close"] / out["ma_50"]) - 1
        out["price_vs_ma200"] = (out["close"] / out["ma_200"]) - 1

        return out

    def _merge_fundamentals(self, features: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
        if fundamentals.empty:
            for col in [
                "pe_ratio",
                "roe",
                "revenue_growth",
                "eps_growth",
                "debt_to_equity",
                "free_cash_flow",
                "market_cap",
            ]:
                features[col] = pd.NA
            return features

        fcols = [
            "symbol",
            "pe_ratio",
            "roe",
            "revenue_growth",
            "eps_growth",
            "debt_to_equity",
            "free_cash_flow",
            "market_cap",
        ]

        fdf = fundamentals[fcols].drop_duplicates(subset=["symbol"])
        merged = features.merge(fdf, on="symbol", how="left")
        return merged
