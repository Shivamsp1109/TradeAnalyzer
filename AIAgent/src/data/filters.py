from __future__ import annotations

import pandas as pd

from src.common.config import UniverseFilterConfig


class UniverseFilterEngine:
    def __init__(self, config: UniverseFilterConfig) -> None:
        self.config = config

    def build_filter_report(self, price_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        if price_df.empty:
            return pd.DataFrame(columns=["symbol", "eligible", "reasons"])

        latest = (
            price_df.sort_values("date")
            .groupby("symbol", as_index=False)
            .tail(1)[["symbol", "date", "close"]]
            .rename(columns={"close": "latest_close"})
        )

        turnover_stats = (
            price_df.groupby("symbol", as_index=False)["turnover"]
            .median()
            .rename(columns={"turnover": "median_turnover"})
        )

        fcols = ["symbol", "market_cap", "pe_ratio", "revenue_growth", "eps_growth", "free_cash_flow"]
        fdf = fundamentals_df[fcols].drop_duplicates(subset=["symbol"]) if not fundamentals_df.empty else pd.DataFrame(columns=fcols)

        report = latest.merge(turnover_stats, on="symbol", how="left").merge(fdf, on="symbol", how="left")
        report["eligible"] = True
        report["reasons"] = ""

        report = self._apply_close_price_filter(report)
        report = self._apply_liquidity_filter(report)
        report = self._apply_market_cap_filter(report)

        return report[["symbol", "eligible", "reasons", "latest_close", "median_turnover", "market_cap"]]

    def _apply_close_price_filter(self, report: pd.DataFrame) -> pd.DataFrame:
        mask = report["latest_close"].fillna(0) < self.config.min_close_price
        report.loc[mask, "eligible"] = False
        report.loc[mask, "reasons"] = report.loc[mask, "reasons"] + "below_min_price;"
        return report

    def _apply_liquidity_filter(self, report: pd.DataFrame) -> pd.DataFrame:
        mask = report["median_turnover"].fillna(0) < self.config.min_median_daily_turnover_inr
        report.loc[mask, "eligible"] = False
        report.loc[mask, "reasons"] = report.loc[mask, "reasons"] + "below_min_turnover;"
        return report

    def _apply_market_cap_filter(self, report: pd.DataFrame) -> pd.DataFrame:
        mask = report["market_cap"].fillna(0) < self.config.min_market_cap_inr
        report.loc[mask, "eligible"] = False
        report.loc[mask, "reasons"] = report.loc[mask, "reasons"] + "below_min_market_cap;"
        return report
