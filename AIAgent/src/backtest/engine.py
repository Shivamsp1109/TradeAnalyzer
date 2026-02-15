from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.common.config import AppConfig, get_config
from src.decision.engine import DecisionEngine
from src.model.contracts import InferenceResult

from .contracts import BacktestInputs, BacktestRunResult
from .metrics import cagr, max_drawdown, sharpe_ratio, win_rate


class WalkForwardBacktester:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.decision_engine = DecisionEngine(self.config)

    def run(self, inputs: BacktestInputs) -> BacktestRunResult:
        df = inputs.data.copy()
        if df.empty:
            raise ValueError("Backtest input dataset is empty")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

        unique_dates = sorted(df["date"].unique())
        min_train = self.config.backtest.min_training_days
        step = self.config.backtest.walk_forward_step_days

        max_horizon = max(self.config.training.prediction_horizons_days)
        if len(unique_dates) <= (min_train + max_horizon):
            raise ValueError("Insufficient data for walk-forward backtest")

        trade_rows: list[dict[str, Any]] = []
        window_rows: list[dict[str, Any]] = []

        start_idx = min_train
        end_limit = len(unique_dates) - max_horizon

        while start_idx < end_limit:
            train_dates = unique_dates[max(0, start_idx - self.config.training.train_window_days):start_idx]
            test_dates = unique_dates[start_idx:min(start_idx + step, end_limit)]

            if not train_dates or not test_dates:
                break

            train_df = df[df["date"].isin(train_dates)].copy()
            test_df = df[df["date"].isin(test_dates)].copy()

            models = self._fit_models(train_df, inputs.feature_columns)
            window_trades = self._simulate_window(
                test_df=test_df,
                feature_columns=inputs.feature_columns,
                models=models,
            )
            trade_rows.extend(window_trades)
            window_rows.append(
                {
                    "train_start": min(train_dates),
                    "train_end": max(train_dates),
                    "test_start": min(test_dates),
                    "test_end": max(test_dates),
                    "trades": len(window_trades),
                }
            )

            start_idx += step

        trades_df = pd.DataFrame(trade_rows)
        windows_df = pd.DataFrame(window_rows)
        daily_df = self._build_daily_performance(trades_df, inputs.benchmark_prices)

        strategy_returns = daily_df["strategy_return"] if not daily_df.empty else pd.Series(dtype=float)
        benchmark_returns = daily_df["benchmark_return"] if not daily_df.empty else pd.Series(dtype=float)

        summary = {
            "cagr": cagr(strategy_returns),
            "sharpe": sharpe_ratio(strategy_returns, self.config.backtest.risk_free_rate_annual),
            "max_drawdown": max_drawdown(strategy_returns),
            "win_rate": win_rate(trades_df.get("realized_return", pd.Series(dtype=float))),
            "benchmark_cagr": cagr(benchmark_returns),
            "benchmark_sharpe": sharpe_ratio(benchmark_returns, self.config.backtest.risk_free_rate_annual),
            "total_trades": int(len(trades_df)),
            "buy_trades": int((trades_df.get("decision", pd.Series(dtype=str)) == "BUY").sum()) if not trades_df.empty else 0,
            "walk_forward_windows": int(len(windows_df)),
        }

        period_start = min(unique_dates)
        period_end = max(unique_dates)

        return BacktestRunResult(
            strategy_name=inputs.strategy_name,
            universe_name=inputs.universe_name,
            period_start=period_start,
            period_end=period_end,
            benchmark_symbol=self.config.backtest.benchmark_symbol,
            summary=summary,
            daily_performance=daily_df,
            trades=trades_df,
            windows=windows_df,
        )

    def _fit_models(self, train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        X = train_df[feature_columns].to_numpy(dtype=float)

        reg30 = self._build_regressor()
        reg60 = self._build_regressor()
        reg90 = self._build_regressor()

        reg30.fit(X, train_df["target_ret_30d"].to_numpy(dtype=float))
        reg60.fit(X, train_df["target_ret_60d"].to_numpy(dtype=float))
        reg90.fit(X, train_df["target_ret_90d"].to_numpy(dtype=float))

        y_cls = train_df["target_prob_gt_8pct"].to_numpy(dtype=int)
        if len(np.unique(y_cls)) < 2:
            # Degenerate case: constant probability from class frequency.
            p = float(np.mean(y_cls)) if len(y_cls) else 0.0
            classifier = ("constant", p)
        else:
            classifier = self._build_classifier()
            classifier.fit(X, y_cls)

        return {"reg30": reg30, "reg60": reg60, "reg90": reg90, "clf": classifier}

    def _simulate_window(
        self,
        test_df: pd.DataFrame,
        feature_columns: list[str],
        models: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for _, row in test_df.iterrows():
            X = row[feature_columns].to_numpy(dtype=float).reshape(1, -1)
            pred30 = float(models["reg30"].predict(X)[0])
            pred60 = float(models["reg60"].predict(X)[0])
            pred90 = float(models["reg90"].predict(X)[0])

            clf = models["clf"]
            if isinstance(clf, tuple) and clf[0] == "constant":
                prob = float(clf[1])
            else:
                prob = float(clf.predict_proba(X)[:, 1][0])

            confidence = self._model_confidence(pred30, pred60, pred90, prob)
            inference = InferenceResult(
                symbol=str(row["symbol"]),
                as_of_date=row["date"],
                expected_return_30d=pred30,
                expected_return_60d=pred60,
                expected_return_90d=pred90,
                probability_return_gt_8pct=prob,
                model_confidence=confidence,
                feature_snapshot={col: self._safe_float(row[col]) for col in feature_columns},
            )
            decision = self.decision_engine.evaluate(inference=inference, latest_row=row)

            realized_return = self._realized_return_for_horizon(row, decision.suggested_horizon_days)
            rows.append(
                {
                    "date": row["date"],
                    "symbol": row["symbol"],
                    "decision": decision.decision,
                    "selected_horizon": decision.suggested_horizon_days,
                    "pred_ret_30d": pred30,
                    "pred_ret_60d": pred60,
                    "pred_ret_90d": pred90,
                    "prob_gt_8pct": prob,
                    "model_confidence": confidence,
                    "entry_price": decision.entry_target.entry_price,
                    "target_price": decision.entry_target.target_price,
                    "realized_return": realized_return,
                    "rule_checks": decision.rule_checks,
                    "risk_adjusted_returns": decision.risk_adjusted_returns,
                }
            )

        return rows

    def _build_daily_performance(self, trades_df: pd.DataFrame, benchmark_prices: pd.DataFrame | None) -> pd.DataFrame:
        if trades_df.empty:
            return pd.DataFrame(columns=["date", "strategy_return", "benchmark_return"])

        buys = trades_df[trades_df["decision"] == "BUY"].copy()
        if buys.empty:
            strategy_daily = (
                trades_df[["date"]]
                .drop_duplicates()
                .assign(strategy_return=0.0)
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            strategy_daily = (
                buys.groupby("date", as_index=False)["realized_return"]
                .mean()
                .rename(columns={"realized_return": "strategy_return"})
            )

        benchmark_daily = self._benchmark_daily_returns(benchmark_prices)

        daily = strategy_daily.merge(benchmark_daily, on="date", how="left")
        daily["benchmark_return"] = daily["benchmark_return"].fillna(0.0)
        daily = daily.sort_values("date").reset_index(drop=True)
        return daily

    def _benchmark_daily_returns(self, benchmark_prices: pd.DataFrame | None) -> pd.DataFrame:
        if benchmark_prices is None or benchmark_prices.empty:
            return pd.DataFrame(columns=["date", "benchmark_return"])

        b = benchmark_prices.copy()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        b = b.sort_values("date")
        b["benchmark_return"] = b["close"].pct_change().fillna(0.0)
        return b[["date", "benchmark_return"]]

    def _realized_return_for_horizon(self, row: pd.Series, horizon: int) -> float:
        col = f"target_ret_{horizon}d"
        value = row.get(col)
        return self._safe_float(value)

    def _build_regressor(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=self.config.model.n_estimators,
            learning_rate=self.config.model.learning_rate,
            max_depth=self.config.model.max_depth,
            subsample=self.config.model.subsample,
            colsample_bytree=self.config.model.colsample_bytree,
            random_state=self.config.model.random_state,
            objective="reg:squarederror",
            n_jobs=4,
        )

    def _build_classifier(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.config.model.n_estimators,
            learning_rate=self.config.model.learning_rate,
            max_depth=self.config.model.max_depth,
            subsample=self.config.model.subsample,
            colsample_bytree=self.config.model.colsample_bytree,
            random_state=self.config.model.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
        )

    @staticmethod
    def _model_confidence(r30: float, r60: float, r90: float, prob: float) -> float:
        trend = np.array([r30, r60, r90], dtype=float)
        spread = float(np.std(trend))
        prob_strength = abs(prob - 0.5) * 2.0
        # Higher confidence with stable horizon trend and stronger class probability.
        score = (1.0 / (1.0 + spread * 10.0)) * 0.6 + prob_strength * 0.4
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if value is None or pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
