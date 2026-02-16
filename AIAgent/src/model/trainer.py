from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier, XGBRegressor

from src.common.config import AppConfig, get_config
from src.features.contracts import LabeledDataset
from src.model.contracts import ModelArtifacts, TrainingResult
from src.model.metrics import classification_metrics, regression_metrics


class XGBoostTrainingService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def train(self, dataset: LabeledDataset) -> TrainingResult:
        if dataset.data.empty:
            raise ValueError("Cannot train model: empty dataset")

        df = dataset.data.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
        train_df, val_df = self._time_split(df)

        feature_cols = dataset.feature_columns
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        X_val = val_df[feature_cols].to_numpy(dtype=float)

        reg_targets = {30: "target_ret_30d", 60: "target_ret_60d", 90: "target_ret_90d"}
        regressors: dict[int, XGBRegressor] = {}
        reg_metrics: dict[str, Any] = {}

        for horizon, tcol in reg_targets.items():
            model = self._build_regressor()
            y_train = train_df[tcol].to_numpy(dtype=float)
            y_val = val_df[tcol].to_numpy(dtype=float)
            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)
            regressors[horizon] = model
            reg_metrics[f"ret_{horizon}d"] = regression_metrics(y_val, pred_val)

        cls_target = "target_prob_gt_8pct"
        y_train_cls = train_df[cls_target].to_numpy(dtype=int)
        y_val_cls = val_df[cls_target].to_numpy(dtype=int)

        classifier, classifier_name = self._build_classifier(y_train_cls)
        classifier.fit(X_train, y_train_cls)

        calibrated = self._calibrate_classifier(classifier, X_val, y_val_cls)
        prob_val = calibrated.predict_proba(X_val)[:, 1]
        cls_metrics = classification_metrics(y_val_cls, prob_val)

        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifacts = self._persist_artifacts(
            model_version=model_version,
            regressors=regressors,
            classifier=calibrated,
            feature_columns=feature_cols,
            train_df=train_df,
            val_df=val_df,
            metrics={"regression": reg_metrics, "classification": cls_metrics, "classifier": classifier_name},
        )

        return TrainingResult(
            model_name=self.config.model.model_name,
            model_version=model_version,
            trained_until=pd.to_datetime(df["date"]).dt.date.max(),
            training_start=pd.to_datetime(train_df["date"]).dt.date.min(),
            training_end=pd.to_datetime(train_df["date"]).dt.date.max(),
            feature_set_version=self.config.model.feature_set_version,
            metrics={"regression": reg_metrics, "classification": cls_metrics, "classifier": classifier_name},
            hyperparams=self._hyperparams_dict(),
            artifact_uri=str(artifacts.model_dir),
            artifacts=artifacts,
        )

    def _time_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
        if len(unique_dates) < 30:
            raise ValueError("Insufficient date points for train/validation split")

        split_idx = int(len(unique_dates) * 0.8)
        split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
        split_date = unique_dates[split_idx]

        train_df = df[pd.to_datetime(df["date"]).dt.date < split_date].copy()
        val_df = df[pd.to_datetime(df["date"]).dt.date >= split_date].copy()

        if train_df.empty or val_df.empty:
            raise ValueError("Time split failed: empty train/validation partition")
        return train_df, val_df

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

    def _build_classifier(self, y_train: np.ndarray) -> tuple[Any, str]:
        if len(np.unique(y_train)) < 2:
            return DummyClassifier(strategy="most_frequent"), "dummy_most_frequent"

        clf = XGBClassifier(
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
        return clf, "xgboost_classifier"

    def _calibrate_classifier(self, classifier: Any, X_val: np.ndarray, y_val: np.ndarray) -> Any:
        if len(np.unique(y_val)) < 2:
            return classifier

        calibrated = CalibratedClassifierCV(estimator=classifier, method="sigmoid", cv="prefit")
        calibrated.fit(X_val, y_val)
        return calibrated

    def _hyperparams_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.config.model.n_estimators,
            "learning_rate": self.config.model.learning_rate,
            "max_depth": self.config.model.max_depth,
            "subsample": self.config.model.subsample,
            "colsample_bytree": self.config.model.colsample_bytree,
            "random_state": self.config.model.random_state,
        }

    def _persist_artifacts(
        self,
        model_version: str,
        regressors: dict[int, XGBRegressor],
        classifier: Any,
        feature_columns: list[str],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        metrics: dict[str, Any],
    ) -> ModelArtifacts:
        model_dir = Path(self.config.model.artifacts_dir) / model_version
        model_dir.mkdir(parents=True, exist_ok=True)

        regressor_paths: dict[int, Path] = {}
        for horizon, model in regressors.items():
            path = model_dir / f"regressor_{horizon}d.pkl"
            with path.open("wb") as f:
                pickle.dump(model, f)
            regressor_paths[horizon] = path

        classifier_path = model_dir / "classifier_prob_gt_8pct.pkl"
        with classifier_path.open("wb") as f:
            pickle.dump(classifier, f)

        metadata_path = model_dir / "metadata.json"
        payload = {
            "model_name": self.config.model.model_name,
            "model_version": model_version,
            "feature_set_version": self.config.model.feature_set_version,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "training_period": {
                "start": str(pd.to_datetime(train_df["date"]).dt.date.min()),
                "end": str(pd.to_datetime(train_df["date"]).dt.date.max()),
            },
            "validation_period": {
                "start": str(pd.to_datetime(val_df["date"]).dt.date.min()),
                "end": str(pd.to_datetime(val_df["date"]).dt.date.max()),
            },
            "generated_at_utc": datetime.utcnow().isoformat(),
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return ModelArtifacts(
            model_version=model_version,
            model_dir=model_dir,
            regressor_paths=regressor_paths,
            classifier_path=classifier_path,
            metadata_path=metadata_path,
        )


