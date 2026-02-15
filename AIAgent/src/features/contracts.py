from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureBuildResult:
    features: pd.DataFrame


@dataclass(frozen=True)
class LabeledDataset:
    data: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
