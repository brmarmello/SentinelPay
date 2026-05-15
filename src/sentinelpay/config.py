from __future__ import annotations

from dataclasses import dataclass


DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
TARGET = "Class"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.30


@dataclass(frozen=True)
class ModelConfig:
    test_size: float = 0.30
    threshold: float = DEFAULT_THRESHOLD
    use_smote: bool = True
    scale_pos_weight: float = 10.0
    n_estimators: int = 250
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.90
    colsample_bytree: float = 0.90

