from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sentinelpay.config import RANDOM_STATE, TARGET


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model features while keeping the target separate."""
    work = df.copy()
    work["Amount_log"] = np.log1p(work["Amount"])
    X = work.drop(columns=[TARGET, "Amount"])
    y = work[TARGET].astype(int)
    return X, y


def split_and_scale(
    df: pd.DataFrame,
    test_size: float,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = build_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Amount_log"] = scaler.fit_transform(X_train[["Amount_log"]])
    X_test["Amount_log"] = scaler.transform(X_test[["Amount_log"]])
    return X_train, X_test, y_train, y_test
