from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from sentinelpay.config import ModelConfig, RANDOM_STATE


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
) -> XGBClassifier:
    if config.use_smote:
        sampler = SMOTE(random_state=RANDOM_STATE)
        X_fit, y_fit = sampler.fit_resample(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=config.scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)
    return model


def predict_with_threshold(model: XGBClassifier, X: pd.DataFrame, threshold: float) -> tuple[pd.Series, pd.Series]:
    probabilities = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="fraud_probability")
    predictions = (probabilities >= threshold).astype(int).rename("prediction")
    return probabilities, predictions
