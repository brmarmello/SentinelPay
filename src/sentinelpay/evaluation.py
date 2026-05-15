from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate_predictions(y_true: pd.Series, y_prob: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "classification_report": pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def precision_recall_points(y_true: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return pd.DataFrame(
        {
            "threshold": list(thresholds) + [1.0],
            "precision": precision,
            "recall": recall,
        }
    )

