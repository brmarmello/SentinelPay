from __future__ import annotations

import pandas as pd

from sentinelpay.data import dataset_summary
from sentinelpay.features import build_feature_frame


def test_build_feature_frame_adds_amount_log_and_removes_amount() -> None:
    df = pd.DataFrame(
        {
            "Time": [1, 2],
            "V1": [0.1, -0.2],
            "Amount": [10.0, 20.0],
            "Class": [0, 1],
        }
    )

    X, y = build_feature_frame(df)

    assert "Amount_log" in X.columns
    assert "Amount" not in X.columns
    assert y.tolist() == [0, 1]


def test_dataset_summary_counts_fraud_rate() -> None:
    df = pd.DataFrame({"Class": [0, 0, 1, 0]})

    summary = dataset_summary(df)

    assert summary["total_transactions"] == 4
    assert summary["fraud_transactions"] == 1
    assert summary["fraud_rate"] == 0.25
