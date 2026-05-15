from __future__ import annotations

import pandas as pd

from sentinelpay.config import DATASET_URL, TARGET


def load_transactions(source: str = DATASET_URL) -> pd.DataFrame:
    """Load the credit card transactions dataset."""
    df = pd.read_csv(source)
    required = {"Time", "Amount", TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def dataset_summary(df: pd.DataFrame) -> dict[str, float | int]:
    fraud_count = int(df[TARGET].sum())
    total = int(len(df))
    return {
        "total_transactions": total,
        "fraud_transactions": fraud_count,
        "legitimate_transactions": total - fraud_count,
        "fraud_rate": float(df[TARGET].mean()),
    }
