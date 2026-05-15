from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from sentinelpay.config import ModelConfig
from sentinelpay.data import dataset_summary, load_transactions
from sentinelpay.evaluation import evaluate_predictions
from sentinelpay.features import split_and_scale
from sentinelpay.model import predict_with_threshold, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SentinelPay fraud risk model.")
    parser.add_argument("--output", default="artifacts/sentinelpay_model.joblib", help="Path to save the trained model.")
    parser.add_argument("--threshold", type=float, default=0.30, help="Decision threshold used for evaluation.")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE during training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(threshold=args.threshold, use_smote=not args.no_smote)

    df = load_transactions()
    summary = dataset_summary(df)
    print(f"Loaded {summary['total_transactions']:,} transactions; fraud rate: {summary['fraud_rate']:.3%}")

    X_train, X_test, y_train, y_test = split_and_scale(df, test_size=config.test_size)
    model = train_classifier(X_train, y_train, config)
    y_prob, y_pred = predict_with_threshold(model, X_test, threshold=config.threshold)
    metrics = evaluate_predictions(y_test, y_prob, y_pred)

    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(metrics["classification_report"].round(3))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": config, "feature_names": list(X_train.columns)}, output)
    print(f"Saved model artifact to {output}")


if __name__ == "__main__":
    main()
