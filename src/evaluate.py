"""Evaluate a trained classifier and save metrics/predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    with args.params.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)

    test_data = pd.read_csv(ROOT / "data" / "test.csv")
    x_test = test_data.drop(columns="target")
    y_test = test_data["target"]
    model = joblib.load(ROOT / "model" / "model.joblib")
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "meets_accuracy_threshold": bool(
            accuracy_score(y_test, predictions) >= params["evaluation"]["accuracy_threshold"]
        ),
    }
    metrics_path = ROOT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    prediction_frame = x_test.copy()
    prediction_frame["actual"] = y_test.to_numpy()
    prediction_frame["predicted"] = predictions
    prediction_frame.to_csv(ROOT / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
