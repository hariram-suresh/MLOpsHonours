"""Train a parameterized classifier and save it for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_model(model_params: dict):
    model_type = model_params["type"].lower()
    random_state = model_params["random_state"]
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=model_params["C"],
            max_iter=model_params["max_iter"],
            random_state=random_state,
        )
    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=model_params["max_depth"],
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    training_data = pd.read_csv(ROOT / "data" / "train.csv")
    x_train = training_data.drop(columns="target")
    y_train = training_data["target"]

    model = build_model(params["model"])
    model.fit(x_train, y_train)

    model_path = ROOT / "model" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved {params['model']['type']} model to {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
