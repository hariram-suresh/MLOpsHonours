"""Create or split the Iris dataset for the DVC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_params(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_dataset(output_path: Path) -> None:
    """Export sklearn's bundled Iris dataset as a versionable CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    dataset = iris.frame.rename(columns={"target": "target"})
    dataset.to_csv(output_path, index=False)
    print(f"Wrote {len(dataset)} rows to {output_path.relative_to(ROOT)}")


def prepare_data(params_path: Path) -> None:
    params = load_params(params_path)
    dataset_path = DATA_DIR / "iris.csv"
    if not dataset_path.exists():
        generate_dataset(dataset_path)

    dataset = pd.read_csv(dataset_path)
    features = dataset.drop(columns="target")
    labels = dataset["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=labels,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([x_train, y_train], axis=1).to_csv(DATA_DIR / "train.csv", index=False)
    pd.concat([x_test, y_test], axis=1).to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Prepared {len(x_train)} training and {len(x_test)} test rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Export the source CSV without creating train/test splits.",
    )
    args = parser.parse_args()
    if args.generate_only:
        generate_dataset(DATA_DIR / "iris.csv")
    else:
        prepare_data(args.params)


if __name__ == "__main__":
    main()
