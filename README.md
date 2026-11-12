# Reproducible Iris ML Experiment — Git, DVC & CML

This project trains a Logistic Regression classifier on scikit-learn's bundled Iris dataset. The dataset source is deterministic and can be recreated locally; the CSV is tracked by DVC. The three pipeline stages split data, train the model, and evaluate it.

## Quick start

1. Create/activate a Python 3.10+ environment and install `requirements.txt`.
2. Generate the raw source CSV once: `python src/prepare.py --generate-only`.
3. Run the experiment: `dvc repro`.
4. Review `metrics.json`; `accuracy`, weighted F1, and macro F1 are reported.

The local DVC remote is configured at `../dvc-storage` for this classroom demonstration. It is intentionally not a hosted service. For team use, change the remote to a shared S3, Azure Blob, Google Drive, SSH, or other supported DVC remote and configure credentials outside Git.

## Reproduction and comparison

Commit the initial `params.yaml` and the DVC metadata/metrics baseline first. Change a value such as `model.C` or `data.test_size`, run `dvc repro`, then compare the working tree against `HEAD` with `dvc metrics diff` and `dvc params diff`. To restore the baseline, use `git checkout -- params.yaml` followed by `dvc repro`.

## GitHub Actions + CML

`.github/workflows/cml.yaml` runs on pull requests. As Iris is bundled in scikit-learn, CI regenerates the source CSV and runs `dvc repro`; this keeps the example workflow independent of the classroom's local DVC remote. The CML comment includes the changed metrics and a short report. Configure repository Actions permissions to allow the workflow to write pull-request comments.
