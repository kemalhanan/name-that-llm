"""Predict the model for each test text and write a submission in the sample format.

Predictions are class-balanced with Sinkhorn over the whole test set (see postprocess.py);
``--balance-strength 0`` turns that off.

    uv run python -m name_that_llm.modeling.predict
"""

from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from name_that_llm.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from name_that_llm.modeling.postprocess import sinkhorn, softmax

app = typer.Typer()


def predict_balanced(model, texts, temperature: float, strength: float) -> np.ndarray:
    """Model probabilities -> Sinkhorn balancing -> label with the highest probability."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
    else:  # LinearSVC pipelines only give decision scores
        proba = softmax(model.decision_function(texts), temperature)
    return np.asarray(model.classes_)[sinkhorn(proba, strength).argmax(axis=1)]


@app.command()
def main(
    test_path: Path = RAW_DATA_DIR / "test.csv",
    sample_path: Path = RAW_DATA_DIR / "sample_submission.parquet",
    model_path: Path = MODELS_DIR / "pool2000.joblib",
    submission_path: Path = PROCESSED_DATA_DIR / "submission.parquet",
    balance_temperature: float = 1.0,
    balance_strength: float = 0.3,
):
    test = pd.read_csv(test_path)
    sample = pd.read_parquet(sample_path)
    model = joblib.load(model_path)

    labels = predict_balanced(model, test.text, balance_temperature, balance_strength)
    pred = pd.DataFrame({"id": test.id, "model": labels})
    sub = sample[["id"]].merge(pred, on="id", how="left")  # keep the sample's row order
    if sub.model.isna().any():
        raise ValueError(f"{sub.model.isna().sum()} sample ids have no prediction")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(submission_path, index=False)
    sub.to_csv(submission_path.with_suffix(".csv"), index=False)
    logger.success(f"Wrote {len(sub)} predictions to {submission_path}")
    logger.info(f"\n{sub.to_string(index=False)}")


if __name__ == "__main__":
    app()
