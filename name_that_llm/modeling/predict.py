"""Predict the model for each test text and write a submission in the sample format.

uv run python -m name_that_llm.modeling.predict
"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from name_that_llm.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    test_path: Path = RAW_DATA_DIR / "test.csv",
    sample_path: Path = RAW_DATA_DIR / "sample_submission.parquet",
    model_path: Path = MODELS_DIR / "blend.joblib",
    submission_path: Path = PROCESSED_DATA_DIR / "submission.parquet",
):
    test = pd.read_csv(test_path)
    sample = pd.read_parquet(sample_path)
    model = joblib.load(model_path)

    pred = pd.DataFrame({"id": test.id, "model": model.predict(test.text)})
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
