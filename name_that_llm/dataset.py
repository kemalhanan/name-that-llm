"""External training data: a style-matched sample of the community Open Pool.

Source: https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool (v12, CC BY / CC0 rows).
Download ``ntl_open_pool.parquet`` into ``data/external/open_pool/``. The pool is already cleaned to
single plain paragraphs, drops passages written for (or close to) train prompts, and keeps only
gemini-3.7-flash rows with a documented Vertex AI route.

Its ``match_weight`` column reweights passages within each label so typography and length match the
real passages. We draw the same number of passages per label with those weights, so the sample is
class-balanced and looks like the competition data.

    uv run python -m name_that_llm.dataset --n-per-label 2000
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from name_that_llm.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

POOL_PATH = EXTERNAL_DATA_DIR / "open_pool" / "ntl_open_pool.parquet"


def pool_sample_path(n_per_label: int, seed: int = 0) -> Path:
    return INTERIM_DATA_DIR / f"pool_n{n_per_label}_s{seed}.parquet"


def build_pool_sample(
    n_per_label: int, seed: int = 0, pool_path: Path = POOL_PATH
) -> pd.DataFrame:
    cols = ["id", "text", "model", "prompt_canon", "match_weight", "source_dataset"]
    pool = pd.read_parquet(pool_path, columns=cols)
    parts = []
    for label, d in pool.groupby("model"):
        n = min(n_per_label, len(d))
        if n < n_per_label:
            logger.warning(f"{label}: only {len(d)} passages, taking all")
        # weighted sampling without replacement (Efraimidis-Spirakis): keep the n largest
        # u^(1/w). pandas refuses this when the weights are very uneven, as match_weight is.
        rng = np.random.default_rng([seed, len(parts)])
        keys = rng.random(len(d)) ** (1.0 / d.match_weight.clip(lower=1e-6).to_numpy())
        parts.append(d.iloc[np.argsort(-keys)[:n]])
    return pd.concat(parts, ignore_index=True)


def load_pool_sample(n_per_label: int, seed: int = 0) -> pd.DataFrame:
    """Load a cached sample, building it on first use."""
    path = pool_sample_path(n_per_label, seed)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_pool_sample(n_per_label, seed).to_parquet(path, index=False)
        logger.info(f"Built pool sample {path}")
    return pd.read_parquet(path)


@app.command()
def main(n_per_label: int = 2000, seed: int = 0):
    sample = load_pool_sample(n_per_label, seed)
    logger.success(
        f"{len(sample)} pool passages, {sample.model.nunique()} labels, "
        f"{sample.prompt_canon.nunique()} prompts -> {pool_sample_path(n_per_label, seed)}"
    )
    logger.info(f"\n{sample.model.value_counts().to_string()}")


if __name__ == "__main__":
    app()
