"""Compare the LoRA model with the TF-IDF SVM on the LoRA run's held-out prompts, and pick a blend.

The Kaggle training notebook (kaggle_lora_train.py, MODE='full') writes
``lora_out/holdout_logits.parquet``: LoRA logits for prompt-grouped fold 0 of train.csv. The SVM's
out-of-fold scores for the same passages come from ``train.py --tag pool2000_w1``. Both models
were trained without these prompts, so the comparison is fair.

The blend adds log-probabilities: ``log softmax(svm * t_svm) + w * log softmax(lora)``, then
Sinkhorn balancing as in predict.py. With only ~540 passages the grid is kept coarse; the chosen
weight is a starting point, not a precise optimum.

    uv run python -m name_that_llm.modeling.lora_blend data/interim/lora4b/holdout_logits.parquet
"""

import json
from pathlib import Path
import sys

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from name_that_llm.config import INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from name_that_llm.modeling.postprocess import balanced_accuracy, sinkhorn

app = typer.Typer()


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def blend(svm, lora, w: float, t_svm: float = 1.0) -> np.ndarray:
    return log_softmax(svm * t_svm) + w * log_softmax(lora)


@app.command()
def main(
    holdout_path: Path,
    svm_scores_path: Path = INTERIM_DATA_DIR / "pool2000_w1_oof_scores.joblib",
    train_path: Path = RAW_DATA_DIR / "train.csv",
    balance_strength: float = 0.3,
    tag: str = "lora4b",
):
    hold = pd.read_parquet(holdout_path)
    labels = [c for c in hold.columns if c not in ("id", "model")]
    oof = joblib.load(svm_scores_path)
    assert list(oof["classes"]) == labels, "label order differs between the SVM and the LoRA"
    train = pd.read_csv(train_path)
    row = pd.Series(np.arange(len(train)), index=train.id)
    idx = row.loc[hold.id].to_numpy()
    assert (train.model.iloc[idx].to_numpy() == hold.model.to_numpy()).all()

    y = hold.model.map({m: i for i, m in enumerate(labels)}).to_numpy()
    svm, lora = oof["scores"][idx], hold[labels].to_numpy(dtype=float)
    k = len(labels)

    def score(logp, strength=0.0):
        p = sinkhorn(np.exp(logp - logp.max(1, keepdims=True)), strength)
        return balanced_accuracy(y, p.argmax(1), k)

    res = {
        "passages": len(hold),
        "prompts": int(train.prompt_id.iloc[idx].nunique()),
        "svm": score(log_softmax(svm)),
        "lora": score(log_softmax(lora)),
        "agreement": float(np.mean(svm.argmax(1) == lora.argmax(1))),
    }
    grid = {}
    for t in (0.5, 1.0, 2.0):
        for w in (0.25, 0.5, 1.0, 2.0, 4.0):
            grid[f"t{t}_w{w}"] = score(blend(svm, lora, w, t))
    best = max(grid, key=grid.get)
    t_best, w_best = (float(x[1:]) for x in best.split("_"))
    res |= {
        "blend_grid": grid,
        "blend_best": {"t_svm": t_best, "w_lora": w_best, "bal_acc": grid[best]},
        "blend_best_sinkhorn": score(blend(svm, lora, w_best, t_best), balance_strength),
        "lora_sinkhorn": score(log_softmax(lora), balance_strength),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{tag}_holdout_blend.json"
    out.write_text(json.dumps(res, indent=2))
    logger.success(
        f"fold 0 ({res['passages']} passages): SVM {res['svm']:.4f} | LoRA {res['lora']:.4f} | "
        f"best blend {grid[best]:.4f} (t_svm={t_best}, w_lora={w_best}), "
        f"+Sinkhorn {res['blend_best_sinkhorn']:.4f} | agree {res['agreement']:.2f} -> {out}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
