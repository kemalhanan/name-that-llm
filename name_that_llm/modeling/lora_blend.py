"""Compare LoRA members with the TF-IDF SVM on their held-out prompts, and pick blend weights.

Each Kaggle training notebook (kaggle_lora_train.py, MODE='full') writes held-out LoRA logits:
``holdout_logits.parquet`` covers prompt-grouped fold 0 of train.csv, and the 5-fold runs also
write ``oof_logits.parquet`` covering every passage. The comparison uses the passages that every
given member has held-out logits for: all 2,699 when each member comes from a 5-fold run, fold 0
when any member only has fold 0. The SVM's out-of-fold scores (``train.py --tag pool2000_w1``)
cover every passage with the same folds. No model saw these prompts in training, so the
comparison is fair.

The blend adds log-probabilities: ``log softmax(svm * t_svm) + sum_i w_i * log softmax(lora_i)``,
then Sinkhorn balancing as in predict.py. With only ~540 passages single grid points are noisy, so
the pick is the point with the best mean over itself and its grid neighbours.

    uv run python -m name_that_llm.modeling.lora_blend \\
        data/interim/lora4b/holdout_logits.parquet data/interim/lora8b/holdout_logits.parquet \\
        --names lora_out,lora_out_8b --tag lora4b_8b
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


def neighbour_mean(grid: dict, key: tuple, step: dict) -> float:
    """Mean score over a grid point and its immediate neighbours, which steadies the pick."""
    vals = [grid[key]]
    for i in range(len(key)):
        for d in (-1, 1):
            k2 = list(key)
            j = step[i].index(key[i]) + d
            if 0 <= j < len(step[i]):
                k2[i] = step[i][j]
                if tuple(k2) in grid:
                    vals.append(grid[tuple(k2)])
    return float(np.mean(vals))


HOLDOUT_ARG = typer.Argument(
    ..., help="holdout_logits.parquet or oof_logits.parquet of each LoRA member."
)


@app.command()
def main(
    holdout_paths: list[Path] = HOLDOUT_ARG,
    names: str = typer.Option(
        "", help="Comma-separated member names (default: each file's parent folder)."
    ),
    svm_scores_path: Path = INTERIM_DATA_DIR / "pool2000_w1_oof_scores.joblib",
    train_path: Path = RAW_DATA_DIR / "train.csv",
    balance_strength: float = 0.3,
    tag: str = "lora4b",
):
    member_names = names.split(",") if names else [p.parent.name for p in holdout_paths]
    assert len(member_names) == len(holdout_paths) == len(set(member_names))
    holds = [pd.read_parquet(p) for p in holdout_paths]
    # oof_logits.parquet: fold -1 marks a passage the run didn't score (only in timing runs)
    holds = [h[h.fold >= 0].drop(columns="fold") if "fold" in h else h for h in holds]
    labels = [c for c in holds[0].columns if c not in ("id", "model")]
    common = set(holds[0].id)
    for h in holds[1:]:
        assert list(h.columns) == list(holds[0].columns), "label columns differ between members"
        common &= set(h.id)
    sizes = ", ".join(f"{n} {len(h)}" for n, h in zip(member_names, holds))
    # the passages every member has held-out logits for, in the first member's order
    holds = [
        h.set_index("id").loc[[i for i in holds[0].id if i in common]].reset_index() for h in holds
    ]
    logger.info(
        f"{len(common)} passages held out by every member (each member holds out: {sizes})"
    )
    oof = joblib.load(svm_scores_path)
    assert list(oof["classes"]) == labels, "label order differs between the SVM and the LoRA"
    train = pd.read_csv(train_path)
    row = pd.Series(np.arange(len(train)), index=train.id)
    idx = row.loc[holds[0].id].to_numpy()
    assert (train.model.iloc[idx].to_numpy() == holds[0].model.to_numpy()).all()

    y = holds[0].model.map({m: i for i, m in enumerate(labels)}).to_numpy()
    svm = oof["scores"][idx]
    loras = [h[labels].to_numpy(dtype=float) for h in holds]
    k = len(labels)

    def score(logp, strength=0.0):
        p = sinkhorn(np.exp(logp - logp.max(1, keepdims=True)), strength)
        return balanced_accuracy(y, p.argmax(1), k)

    def mix(t, ws):
        logp = log_softmax(svm * t)
        for w, z in zip(ws, loras):
            logp = logp + w * log_softmax(z)
        return logp

    preds = {"svm": svm.argmax(1)} | {n: z.argmax(1) for n, z in zip(member_names, loras)}
    res = {
        "passages": len(y),
        "prompts": int(train.prompt_id.iloc[idx].nunique()),
        "members": member_names,
        "single": {"svm": score(log_softmax(svm))}
        | {n: score(log_softmax(z)) for n, z in zip(member_names, loras)},
        "agreement": {
            f"{a}~{b}": float(np.mean(preds[a] == preds[b]))
            for i, a in enumerate(preds)
            for b in list(preds)[i + 1 :]
        },
    }
    if len(loras) > 1:  # the LoRA members alone, equal weights
        res["single"]["loras_equal"] = score(sum(log_softmax(z) for z in loras))

    t_steps = (0.5, 1.0, 2.0)
    w_steps = (0.0, 0.25, 0.5, 1.0, 2.0)
    grid = {}
    for t in t_steps:
        for ws in np.array(np.meshgrid(*[w_steps] * len(loras))).T.reshape(-1, len(loras)):
            if ws.sum() > 0:
                grid[(t, *map(float, ws))] = score(mix(t, ws))
    steps = {0: t_steps} | {i + 1: w_steps for i in range(len(loras))}
    smooth = {key: neighbour_mean(grid, key, steps) for key in grid}
    best_raw = max(grid, key=grid.get)
    best = max(smooth, key=smooth.get)  # the pick: best on average with its neighbours

    def describe(key):
        return {"t_svm": key[0]} | {f"w_{n}": w for n, w in zip(member_names, key[1:])}

    top = sorted(grid, key=grid.get, reverse=True)[:10]
    res |= {
        "blend_top10": [describe(key) | {"bal_acc": grid[key]} for key in top],
        "blend_best_raw": describe(best_raw) | {"bal_acc": grid[best_raw]},
        "blend_pick": describe(best) | {"bal_acc": grid[best], "neighbourhood_mean": smooth[best]},
        "blend_pick_sinkhorn": score(mix(best[0], best[1:]), balance_strength),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{tag}_holdout_blend.json"
    out.write_text(json.dumps(res, indent=2))
    singles = " | ".join(f"{n} {s:.4f}" for n, s in res["single"].items())
    logger.success(
        f"{res['passages']} passages, {res['prompts']} prompts: {singles} | best blend {grid[best_raw]:.4f} "
        f"{describe(best_raw)} | pick {grid[best]:.4f} (neighbourhood {smooth[best]:.4f}) "
        f"{describe(best)}, +Sinkhorn {res['blend_pick_sinkhorn']:.4f} -> {out}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
