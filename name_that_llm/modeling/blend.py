"""Blend the flat 18-class model with the two-stage (family -> model) model.

The two models make different mistakes: the flat model handles the one-model families
(kimi, qwen) better, the two-stage model separates gpt and gemini siblings better. Instead of
routing on a hard family decision, both are turned into probabilities over all 18 models and
combined:

    p_flat(m)      = softmax(a * flat_scores)[m]
    p_two(m)       = softmax(b * family_scores)[family(m)] * softmax(c * member_scores)[m]
    blend (geo)    = w * log p_flat + (1 - w) * log p_two
    blend (linear) = w * p_flat     + (1 - w) * p_two

Temperatures a, b, c, weight w and the mode are tuned on out-of-fold scores from
prompt-grouped CV. The reported accuracy is **nested**: for each fold, the blend parameters are
tuned on the other folds' OOF scores only, so the number is not flattered by the tuning.

    uv run python -m name_that_llm.modeling.blend            # CV, tune, fit final model
    uv run python -m name_that_llm.modeling.blend --no-fit   # CV and tuning only
"""

from itertools import product
import json
from pathlib import Path
import time

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import typer

from name_that_llm.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from name_that_llm.modeling.train import plot_confusion, svc_pipeline
from name_that_llm.modeling.two_stage import (
    BlendClassifier,
    TwoStageClassifier,
    blend_proba,
    family_of,
)

app = typer.Typer()

TEMPS = [0.5, 1, 2, 4, 8, 16]
WEIGHTS = np.round(np.linspace(0, 1, 11), 2)
MODES = ["geo", "linear"]


def subset(s: dict, idx) -> dict:
    return {k: (v[idx] if k in ("flat", "family", "member") else v) for k, v in s.items()}


def balanced_accuracy_idx(y_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall (the competition metric), on integer class indices."""
    hits = np.bincount(y_idx[pred_idx == y_idx], minlength=n_classes)
    support = np.bincount(y_idx, minlength=n_classes)
    present = support > 0
    return float(np.mean(hits[present] / support[present]))


def tune(s: dict, y, classes) -> tuple[dict, float]:
    """Grid-search blend parameters by balanced accuracy on the given scores."""
    y_idx = np.searchsorted(classes, y)
    best, best_acc = None, -1.0
    for a, b, c in product(TEMPS, TEMPS, TEMPS):
        for mode in MODES:
            for w in WEIGHTS:
                pred = blend_proba(s, a, b, c, w, mode).argmax(1)
                acc = balanced_accuracy_idx(y_idx, pred, len(classes))
                if acc > best_acc:
                    best = {"a": a, "b": b, "c": c, "w": float(w), "mode": mode}
                    best_acc = acc
    return best, best_acc


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "blend.joblib",
    C: float = 0.5,
    style_weight: float = 0.2,
    folds: int = 5,
    fit: bool = typer.Option(True, help="Fit the final model on all data after CV."),
    reuse_scores: bool = typer.Option(
        False, help="Reuse saved out-of-fold scores instead of refitting the CV models."
    ),
    tag: str = "blend",
):
    train = pd.read_csv(train_path)
    X, y, groups = train.text, train.model.to_numpy(dtype=object), train.prompt_id
    proto = BlendClassifier(
        flat=svc_pipeline(C, style_weight),
        two_stage=TwoStageClassifier(svc_pipeline(C, style_weight), svc_pipeline(C, style_weight)),
    )

    # 1. out-of-fold raw scores for both models
    oof_path = INTERIM_DATA_DIR / f"{tag}_oof_scores.joblib"
    splits = list(GroupKFold(n_splits=folds).split(X, y, groups))
    t0 = time.time()
    if reuse_scores:
        saved = joblib.load(oof_path)
        if not np.array_equal(saved["y"], y):
            raise ValueError(f"{oof_path} was computed on different training data")
        S, classes = saved["scores"], saved["classes"]
        logger.info(f"Reusing out-of-fold scores from {oof_path}")
    else:
        parts = []
        for k, (tr, va) in enumerate(splits):
            m = clone(proto).fit(X.iloc[tr], y[tr])
            parts.append((va, m.raw_scores(X.iloc[va])))
            logger.info(f"fold {k}: scores collected ({time.time() - t0:.0f}s)")
        classes = m.classes_
        n = len(X)
        S = {"flat": np.zeros((n, len(classes))), "member": np.zeros((n, len(classes)))}
        S["family"] = np.zeros((n, parts[0][1]["family"].shape[1]))
        for va, s in parts:
            for key in ("flat", "family", "member"):
                S[key][va] = s[key]
        S["fam_index"] = parts[0][1]["fam_index"]
        S["fam_of_class"] = parts[0][1]["fam_of_class"]
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scores": S, "y": y, "classes": classes, "splits": splits}, oof_path)

    # 2. reference points (no tuning needed beyond argmax)
    flat_pred = classes[S["flat"].argmax(1)]
    two_pred = classes[blend_proba(S, 1, 1, 1, 0.0, "geo").argmax(1)]  # soft two-stage only

    # 3. nested evaluation: tune on the other folds' OOF scores, apply to this fold
    nested = np.empty(len(X), dtype=object)
    fold_params = []
    for k, (_, va) in enumerate(splits):
        rest = np.setdiff1d(np.arange(len(X)), va)
        params, _ = tune(subset(S, rest), y[rest], classes)
        nested[va] = classes[blend_proba(subset(S, va), **params).argmax(1)]
        fold_params.append(params)
        logger.info(
            f"fold {k}: nested balanced acc="
            f"{balanced_accuracy_score(y[va], nested[va]):.4f} with {params}"
        )

    # 4. final parameters, tuned on all OOF scores
    params, tuned_acc = tune(S, y, classes)
    fam_true, fam_pred = family_of(y), family_of(nested)
    metrics = {
        "tag": tag,
        "metric": "balanced accuracy (mean per-model recall)",
        "C": C,
        "style_weight": style_weight,
        "folds": folds,
        "balanced_accuracy_nested": balanced_accuracy_score(y, nested),
        "accuracy_nested": accuracy_score(y, nested),
        "macro_f1_nested": f1_score(y, nested, average="macro"),
        "family_accuracy_nested": accuracy_score(fam_true, fam_pred),
        "balanced_accuracy_flat_oof": balanced_accuracy_score(y, flat_pred),
        "balanced_accuracy_soft_two_stage_oof": balanced_accuracy_score(y, two_pred),
        "balanced_accuracy_tuned_in_sample": tuned_acc,
        "final_params": params,
        "fold_params": fold_params,
        "per_model_recall": {m: float(np.mean(nested[y == m] == m)) for m in classes},
        "seconds": round(time.time() - t0, 1),
    }
    logger.success(
        f"balanced acc: flat {metrics['balanced_accuracy_flat_oof']:.4f} | soft two-stage "
        f"{metrics['balanced_accuracy_soft_two_stage_oof']:.4f} | BLEND nested "
        f"{metrics['balanced_accuracy_nested']:.4f} (family acc "
        f"{metrics['family_accuracy_nested']:.4f}) | in-sample tuned {tuned_acc:.4f} | {params}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{tag}_cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"id": train.id, "prompt_id": groups, "model": y, "pred": nested}).to_csv(
        INTERIM_DATA_DIR / f"{tag}_oof.csv", index=False
    )
    labels = sorted(classes, key=lambda m: (family_of([m])[0], m))
    plot_confusion(
        y,
        nested,
        labels,
        FIGURES_DIR / f"{tag}_confusion.png",
        f"{tag}: grouped {folds}-fold CV (nested), "
        f"balanced accuracy {metrics['balanced_accuracy_nested']:.3f}",
    )

    if fit:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        final = clone(proto).set_params(**params).fit(X, y)
        joblib.dump(final, model_path)
        logger.success(f"Final model saved to {model_path}")


if __name__ == "__main__":
    app()
