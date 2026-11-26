"""Measure a GLM-5.3 likelihood member on the out-of-fold blend.

``kaggle_glm_features.py`` scores train.csv under the real GLM-5.3 and GLM-5.3-Flash and writes
24 likelihood features per passage and model. This script turns them into an 18-class member
(a logistic regression on the two models' features plus six Flash - GLM-5.3 differences) and
adds its log-probabilities to the blend's.

Everything is nested in the prompt-grouped folds the LoRA members used: the member that scores a
fold is fitted on the other four. At test time only the passages the blend most suspects of being
GLM are scored (the likelihood run is slow), so the gain is reported per ``cover``: the share of
passages scored, taken from the top of the blend's own GLM probability.

    uv run python -m name_that_llm.modeling.glm_blend data/interim/glm_ll
"""

import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from name_that_llm.config import INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from name_that_llm.glm_member import (
    GLM_LABELS,
    GLM_TAGS,
    fit_member,
    glm_mass,
    likelihood_frame,
)
from name_that_llm.modeling.lora_blend import log_softmax
from name_that_llm.modeling.postprocess import balanced_accuracy, sinkhorn

app = typer.Typer()

# the submitted blend (README): folder with oof_logits.parquet -> weight
MEMBERS = {
    "lora4b_full": 0.25,
    "lora8b_full": 0.5,
    "lora_mistral7b_full": 0.5,
    "lora14b_full": 1.0,
}


def read_features(features_dir: Path, tag: str) -> pd.DataFrame:
    """One model's features by passage id: ``{tag}_features.csv``, or the parts of a run split over
    sessions (``{tag}_part{k}of{n}_features.csv``, from the TPU notebook's ``SHARD``)."""
    paths = sorted(features_dir.glob(f"{tag}_features.csv"))
    paths += sorted(features_dir.glob(f"{tag}_part*of*_features.csv"))
    assert paths, f"no {tag} features in {features_dir}"
    f = pd.concat([pd.read_csv(p, dtype={"id": str}) for p in paths], ignore_index=True)
    assert f.id.is_unique, f"{tag}: a passage is in more than one file of {features_dir}"
    return f.set_index("id")


def likelihood_matrix(features_dir: Path, ids) -> tuple[np.ndarray, list[str]]:
    """Feature matrix in ``ids`` order: both models' features, then the Flash - GLM-5.3 differences."""
    frame = likelihood_frame(*(read_features(features_dir, tag) for tag in GLM_TAGS), ids)
    return frame.to_numpy(np.float64), list(frame.columns)


def oof_member(x: np.ndarray, y: np.ndarray, fold: np.ndarray, k: int, C: float) -> np.ndarray:
    """Out-of-fold log-probabilities of the likelihood member."""
    out = np.zeros((len(y), k))
    for f in np.unique(fold):
        fit, held = fold != f, fold == f
        model = fit_member(x[fit], y[fit], C)
        assert len(model.classes_) == k
        out[held] = model.predict_log_proba(x[held])
    return out


def load_blend(train: pd.DataFrame, t_svm: float, svm_tag: str):
    """The submitted blend's out-of-fold log-probabilities, with labels, targets and folds."""
    holds = [pd.read_parquet(INTERIM_DATA_DIR / m / "oof_logits.parquet") for m in MEMBERS]
    ids = holds[0].id.to_numpy()
    holds = [h.set_index("id").loc[ids] for h in holds]
    labels = [c for c in holds[0].columns if c not in ("model", "fold")]
    fold = holds[0].fold.to_numpy()
    assert all((h.fold.to_numpy() == fold).all() for h in holds) and (fold >= 0).all()
    idx = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()
    svm = joblib.load(INTERIM_DATA_DIR / f"{svm_tag}_oof_scores.joblib")
    assert list(svm["classes"]) == labels
    logp = log_softmax(svm["scores"][idx] * t_svm)
    for h, w in zip(holds, MEMBERS.values()):
        logp = logp + w * log_softmax(h[labels].to_numpy(float))
    y = holds[0].model.map({m: i for i, m in enumerate(labels)}).to_numpy()
    return ids, idx, labels, y, fold, logp


FEATURES_ARG = typer.Argument(..., help="Folder with flash_features.csv and glm53_features.csv.")


@app.command()
def main(
    features_dir: Path = FEATURES_ARG,
    train_path: Path = RAW_DATA_DIR / "train.csv",
    svm_tag: str = "pool2000_w1",
    t_svm: float = 2.0,
    balance_strength: float = 0.3,
    C: float = 0.05,
    n_boot: int = 1000,
    tag: str = "glm_ll",
):
    train = pd.read_csv(train_path)
    ids, idx, labels, y, fold, logp = load_blend(train, t_svm, svm_tag)
    k = len(labels)
    x, names = likelihood_matrix(features_dir, ids)
    scored = np.isfinite(x).all(1)
    logger.info(
        f"{scored.sum()} of {len(y)} passages have likelihood features ({len(names)} each)"
    )
    assert scored.sum() >= 0.5 * len(y), "too few scored passages to fit the member"

    member = np.zeros((len(y), k))
    member[scored] = oof_member(x[scored], y[scored], fold[scored], k, C)

    def predict(lp):
        return sinkhorn(np.exp(lp - lp.max(1, keepdims=True)), balance_strength).argmax(1)

    base_pred = predict(logp)
    base = balanced_accuracy(y, base_pred, k)

    def recall(pred, c):
        return float((pred[y == c] == c).mean())

    # passages ranked by the blend's own probability of the two GLM labels, scored ones first
    mass = glm_mass(logp, labels)
    order = np.argsort(-np.where(scored, mass, -1.0), kind="stable")

    prompts = train.prompt_id.iloc[idx].to_numpy()
    by_prompt = [np.where(prompts == p)[0] for p in np.unique(prompts)]
    rng = np.random.default_rng(0)
    draws = [
        np.concatenate([by_prompt[j] for j in rng.integers(0, len(by_prompt), len(by_prompt))])
        for _ in range(n_boot)
    ]

    def interval(pred):
        d = np.array(
            [
                balanced_accuracy(y[i], pred[i], k) - balanced_accuracy(y[i], base_pred[i], k)
                for i in draws
            ]
        )
        return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.mean(d > 0))]

    rows = []
    for cover in (0.10, 0.15, 0.20, 0.30, 1.0):
        sel = order[: min(round(cover * len(y)), int(scored.sum()))]
        for w in (0.25, 0.5, 1.0, 2.0, 4.0):
            lp = logp.copy()
            lp[sel] += w * member[sel]
            pred = predict(lp)
            rows.append(
                {
                    "cover": cover,
                    "w": w,
                    "bal_acc": balanced_accuracy(y, pred, k),
                    "changed": int((pred != base_pred).sum()),
                    **{f"recall_{c}": recall(pred, labels.index(c)) for c in GLM_LABELS},
                    "pred": pred,
                }
            )
    # the weight is picked on the same passages it is scored on: 5 values, so the optimism is small
    for r in rows:
        lo, hi, win = interval(r.pop("pred"))
        r |= {"delta": r["bal_acc"] - base, "delta_lo": lo, "delta_hi": hi, "win": win}
    table = pd.DataFrame(rows)
    res = {
        "passages": len(y),
        "scored": int(scored.sum()),
        "C": C,
        "member_alone": balanced_accuracy(y[scored], member[scored].argmax(1), k),
        "blend": base,
        **{f"blend_recall_{c}": recall(base_pred, labels.index(c)) for c in GLM_LABELS},
        "grid": rows,
        "best": max(rows, key=lambda r: r["bal_acc"]),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{tag}_blend.json"
    out.write_text(json.dumps(res, indent=2))
    logger.info(f"member alone {res['member_alone']:.4f}; blend {base:.4f}")
    logger.info("\n" + table.round(4).to_string(index=False))
    logger.success(f"wrote {out}")


if __name__ == "__main__":
    app()
