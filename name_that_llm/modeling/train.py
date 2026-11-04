"""Baseline: TF-IDF (char + word) + stylometric features -> linear classifier.

Cross-validation is grouped by ``prompt_id``: every model answered every prompt, so a random
split would leak topic between train and validation folds.

    uv run python -m name_that_llm.modeling.train            # CV + fit final model
    uv run python -m name_that_llm.modeling.train --no-fit   # CV only
"""

from enum import Enum
import json
from pathlib import Path
import time

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import typer

from name_that_llm.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from name_that_llm.dataset import load_pool_sample
from name_that_llm.features import build_features
from name_that_llm.modeling.two_stage import TwoStageClassifier, family_of

app = typer.Typer()


class Classifier(str, Enum):
    svc = "svc"
    logreg = "logreg"
    two_stage = "two-stage"


def svc_pipeline(C: float, style_weight: float) -> Pipeline:
    clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000)
    return Pipeline([("features", build_features(style_weight)), ("clf", clf)])


def build_model(
    classifier: Classifier, C: float, style_weight: float, member_C: float | None = None
):
    if classifier == Classifier.svc:
        return svc_pipeline(C, style_weight)
    if classifier == Classifier.two_stage:
        return TwoStageClassifier(
            family_estimator=svc_pipeline(C, style_weight),
            member_estimator=svc_pipeline(member_C if member_C is not None else C, style_weight),
        )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", build_features(style_weight)), ("clf", clf)])


def fit_with_pool(model, X, y, pool: pd.DataFrame | None, pool_weight: float = 1.0):
    """Fit on the given rows plus (optionally) external pool rows, down-weighted by pool_weight."""
    if pool is None:
        return model.fit(X, y)
    if not isinstance(model, Pipeline):
        raise NotImplementedError("pool training is only wired up for the flat svc/logreg models")
    X_all = pd.concat([pd.Series(X, dtype=object), pool.text.astype(object)], ignore_index=True)
    y_all = np.concatenate([np.asarray(y, dtype=object), pool.model.to_numpy(dtype=object)])
    w = np.concatenate([np.ones(len(X)), np.full(len(pool), pool_weight)])
    return model.fit(X_all, y_all, clf__sample_weight=w)


def plot_confusion(y_true, y_pred, labels, path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        if v >= 0.05:
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "baseline.joblib",
    classifier: Classifier = Classifier.svc,
    C: float = 0.5,
    style_weight: float = 0.2,
    member_C: float = typer.Option(None, help="two-stage only: C of the per-family models."),
    folds: int = 5,
    fit: bool = typer.Option(True, help="Fit the final model on all data after CV."),
    pool_n: int = typer.Option(
        0, help="Open Pool passages per label added to training (0 = none)."
    ),
    pool_weight: float = typer.Option(1.0, help="Sample weight of pool passages."),
    cv: bool = typer.Option(True, help="Run grouped CV (--no-cv: only fit the final model)."),
    tag: str = "baseline",
):
    train = pd.read_csv(train_path)
    X, y, groups = train.text, train.model.to_numpy(dtype=object), train.prompt_id
    labels = sorted(np.unique(y), key=lambda m: (family_of([m])[0], m))
    model = build_model(classifier, C, style_weight, member_C)
    pool = load_pool_sample(pool_n) if pool_n > 0 else None
    logger.info(
        f"{len(train)} texts, {len(labels)} models, {groups.nunique()} prompts; "
        f"{classifier.value} C={C} member_C={member_C} style_weight={style_weight}; "
        f"pool {0 if pool is None else len(pool)} rows x weight {pool_weight}"
    )
    if not cv:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit_with_pool(clone(model), X, y, pool, pool_weight), model_path)
        logger.success(f"Final model saved to {model_path} (CV skipped)")
        return

    oof = np.empty(len(train), dtype=object)
    oracle = np.empty(len(train), dtype=object)  # two-stage: stage 2 routed by the TRUE family
    classes = np.unique(y)
    oof_scores = np.full(
        (len(train), len(classes)), np.nan
    )  # decision scores, for post-processing
    fold_acc = []
    t0 = time.time()
    # pool passages only ever join the training side: validation stays real train.csv passages
    for k, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        m = fit_with_pool(clone(model), X.iloc[tr], y[tr], pool, pool_weight)
        oof[va] = m.predict(X.iloc[va])
        if hasattr(m, "decision_function"):
            order = [list(m.classes_).index(c) for c in classes]
            oof_scores[va] = m.decision_function(X.iloc[va])[:, order]
        if isinstance(m, TwoStageClassifier):
            oracle[va] = m.predict_given_family(X.iloc[va], family_of(y[va]))
        fold_acc.append(balanced_accuracy_score(y[va], oof[va]))
        logger.info(f"fold {k}: balanced acc={fold_acc[-1]:.4f}")

    fam_true, fam_pred = family_of(y), family_of(oof)
    metrics = {
        "tag": tag,
        "classifier": classifier.value,
        "C": C,
        "member_C": member_C,
        "style_weight": style_weight,
        "pool_n_per_label": pool_n,
        "pool_rows": 0 if pool is None else len(pool),
        "pool_weight": pool_weight,
        "folds": folds,
        "metric": "balanced accuracy (mean per-model recall)",
        "balanced_accuracy": balanced_accuracy_score(y, oof),
        "accuracy": accuracy_score(y, oof),
        "balanced_accuracy_fold_std": float(np.std(fold_acc)),
        "macro_f1": f1_score(y, oof, average="macro"),
        "family_accuracy": accuracy_score(fam_true, fam_pred),
        "within_family_accuracy": float(
            np.mean(oof[fam_true == fam_pred] == y[fam_true == fam_pred])
        ),
        "per_model_recall": {m: float(np.mean(oof[y == m] == m)) for m in labels},
        "cv_seconds": round(time.time() - t0, 1),
    }
    if classifier == Classifier.two_stage:
        metrics["oracle_family_accuracy"] = balanced_accuracy_score(y, oracle)
    logger.success(
        f"CV balanced acc {metrics['balanced_accuracy']:.4f} (±{metrics['balanced_accuracy_fold_std']:.4f}) | "
        f"macro-F1 {metrics['macro_f1']:.4f} | family acc {metrics['family_accuracy']:.4f} | "
        f"model acc given right family {metrics['within_family_accuracy']:.4f}"
        + (
            f" | accuracy if family were known {metrics['oracle_family_accuracy']:.4f}"
            if "oracle_family_accuracy" in metrics
            else ""
        )
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{tag}_cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"id": train.id, "prompt_id": groups, "model": y, "pred": oof}).to_csv(
        INTERIM_DATA_DIR / f"{tag}_oof.csv", index=False
    )
    if not np.isnan(oof_scores).any():
        joblib.dump(
            {"scores": oof_scores, "classes": classes, "y": y, "groups": groups.to_numpy()},
            INTERIM_DATA_DIR / f"{tag}_oof_scores.joblib",
        )
    plot_confusion(
        y,
        oof,
        labels,
        FIGURES_DIR / f"{tag}_confusion.png",
        f"{tag}: grouped {folds}-fold CV, balanced accuracy {metrics['balanced_accuracy']:.3f}",
    )

    if fit:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit_with_pool(clone(model), X, y, pool, pool_weight), model_path)
        logger.success(f"Final model saved to {model_path}")


if __name__ == "__main__":
    app()
