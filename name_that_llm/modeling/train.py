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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
from name_that_llm.features import build_features

app = typer.Typer()


class Classifier(str, Enum):
    svc = "svc"
    logreg = "logreg"


def family_of(models: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(models).str.extract(r"^([a-z]+)", expand=False).to_numpy()


def build_model(classifier: Classifier, C: float, style_weight: float) -> Pipeline:
    if classifier == Classifier.svc:
        clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000)
    else:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", build_features(style_weight)), ("clf", clf)])


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
    folds: int = 5,
    fit: bool = typer.Option(True, help="Fit the final model on all data after CV."),
    tag: str = "baseline",
):
    train = pd.read_csv(train_path)
    X, y, groups = train.text, train.model.to_numpy(), train.prompt_id
    labels = sorted(np.unique(y), key=lambda m: (family_of([m])[0], m))
    model = build_model(classifier, C, style_weight)
    logger.info(
        f"{len(train)} texts, {len(labels)} models, {groups.nunique()} prompts; "
        f"{classifier.value} C={C} style_weight={style_weight}"
    )

    oof = np.empty(len(train), dtype=object)
    fold_acc = []
    t0 = time.time()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
        fold_acc.append(accuracy_score(y[va], oof[va]))
        logger.info(f"fold {k}: acc={fold_acc[-1]:.4f}")

    fam_true, fam_pred = family_of(y), family_of(oof)
    metrics = {
        "tag": tag,
        "classifier": classifier.value,
        "C": C,
        "style_weight": style_weight,
        "folds": folds,
        "accuracy": accuracy_score(y, oof),
        "accuracy_fold_std": float(np.std(fold_acc)),
        "macro_f1": f1_score(y, oof, average="macro"),
        "family_accuracy": accuracy_score(fam_true, fam_pred),
        "within_family_accuracy": float(
            np.mean(oof[fam_true == fam_pred] == y[fam_true == fam_pred])
        ),
        "per_model_recall": {m: float(np.mean(oof[y == m] == m)) for m in labels},
        "cv_seconds": round(time.time() - t0, 1),
    }
    logger.success(
        f"CV accuracy {metrics['accuracy']:.4f} (±{metrics['accuracy_fold_std']:.4f}) | "
        f"macro-F1 {metrics['macro_f1']:.4f} | family acc {metrics['family_accuracy']:.4f} | "
        f"model acc given right family {metrics['within_family_accuracy']:.4f}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{tag}_cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"id": train.id, "prompt_id": groups, "model": y, "pred": oof}).to_csv(
        INTERIM_DATA_DIR / f"{tag}_oof.csv", index=False
    )
    plot_confusion(
        y,
        oof,
        labels,
        FIGURES_DIR / f"{tag}_confusion.png",
        f"{tag}: grouped {folds}-fold CV, accuracy {metrics['accuracy']:.3f}",
    )

    if fit:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clone(model).fit(X, y), model_path)
        logger.success(f"Final model saved to {model_path}")


if __name__ == "__main__":
    app()
