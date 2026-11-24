"""The GLM-5.3 likelihood member, shared by its out-of-fold measurement and the submission notebook.

``modeling/glm_blend.py`` measures it on the out-of-fold blend; ``kaggle_notebook.py`` copies this
file's body into the submission notebook, so both fit and apply the member the same way.

The member is a logistic regression on the 24 likelihood features of each model (``glm_ll/``)
plus six Flash - GLM-5.3 differences. Only the passages the blend most suspects of being GLM are
scored (the likelihood run is slow); the member's log-probabilities are added to theirs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GLM_TAGS = ("flash", "glm53")
GLM_LABELS = ("glm-5.3", "glm-5.3-flash")
DIFF_FEATURES = ("lp_mean", "lp_q50", "ent_mean", "top1", "logrank_mean", "fdg")


def likelihood_frame(flash: pd.DataFrame, glm53: pd.DataFrame, ids) -> pd.DataFrame:
    """Member inputs in ``ids`` order: each model's features (frames indexed by passage id as
    str), then the Flash - GLM-5.3 differences. A passage missing from either frame has NaNs."""
    blocks = [f.add_prefix(f"{tag}:") for tag, f in zip(GLM_TAGS, (flash, glm53))]
    frame = pd.concat(blocks, axis=1).reindex([str(i) for i in ids])
    for c in DIFF_FEATURES:
        frame[f"d:{c}"] = frame[f"flash:{c}"] - frame[f"glm53:{c}"]
    return frame


def fit_member(x: np.ndarray, y: np.ndarray, C: float):
    """Standardised features, then a multinomial logistic regression."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return model.fit(x, y)


def glm_mass(logp: np.ndarray, labels) -> np.ndarray:
    """The blend's probability of the two GLM labels, per passage (rows of log-probabilities)."""
    cols = [list(labels).index(c) for c in GLM_LABELS]
    z = np.exp(logp - logp.max(axis=1, keepdims=True))
    return (z[:, cols].sum(axis=1)) / z.sum(axis=1)
