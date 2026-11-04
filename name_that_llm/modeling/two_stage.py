"""Hierarchical classifiers: hard two-stage routing (family -> model) and a soft blend.

Most of the baseline's errors are between sibling models of one family, while the family itself
is predicted far more reliably. A dedicated stage-2 classifier per family only has to learn the
small differences between siblings, with feature vocabularies fitted on that family alone.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def family_of(models) -> np.ndarray:
    """'claude-opus-5' -> 'claude', 'qwen3.8-max' -> 'qwen'."""
    return (
        pd.Series(np.asarray(models, dtype=object))
        .str.extract(r"^([a-z]+)", expand=False)
        .to_numpy(dtype=object)
    )


class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    """Hard routing: stage 1 picks the family, stage 2 (fitted per family) picks the model.

    Parameters
    ----------
    family_estimator, member_estimator : sklearn estimators (pipelines) taking raw texts.
    """

    def __init__(self, family_estimator, member_estimator):
        self.family_estimator = family_estimator
        self.member_estimator = member_estimator

    def fit(self, X, y):
        X = pd.Series(np.asarray(X, dtype=object))
        y = np.asarray(y, dtype=object)
        fam = family_of(y)
        self.classes_ = np.unique(y)
        self.family_ = clone(self.family_estimator).fit(X, fam)
        self.members_ = {}
        for f in np.unique(fam):
            mask = fam == f
            members = np.unique(y[mask])
            # a single-model family needs no second stage
            self.members_[f] = (
                members[0]
                if len(members) == 1
                else clone(self.member_estimator).fit(X[mask], y[mask])
            )
        return self

    def predict_family(self, X):
        return np.asarray(
            self.family_.predict(pd.Series(np.asarray(X, dtype=object))), dtype=object
        )

    def predict_given_family(self, X, families):
        """Stage 2 only, routed by the given families (pass the true ones for an oracle score)."""
        X = pd.Series(np.asarray(X, dtype=object))
        families = np.asarray(families, dtype=object)
        out = np.empty(len(X), dtype=object)
        for f in np.unique(families):
            mask = families == f
            stage2 = self.members_[f]
            out[mask] = stage2 if isinstance(stage2, str) else stage2.predict(X[mask])
        return out

    def predict(self, X):
        return self.predict_given_family(X, self.predict_family(X))


def _decision_2d(est, X) -> np.ndarray:
    d = est.decision_function(X)
    return np.c_[np.zeros(len(d)), d] if d.ndim == 1 else d  # binary: softmax([0, d]) = sigmoid(d)


def _softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z * t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class BlendClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, flat, two_stage, a=1.0, b=1.0, c=1.0, w=0.5, mode="geo"):
        self.flat = flat
        self.two_stage = two_stage
        self.a, self.b, self.c, self.w, self.mode = a, b, c, w, mode

    def fit(self, X, y):
        X = pd.Series(np.asarray(X, dtype=object))
        y = np.asarray(y, dtype=object)
        self.classes_ = np.unique(y)
        self.flat_ = clone(self.flat).fit(X, y)
        self.two_stage_ = clone(self.two_stage).fit(X, y)
        return self

    def raw_scores(self, X) -> dict:
        """Unscaled decision scores, aligned to ``self.classes_`` (and to families)."""
        X = pd.Series(np.asarray(X, dtype=object))
        classes = self.classes_
        fam_of_class = family_of(classes)

        flat = _decision_2d(self.flat_, X)
        flat = flat[:, [list(self.flat_.classes_).index(m) for m in classes]]

        ts = self.two_stage_
        fam_scores = _decision_2d(ts.family_, X)
        fam_index = np.array([list(ts.family_.classes_).index(f) for f in fam_of_class])

        member = np.zeros((len(X), len(classes)))  # one-model families keep score 0 (prob 1)
        for est in ts.members_.values():
            if isinstance(est, str):
                continue
            d = _decision_2d(est, X)
            for j, m in enumerate(est.classes_):
                member[:, list(classes).index(m)] = d[:, j]
        return {
            "flat": flat,
            "family": fam_scores,
            "fam_index": fam_index,
            "member": member,
            "fam_of_class": fam_of_class,
        }

    def predict_proba(self, X):
        return blend_proba(self.raw_scores(X), self.a, self.b, self.c, self.w, self.mode)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def blend_proba(s: dict, a, b, c, w, mode) -> np.ndarray:
    eps = 1e-12
    p_flat = _softmax(s["flat"], a)
    p_fam = _softmax(s["family"], b)[:, s["fam_index"]]
    p_member = np.zeros_like(s["member"])
    for f in np.unique(s["fam_of_class"]):
        cols = np.where(s["fam_of_class"] == f)[0]
        p_member[:, cols] = _softmax(s["member"][:, cols], c)
    p_two = p_fam * p_member
    if mode == "linear":
        p = w * p_flat + (1 - w) * p_two
    else:
        logp = w * np.log(p_flat + eps) + (1 - w) * np.log(p_two + eps)
        p = np.exp(logp - logp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)
