"""Two-stage (hierarchical) classifier: predict the model family, then the model within it.

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
