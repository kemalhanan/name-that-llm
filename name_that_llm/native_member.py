"""Prompt groups and the group-level steps of the blend, shared by their out-of-fold measurement
(``modeling/native_blend.py``) and the submission notebook (``kaggle_notebook.py`` copies this
file's body into it).

Every train.csv prompt has exactly one passage from each of the 18 models, and the public stacks
assume the hidden test is built the same way. The test has no prompt ids, so passages are grouped
by TF-IDF similarity: ``capped_agglo`` is copied unchanged from the 0.716 public notebook
(estopadilla/0-716-robust-glm-5-3-likelihood-user-prefix, lumibee's and lightningv08's stack).

Three uses of the groups:
  * CHEN Han's (concyclics) native Qwen3-Embedding-8B expert mixes in a centred head, which reads
    each passage against the mean of its group (``native_q8.calibrate``);
  * ``centre``: the blend's log-probabilities minus a share of their group mean;
  * ``assign``: within each group, every passage gets a different model (Hungarian assignment).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer

GROUP_MIN, GROUP_MAX = 2, 18


def capped_agglo(S, cap=18, thr=0.02):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)  # noqa: E702
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)  # noqa: E702
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)  # noqa: E702
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break  # noqa: E701
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf  # noqa: E702
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i  # noqa: E702
    return lab


def prompt_groups(texts, thr=0.15):
    """Group ids (0..g-1) for passages that look written for the same prompt, at most 18 a group."""
    try:
        v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
        x = v.fit_transform(list(texts))
        lab = capped_agglo((x @ x.T).toarray(), GROUP_MAX, thr)
    except ValueError:  # too few passages for the vocabulary limits
        lab = np.arange(len(texts))
    return np.unique(lab, return_inverse=True)[1]


def eligible(groups):
    size = np.bincount(groups)[groups]
    return (size >= GROUP_MIN) & (size <= GROUP_MAX)


def centre(logp, groups, f):
    """Each passage's log-probabilities minus ``f`` times their mean over its group."""
    means = np.zeros((groups.max() + 1, logp.shape[1]))
    np.add.at(means, groups, logp)
    means /= np.bincount(groups)[:, None]
    return np.where(eligible(groups)[:, None], logp - f * means[groups], logp)


def assign(logp, groups):
    """Argmax, except that within each group of 2..18 passages no two get the same model."""
    pred = logp.argmax(axis=1)
    ok = eligible(groups)
    for g in np.unique(groups[ok]):
        rows = np.flatnonzero(groups == g)
        r, c = linear_sum_assignment(-logp[rows])
        pred[rows[r]] = c
    return pred
