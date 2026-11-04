"""Sinkhorn class balancing of test-time predictions.

Balanced accuracy weighs every model equally, so a classifier that over-predicts a few labels on the
test set loses recall on the rest. Sinkhorn balancing rescales the probability matrix so that each
class gets roughly the same total probability mass over the whole test set, then mixes the result
with the original probabilities (``strength``: 0 = off, 1 = fully balanced).

The hidden test's class mix is NOT uniform (and differs between public and private LB), so full
balancing can hurt. ``evaluate`` measures the effect on out-of-fold scores, on simulated test sets
whose class proportions are drawn from a Dirichlet distribution.

    uv run python -m name_that_llm.modeling.postprocess --tag pool2000_w1
"""

import json

import joblib
from loguru import logger
import numpy as np
import typer

from name_that_llm.config import INTERIM_DATA_DIR, REPORTS_DIR

app = typer.Typer()


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores * temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sinkhorn(proba: np.ndarray, strength: float = 1.0, n_iter: int = 50) -> np.ndarray:
    """Geometric mix of ``proba`` and its column-balanced Sinkhorn projection."""
    if strength <= 0:
        return proba
    n, k = proba.shape
    q = proba + 1e-12
    for _ in range(n_iter):
        q = q / q.sum(axis=0, keepdims=True) * (n / k)  # each class gets n/k total mass
        q = q / q.sum(axis=1, keepdims=True)  # each row is a distribution
    logp = (1 - strength) * np.log(proba + 1e-12) + strength * np.log(q)
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def balanced_accuracy(y_idx: np.ndarray, pred_idx: np.ndarray, k: int) -> float:
    support = np.bincount(y_idx, minlength=k)
    hits = np.bincount(y_idx[pred_idx == y_idx], minlength=k)
    present = support > 0
    return float(np.mean(hits[present] / support[present]))


@app.command()
def evaluate(
    tag: str = "baseline",
    n_sims: int = 200,
    test_size: int = 1200,
    concentration: float = typer.Option(
        5.0, help="Dirichlet concentration of simulated class mixes (lower = more uneven)."
    ),
    seed: int = 0,
):
    saved = joblib.load(INTERIM_DATA_DIR / f"{tag}_oof_scores.joblib")
    scores, classes = saved["scores"], saved["classes"]
    y_idx = np.searchsorted(classes, saved["y"])
    k = len(classes)
    rng = np.random.default_rng(seed)
    by_class = [np.where(y_idx == c)[0] for c in range(k)]

    # simulated test sets: class proportions ~ Dirichlet, rows drawn from the OOF pool
    sims = []
    for _ in range(n_sims):
        props = rng.dirichlet(np.full(k, concentration))
        counts = np.maximum(1, np.round(props * test_size).astype(int))
        sims.append(np.concatenate([rng.choice(by_class[c], counts[c]) for c in range(k)]))

    temps = [0.5, 1, 2, 4, 8]
    strengths = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]
    results = {}
    for t in temps:
        for s in strengths:
            accs = [
                balanced_accuracy(y_idx[idx], sinkhorn(softmax(scores[idx], t), s).argmax(1), k)
                for idx in sims
            ]
            results[(t, s)] = (float(np.mean(accs)), float(np.std(accs)))

    base = results[(1, 0.0)][0]  # strength 0 = plain argmax, independent of temperature
    full_oof = balanced_accuracy(y_idx, scores.argmax(1), k)
    rows = sorted(results.items(), key=lambda kv: -kv[1][0])
    logger.info(f"{tag}: plain argmax on full OOF {full_oof:.4f}; on simulated tests {base:.4f}")
    for (t, s), (m, sd) in rows[:8]:
        logger.info(f"  T={t:<4} strength={s:<4}: {m:.4f} ± {sd:.4f}  ({m - base:+.4f})")
    (t_best, s_best), (m_best, _) = rows[0]
    out = {
        "tag": tag,
        "concentration": concentration,
        "test_size": test_size,
        "n_sims": n_sims,
        "plain_full_oof": full_oof,
        "plain_simulated": base,
        "best": {"temperature": t_best, "strength": s_best, "simulated": m_best},
        "grid": {f"T={t},s={s}": v[0] for (t, s), v in results.items()},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{tag}_sinkhorn.json").write_text(json.dumps(out, indent=2))
    logger.success(f"best T={t_best}, strength={s_best}: {m_best:.4f} ({m_best - base:+.4f})")


if __name__ == "__main__":
    app()
