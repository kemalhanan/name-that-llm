"""Compare two runs of the GLM likelihood scoring on the passages both scored.

The TPU notebook scores train.csv on CPUs, while a submission would score the test on two T4s. The
likelihood member only transfers if the two give the same features, so the TPU timing run scores
the same 40 passages as the GPU timing run, and this script compares them: per token (log-prob,
entropy, rank) and per passage (the 24 features, each against its spread between passages).

    uv run python -m name_that_llm.modeling.glm_compare data/interim/glm_ll_gpu_timing data/interim/glm_ll_tpu_timing
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from name_that_llm.modeling.glm_blend import read_features

app = typer.Typer()

# A passage feature's difference is judged against its spread between passages, which is what the
# member (a logistic regression on standardised features) reads. A systematic shift matters most:
# train would be scored on one backend and the test on the other. Unbiased noise of a fraction of a
# spread only blurs a feature, and some features (the upper log-prob quantiles) have so little spread
# that ~0.01 nats of per-token noise, normal between backends, is already 0.1 of it. So noise gets a
# loose limit, and a shift fails only when it is both clear (3 standard errors over the passages)
# and material. Within one backend f32 runs agree to 1e-3 per token (the author's compare.py).
MAX_REL_DIFF = 0.25  # mean |difference| / std between passages
MAX_REL_BIAS = 0.05  # |mean difference| / std, when it is also beyond 3 standard errors
MIN_CORR = 0.97
# per token
MAX_LP_MEAN, MAX_LP_P99, MAX_LP_MAX = 0.02, 0.15, 1.0  # |d logprob|, nats
MAX_ENT_MEAN = 0.02  # |d entropy|, nats
# The rank-share features (share of tokens ranked under k) move in steps of 1/n_tok per passage, so
# over 40 passages a harmless 1% of tokens crossing a boundary already fails the passage limits above.
# They are judged per token instead: the crossing rate, and a sign test for a one-way shift. Their
# passage table is only logged.
SHARE_FEATURES = {"top1": 1, "top5": 5, "top20": 20, "rank_ge100": 100}
MAX_CROSSINGS = 0.01


def token_scores(run_dir: Path, tag: str) -> dict[str, np.ndarray]:
    """Per-token [logprob, entropy, E[logp^2], rank] of each passage, by id. The ids come from the
    features file next to each scores file (same order; the npz stores them as a pickled array)."""
    out = {}
    paths = sorted(run_dir.glob(f"{tag}_features.csv"))
    paths += sorted(run_dir.glob(f"{tag}_part*of*_features.csv"))
    for p in paths:
        ids = pd.read_csv(p, dtype={"id": str}).id
        with np.load(p.with_name(p.name.replace("_features.csv", "_scores.npz"))) as z:
            scores, off = z["scores"], z["offsets"]
        assert len(off) == len(ids) + 1, p
        out |= {i: scores[off[j] : off[j + 1]] for j, i in enumerate(ids)}
    return out


@app.command()
def main(reference: Path, other: Path):
    ok = True
    for tag in ("flash", "glm53"):
        a, b = read_features(reference, tag), read_features(other, tag)
        ids = a.index.intersection(b.index)
        assert len(ids), f"{tag}: no passage in common"
        logger.info(f"{tag}: {len(ids)} passages in common")

        sa, sb = token_scores(reference, tag), token_scores(other, tag)
        lens = [(len(sa[i]), len(sb[i])) for i in ids]
        if any(x != y for x, y in lens):
            logger.error(
                f"{tag}: token counts differ on {sum(x != y for x, y in lens)} passages (tokenizer?)"
            )
            ok = False
            continue
        ta, tb = np.concatenate([sa[i] for i in ids]), np.concatenate([sb[i] for i in ids])
        d = np.abs(ta[:, 0] - tb[:, 0])
        d_ent = np.abs(ta[:, 1] - tb[:, 1]).mean()
        logger.info(
            f"{tag} per token: |d logprob| mean {d.mean():.3f}, p99 {np.quantile(d, 0.99):.3f}, max {d.max():.3f} nats; "
            f"|d entropy| mean {d_ent:.3f}; rank changes {np.mean(ta[:, 3] != tb[:, 3]):.1%}"
        )
        if not (
            d.mean() <= MAX_LP_MEAN
            and np.quantile(d, 0.99) <= MAX_LP_P99
            and d.max() <= MAX_LP_MAX
            and d_ent <= MAX_ENT_MEAN
        ):
            logger.error(f"{tag}: per-token log-prob or entropy differences beyond tolerance")
            ok = False
        for feature, k in SHARE_FEATURES.items():
            below_a, below_b = ta[:, 3] < k, tb[:, 3] < k
            lost, gained = int((below_a & ~below_b).sum()), int((~below_a & below_b).sum())
            rate = (lost + gained) / len(ta)
            one_way = abs(gained - lost) > 3 * np.sqrt(lost + gained)
            logger.info(f"{tag} rank < {k}: {rate:.2%} of tokens cross ({gained} in, {lost} out)")
            if rate > MAX_CROSSINGS or one_way:
                why = "too many crossings" if rate > MAX_CROSSINGS else "a one-way shift"
                logger.error(f"{tag}: rank < {k} ({feature}): {why}")
                ok = False

        fa, fb = a.loc[ids], b.loc[ids]
        rows = []
        for c in fa.columns:
            x, y = fa[c].to_numpy(float), fb[c].to_numpy(float)
            spread = x.std() or 1.0
            corr = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 1.0
            delta = y - x
            se = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else 0.0
            rows.append(
                {
                    "feature": c,
                    "rel_diff": np.abs(delta).mean() / spread,
                    "rel_bias": delta.mean() / spread,
                    "bias_se": abs(delta.mean()) / se if se > 0 else 0.0,
                    "corr": corr,
                }
            )
        t = pd.DataFrame(rows).set_index("feature")
        shift = (t.rel_bias.abs() > MAX_REL_BIAS) & (t.bias_se > 3)
        over = (t.rel_diff > MAX_REL_DIFF) | shift | (t["corr"] < MIN_CORR)
        shares = t.index.isin(list(SHARE_FEATURES))
        logger.info(f"{tag} per passage:\n{t.round(3).to_string()}")
        if (over & shares).any():
            logger.warning(
                f"{tag}: rank shares over the passage limits (judged per token above): "
                + ", ".join(t.index[over & shares])
            )
        if (over & ~shares).any():
            logger.error(
                f"{tag}: passage features differ beyond tolerance: "
                + ", ".join(t.index[over & ~shares])
            )
            ok = False
    if ok:
        logger.success("the two runs give the same features within tolerance")
    else:
        logger.error("the two runs differ: features from one cannot stand in for the other")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
