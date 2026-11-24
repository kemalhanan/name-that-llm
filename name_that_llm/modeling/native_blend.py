"""Measure the native Qwen3-Embedding-8B expert and the prompt-group steps on the out-of-fold blend.

The expert's out-of-fold logits are CHEN Han's published ones (``oof_reference.npz`` in
lightningv08/ntl-olmo2-native24-final-inference-assets, ntl-native-q8-multidepth-fivefold-0679/),
copied with its calibration to ``data/external/native_q8/``. Its folds are prompt-pure like ours,
so both are out-of-fold for every passage.

The blend is the submitted one (``glm_blend.load_blend`` plus the GLM-5.3 member with the user-turn
prefix on the 20% most GLM-like passages, weight 2), with the new steps before the GLM selection:

    logp = centre(blend, groups, f) + w * native  ->  GLM member  ->  Sinkhorn 0.3  ->  argmax or assign

Groups are recovered from the texts by TF-IDF (``native_member.prompt_groups``), as the notebook
must do on the test set; true prompt ids are shown for reference.

    uv run python -m name_that_llm.modeling.native_blend
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import typer

from name_that_llm.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from name_that_llm.glm_member import glm_mass
from name_that_llm.modeling.glm_blend import likelihood_matrix, load_blend, oof_member
from name_that_llm.modeling.postprocess import balanced_accuracy, sinkhorn
from name_that_llm.native_member import assign, centre, eligible, prompt_groups
from name_that_llm.native_q8 import calibrate

app = typer.Typer()

NATIVE_DIR = EXTERNAL_DATA_DIR / "native_q8"


def native_oof(ids, groups):
    """The expert's calibrated out-of-fold log-probabilities in ``ids`` order, with these groups."""
    manifest = json.loads((NATIVE_DIR / "manifest.json").read_text())
    tok = np.load(NATIVE_DIR / "tokens.npz")
    oof = np.load(NATIVE_DIR / "oof_reference.npz")
    bias = np.load(NATIVE_DIR / "centered_bias.npy")
    pos = pd.Series(np.arange(len(tok["ids"])), index=tok["ids"].astype(str)).loc[ids].to_numpy()
    plain, cent, nfold = oof["plain"][pos], oof["centered"][pos], tok["folds"][pos]
    out = np.zeros(plain.shape)
    for f in range(5):
        state = json.loads((NATIVE_DIR / "calibration" / f"fold{f}.json").read_text())
        m = nfold == f
        # calibrate centres within the passages it is given, so one fold at a time
        out[m] = calibrate(plain[m][None], cent[m][None], bias[f : f + 1], groups[m],
                           [state["models"]["qwen8"]])[0]
    return out, manifest["classes"]


@app.command()
def main(
    svm_tag: str = "pool2000_w1",
    glm_dir: Path = INTERIM_DATA_DIR / "glm_ll_prefixb",
    thr: float = typer.Option(0.15, help="Similarity threshold of the TF-IDF grouping."),
    tag: str = "native_blend",
    draws: int = 1000,
):
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    ids, idx, labels, y, fold, logp0 = load_blend(train, 2.0, svm_tag)
    k = len(labels)
    prompts = np.unique(train.prompt_id.to_numpy()[idx], return_inverse=True)[1]

    # the notebook groups the whole test set at once; here all of train.csv, like a test set
    tfidf = prompt_groups(train.text.to_numpy()[idx], thr)
    sizes = np.bincount(tfidf)
    print(f"TF-IDF groups: {len(sizes)} (true {prompts.max() + 1}), ARI "
          f"{adjusted_rand_score(prompts, tfidf):.3f}, rows in groups of 2..18 {eligible(tfidf).mean():.3f}, "
          f"size counts {np.bincount(sizes).tolist()}")
    pure = pd.DataFrame({"g": tfidf, "p": prompts}).groupby("g").p.nunique()
    print(f"groups with passages of one prompt: {(pure == 1).mean():.3f}")

    x, _ = likelihood_matrix(glm_dir, ids)
    scored = np.isfinite(x).all(1)
    glm = np.zeros((len(y), k))
    glm[scored] = oof_member(x[scored], y[scored], fold[scored], k, 0.05)

    def finish(lp, how):
        mass = glm_mass(lp, labels)
        sel = np.argsort(-np.where(scored, mass, -1.0), kind="stable")[: round(0.2 * len(y))]
        lp = lp.copy()
        lp[sel] += 2.0 * glm[sel]
        q = np.log(np.maximum(sinkhorn(np.exp(lp - lp.max(1, keepdims=True)), 0.3), 1e-300))
        return q.argmax(1) if how is None else assign(q, how)

    by_prompt = [np.flatnonzero(prompts == p) for p in range(prompts.max() + 1)]
    rng = np.random.default_rng(0)
    boot = [np.concatenate([by_prompt[j] for j in rng.integers(0, len(by_prompt), len(by_prompt))])
            for _ in range(draws)]
    base = finish(logp0, None)
    ba = lambda p: balanced_accuracy(y, p, k)  # noqa: E731
    print(f"submitted blend (v7): {ba(base):.4f}")

    rows = []
    for gname, groups in (("tfidf", tfidf), ("true", prompts)):
        nat, nat_labels = native_oof(ids, groups)
        assert nat_labels == labels
        print(f"[{gname}] native alone {ba(nat.argmax(1)):.4f}")
        for f in (0.0, 0.25, 0.5, 0.75):
            for w in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                lp = centre(logp0, groups, f) + w * nat
                for post in ("argmax", "assign"):
                    pred = finish(lp, None if post == "argmax" else groups)
                    d = np.array([balanced_accuracy(y[i], pred[i], k) - balanced_accuracy(y[i], base[i], k)
                                  for i in boot])
                    rows.append(dict(groups=gname, f=f, w=w, post=post, bal_acc=round(ba(pred), 4),
                                     delta=round(ba(pred) - ba(base), 4),
                                     lo=round(float(np.quantile(d, 0.025)), 4),
                                     hi=round(float(np.quantile(d, 0.975)), 4)))
        t = pd.DataFrame([r for r in rows if r["groups"] == gname])
        print(t.pivot_table(index=["post", "f"], columns="w", values="bal_acc").round(4).to_string())
    t = pd.DataFrame(rows)
    best = t[t.groups == "tfidf"].sort_values("bal_acc").iloc[-1].to_dict()
    print("best with TF-IDF groups:", best)
    out = REPORTS_DIR / f"{tag}.json"
    out.write_text(json.dumps({"baseline": round(ba(base), 4), "thr": thr, "grid": rows, "best": best}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    app()
