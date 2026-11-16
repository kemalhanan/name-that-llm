"""Build the data files the GUI reads, from the project's out-of-fold predictions.

    uv run python gui/build_data.py

Writes two files next to this script:

* ``data.js``      out-of-fold scores of every judge (blend member) and the summary numbers.
* ``passages.js``  the text of a sample of train.csv passages, for the game and the walk-through.
                   It holds competition data, so it is git-ignored; the GUI works without it.
"""

import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from name_that_llm.features import style_features  # noqa: E402
from name_that_llm.modeling.postprocess import balanced_accuracy, sinkhorn  # noqa: E402

GUI = Path(__file__).resolve().parent
INTERIM = ROOT / "data" / "interim"

# judge key -> (name, what it is in plain words, folder with oof_logits.parquet)
LORAS = {
    "q4": ("Qwen3-4B", "a small language model", INTERIM / "lora4b_full"),
    "q8": ("Qwen3-8B", "a mid-size language model", INTERIM / "lora8b_full"),
    "m7": ("Mistral-7B", "a language model from another family", INTERIM / "lora_mistral7b_full"),
    "q14": ("Qwen3-14B", "the largest language model", INTERIM / "lora14b_full"),
}
SVM = ("Style counter", "counts letters, words and punctuation", INTERIM / "pool2000_w1_oof_scores.joblib")
JUDGES = ["svm", "q4", "q8", "m7", "q14"]

# svm: temperature on its scores; the others: weight on their log-probabilities
SUBMITTED = {"svm": 2.0, "q4": 1.0, "q8": 2.0, "m7": 1.0, "q14": 0.0}  # public LB 0.611
CANDIDATE = {"svm": 2.0, "q4": 0.25, "q8": 0.5, "m7": 0.5, "q14": 1.0}  # lora_blend.py's pick
STRENGTH = 0.3

# style clues shown in the game: feature -> what a person would call it
CLUES = {
    "em_dash": "em dashes (—)",
    "curly_quote": "curly quotes (“ ”)",
    "straight_quote": "straight quotes (\" ')",
    "semicolon": "semicolons",
    "colon": "colons",
    "parens": "brackets",
    "words_per_sentence": "long sentences",
}


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def family(label: str) -> str:
    return "".join(c for c in label.split("-")[0] if c.isalpha())


def main():
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    scores, judges, labels = {}, {}, None

    saved = joblib.load(SVM[2])
    labels = list(saved["classes"])
    assert (np.asarray(saved["y"]) == train.model.to_numpy()).all()
    scores["svm"] = saved["scores"]
    judges["svm"] = {"name": SVM[0], "role": SVM[1], "kind": "svm"}

    for key, (name, role, d) in LORAS.items():
        oof = pd.read_parquet(d / "oof_logits.parquet").set_index("id").loc[train.id]
        assert [c for c in oof.columns if c not in ("model", "fold")] == labels
        assert (oof.model.to_numpy() == train.model.to_numpy()).all() and (oof.fold >= 0).all()
        scores[key] = oof[labels].to_numpy(dtype=float)
        m = json.loads((d / "meta.json").read_text())
        runs = [m["stage1"], *m["folds"], m["stage2_all"]]
        # a resumed run only timed its last part, so use the steady time per step when it is there
        hours = sum(r["steps"] * r["steady_s_per_step"] if "steady_s_per_step" in r else r["seconds"] for r in runs) / 3600
        judges[key] = {
            "name": name,
            "role": role,
            "kind": "lora",
            "keep": m["cfg"]["keep"],
            "depth": m["cfg"]["base"]["num_hidden_layers"],
            "taps": m["cfg"]["layers"],
            "hidden": m["cfg"]["base"]["hidden_size"],
            "pool_k": round(m["stage1"]["steps"] * m["cfg"]["bs"] / 1000),
            "hours": round(hours, 1),
        }

    k, n = len(labels), len(train)
    y = train.model.map({m: i for i, m in enumerate(labels)}).to_numpy()
    fams = [family(m) for m in labels]
    fam_of = np.array([sorted(set(fams)).index(f) for f in fams])
    words = train.text.str.split().str.len().to_numpy()
    prompts = train.prompt_id.to_numpy()

    def blend(w, strength=STRENGTH):
        logp = np.zeros((n, k))
        for key, v in w.items():
            if v > 0:
                logp += log_softmax(scores[key] * v) if key == "svm" else v * log_softmax(scores[key])
        p = np.exp(logp - logp.max(1, keepdims=True))
        return sinkhorn(p / p.sum(1, keepdims=True), strength)

    def bal(pred, idx=slice(None)):
        return balanced_accuracy(y[idx], pred[idx], k)

    p_cand = blend(CANDIDATE)
    sub, cand = blend(SUBMITTED).argmax(1), p_cand.argmax(1)
    single = {key: scores[key].argmax(1) for key in JUDGES}
    for key in JUDGES:
        judges[key]["oof"] = bal(single[key])

    # how sure can we be? resample whole prompts
    rng = np.random.default_rng(0)
    by_prompt = [np.where(prompts == p)[0] for p in np.unique(prompts)]
    boot, delta = [], []
    for _ in range(1000):
        idx = np.concatenate([by_prompt[i] for i in rng.integers(0, len(by_prompt), len(by_prompt))])
        boot.append(bal(sub, idx))
        delta.append(bal(cand, idx) - boot[-1])

    # experiments quoted in the recommendations
    X = np.hstack([log_softmax(scores[c] * (2.0 if c == "svm" else 1.0)) for c in JUDGES])
    stacked = np.zeros(n, int)
    for tr, te in GroupKFold(5).split(X, y, prompts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # stops at max_iter; close enough for this check
            stacked[te] = LogisticRegression(C=0.05, max_iter=1500).fit(X[tr], y[tr]).predict(X[te])
    right_any = np.stack([single[c] == y for c in JUDGES], 1).any(1)
    in_family = np.where(fam_of[None, :] == fam_of[y][:, None], p_cand, 0).argmax(1)
    conf = p_cand.max(1)
    sure = conf >= 0.9

    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, cand), 1)
    wrong = cand != y
    edges = [0, 50, 65, 80, 100, 125, 155, 200, 1000]

    sf = style_features(train.text)
    by_model = sf.groupby(train.model.to_numpy()).mean().loc[labels]
    clues = {
        f: {
            "name": name,
            "top": by_model[f].sort_values(ascending=False).index[:3].tolist(),
            "bottom": by_model[f].sort_values().index[:2].tolist(),
        }
        for f, name in CLUES.items()
    }

    pool = pd.read_parquet(
        ROOT / "data" / "external" / "open_pool" / "ntl_open_pool.parquet", columns=["model"]
    ).model.value_counts()

    data = {
        "labels": labels,
        "families": fams,
        "n": n,
        "n_prompts": len(by_prompt),
        "y": y.tolist(),
        "words": words.tolist(),
        "judges": judges,
        "order": JUDGES,
        "scores": {c: np.round(scores[c], 3).ravel().tolist() for c in JUDGES},
        "submitted": SUBMITTED,
        "candidate": CANDIDATE,
        "strength": STRENGTH,
        "stats": {
            "submitted": bal(sub),
            "candidate": bal(cand),
            "boot_sd": float(np.std(boot)),
            "boot_lo": float(np.percentile(boot, 2.5)),
            "boot_hi": float(np.percentile(boot, 97.5)),
            "cand_delta_lo": float(np.percentile(delta, 2.5)),
            "cand_delta_hi": float(np.percentile(delta, 97.5)),
            "cand_wins": float(np.mean(np.array(delta) > 0)),
            "cand_without_q4": bal(blend(CANDIDATE | {"q4": 0.0}).argmax(1)),
            "stacked": bal(stacked),
            "any_judge_right": float(right_any.mean()),
            "family_fixed": bal(in_family),
            "errors": int(wrong.sum()),
            "errors_in_family": int((wrong & (fam_of[cand] == fam_of[y])).sum()),
            "sure_n": int(sure.sum()),
            "sure_conf": float(conf[sure].mean()),
            "sure_acc": float((cand == y)[sure].mean()),
        },
        "length": [
            {"lo": lo, "hi": hi, "n": int(m.sum()), "acc": float((cand == y)[m].mean())}
            for lo, hi in zip(edges[:-1], edges[1:])
            for m in [(words >= lo) & (words < hi)]
        ],
        "clues": clues,
        "pool": {"total": int(pool.sum()), "counts": pool.reindex(labels).fillna(0).astype(int).tolist()},
    }
    (GUI / "data.js").write_text(
        "window.NTL = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8"
    )

    # passages: a fair sample of 12 per author for the game, plus up to 4 more per author that the
    # blend gets wrong, so "show me a miss" always has something to show
    rng = np.random.default_rng(1)
    picked = []
    for c in range(k):
        rows = np.where(y == c)[0]
        fair = rng.choice(rows, 12, replace=False)
        miss = [i for i in rows if wrong[i] and i not in set(fair)]
        extra = rng.choice(miss, min(4, len(miss)), replace=False) if miss else []
        picked += [(int(i), 0) for i in fair] + [(int(i), 1) for i in extra]
    passages = [
        {"i": i, "prompt": train.prompt_id[i], "text": train.text[i], "extra": x}
        for i, x in sorted(picked)
    ]
    (GUI / "passages.js").write_text(
        "window.NTL_PASSAGES = " + json.dumps(passages, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    s = data["stats"]
    print(f"data.js {(GUI / 'data.js').stat().st_size / 1e6:.2f} MB, {len(passages)} passages")
    print({c: round(j["oof"], 4) for c, j in judges.items()})
    print({c: (j.get("pool_k"), j.get("hours")) for c, j in judges.items()})
    print(json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in s.items()}, indent=1))
    print("length", [(d["lo"], round(d["acc"], 3)) for d in data["length"]])
    print("recall", {m: round(cm[i, i] / cm[i].sum(), 2) for i, m in enumerate(labels)})
    print("clues", {f: c["top"] for f, c in clues.items()})


if __name__ == "__main__":
    main()
