"""A test-like validation: out-of-fold scores on train.csv subsampled to look like the hidden test.

train.csv has one passage per model for each of its 150 prompts, and everything so far was tuned on
it whole. Four public submissions show that the hidden test is built differently: the steps that
lean on prompt groups gained out-of-fold and lost on the board.

    v7  -> v10  the native expert at weight 2       +0.028 on the board
    v10 -> v9   + centring the blend in its group    -0.006
    v9  -> v8   + one model per passage in a group  -0.021

Each draw keeps a passage with a rate that depends on its model (``base`` times a log-normal
factor with spread ``sigma``, so per-model counts are unequal), then forms the groups the notebook
would see: TF-IDF groups of the kept passages, or "merged" groups that join the kept passages of
``merge`` prompts (a group that mixes prompts, so a model can appear twice). ``fit`` scores every
structure by how well it reproduces the three board deltas; ``tune`` re-tunes the blend's settings
on the structures that fit best.

Everything is out-of-fold: the SVM, LoRA and GLM members as in ``glm_blend``, the native expert
from CHEN Han's published out-of-fold logits (``native_blend``), calibrated on each draw's groups.

    uv run python -m name_that_llm.modeling.testlike fit
    uv run python -m name_that_llm.modeling.testlike tune
    uv run python -m name_that_llm.modeling.testlike joint
"""

from dataclasses import dataclass, replace
import itertools
import json

import joblib
import numpy as np
import pandas as pd
import typer

from name_that_llm.config import INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from name_that_llm.glm_member import glm_mass
from name_that_llm.modeling.glm_blend import MEMBERS, likelihood_matrix, oof_member
from name_that_llm.modeling.native_blend import NATIVE_DIR
from name_that_llm.modeling.postprocess import balanced_accuracy, sinkhorn
from name_that_llm.native_member import assign, centre, prompt_groups
from name_that_llm.native_q8 import calibrate

app = typer.Typer()

BOARD = {"expert": 0.028, "centre": -0.006, "assign": -0.021}  # v10-v7, v9-v10, v8-v9


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


@dataclass(frozen=True)
class Config:
    t_svm: float = 2.0
    lora: tuple = tuple(MEMBERS.values())
    native_w: float = 2.0
    native_cent: bool = True
    centre_f: float = 0.0
    assign: bool = False
    glm_cover: float = 0.2
    glm_w: float = 2.0
    sinkhorn: float = 0.3


V7 = Config(native_w=0.0)
V10 = Config()
V9 = replace(V10, centre_f=0.5)
V8 = replace(V9, assign=True)


class Parts:
    """Every member's out-of-fold output for train.csv, in one row order."""

    def __init__(self, glm_dir=INTERIM_DATA_DIR / "glm_ll_prefixb", svm_tag="pool2000_w1"):
        train = pd.read_csv(RAW_DATA_DIR / "train.csv")
        holds = [pd.read_parquet(INTERIM_DATA_DIR / m / "oof_logits.parquet") for m in MEMBERS]
        ids = holds[0].id.to_numpy()
        holds = [h.set_index("id").loc[ids] for h in holds]
        self.labels = [c for c in holds[0].columns if c not in ("model", "fold")]
        self.k = len(self.labels)
        fold = holds[0].fold.to_numpy()
        idx = pd.Series(np.arange(len(train)), index=train.id).loc[ids].to_numpy()
        svm = joblib.load(INTERIM_DATA_DIR / f"{svm_tag}_oof_scores.joblib")
        assert list(svm["classes"]) == self.labels
        self.svm = svm["scores"][idx]
        self.lora = np.stack([log_softmax(h[self.labels].to_numpy(float)) for h in holds])
        self.y = holds[0].model.map({m: i for i, m in enumerate(self.labels)}).to_numpy()
        self.prompt = np.unique(train.prompt_id.to_numpy()[idx], return_inverse=True)[1]
        self.text = train.text.to_numpy()[idx]
        x, _ = likelihood_matrix(glm_dir, ids)
        self.scored = np.isfinite(x).all(1)
        self.glm = np.zeros((len(ids), self.k))
        self.glm[self.scored] = oof_member(x[self.scored], self.y[self.scored], fold[self.scored], self.k, 0.05)
        manifest = json.loads((NATIVE_DIR / "manifest.json").read_text())
        assert manifest["classes"] == self.labels
        tok = np.load(NATIVE_DIR / "tokens.npz")
        oof = np.load(NATIVE_DIR / "oof_reference.npz")
        pos = pd.Series(np.arange(len(tok["ids"])), index=tok["ids"].astype(str)).loc[ids].to_numpy()
        self.plain, self.cent, self.nfold = oof["plain"][pos], oof["centered"][pos], tok["folds"][pos]
        self.bias = np.load(NATIVE_DIR / "centered_bias.npy")
        self.states = [json.loads((NATIVE_DIR / "calibration" / f"fold{f}.json").read_text())["models"]["qwen8"]
                       for f in range(5)]

    def native(self, rows, groups, use_cent):
        """The expert's calibrated log-probabilities for ``rows``, centred within ``groups``."""
        out = np.zeros((len(rows), self.k))
        nf = self.nfold[rows]
        for f in range(5):
            m = nf == f
            if m.any():  # calibrate centres within the passages it is given: one fold at a time
                out[m] = calibrate(self.plain[rows][m][None], self.cent[rows][m][None], self.bias[f : f + 1],
                                   groups[m], [self.states[f]], use_cent)[0]
        return out

    def predict(self, rows, groups, nat, c: Config):
        lp = log_softmax(self.svm[rows] * c.t_svm) + np.tensordot(np.asarray(c.lora), self.lora[:, rows], 1)
        lp = centre(lp, groups, c.centre_f) if c.centre_f else lp
        lp = lp + c.native_w * nat[c.native_cent]
        sc = self.scored[rows]
        sel = np.argsort(-np.where(sc, glm_mass(lp, self.labels), -1.0), kind="stable")[: round(c.glm_cover * len(rows))]
        sel = sel[sc[sel]]
        lp[sel] += c.glm_w * self.glm[rows][sel]
        q = np.log(np.maximum(sinkhorn(np.exp(lp - lp.max(1, keepdims=True)), c.sinkhorn), 1e-300))
        return assign(q, groups) if c.assign else q.argmax(1)


@dataclass(frozen=True)
class Structure:
    base: float  # mean share of a prompt's passages kept
    sigma: float  # spread of the per-model keep rates (log-normal)
    grouping: str  # "tfidf", or "merge<c>": groups join the kept passages of c prompts

    def draw(self, parts: Parts, rng):
        rate = np.clip(self.base * np.exp(self.sigma * rng.standard_normal(parts.k) - self.sigma**2 / 2), 0, 1)
        rows = np.flatnonzero(rng.random(len(parts.y)) < rate[parts.y])
        if self.grouping == "tfidf":
            groups = prompt_groups(parts.text[rows])
        else:
            c = int(self.grouping.removeprefix("merge"))
            block = rng.permutation(parts.prompt.max() + 1) // c
            groups = np.unique(block[parts.prompt[rows]], return_inverse=True)[1]
        return rows, groups


def run(parts, structure, configs, draws, seed):
    """Balanced accuracy of each config on each draw: an array (draws, configs)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((draws, len(configs)))
    for d in range(draws):
        rows, groups = structure.draw(parts, rng)
        nat = {u: parts.native(rows, groups, u) for u in (True, False)}
        y = parts.y[rows]
        out[d] = [balanced_accuracy(y, parts.predict(rows, groups, nat, c), parts.k) for c in configs]
    return out


STRUCTURES = [Structure(b, s, g) for b, s, g in itertools.product(
    (0.06, 0.12, 0.25, 0.5), (0.0, 0.6, 1.2), ("tfidf", "merge2", "merge4", "merge8"))]


@app.command()
def fit(draws: int = 40, seed: int = 0, tag: str = "testlike_fit"):
    """How well each structure reproduces the three board deltas."""
    parts = Parts()
    rows = []
    for st in STRUCTURES:
        r = run(parts, st, [V7, V10, V9, V8], draws, seed)
        d = {"expert": r[:, 1] - r[:, 0], "centre": r[:, 2] - r[:, 1], "assign": r[:, 3] - r[:, 2]}
        row = dict(base=st.base, sigma=st.sigma, grouping=st.grouping, v10=round(r[:, 1].mean(), 4),
                   **{f"d_{k}": round(v.mean(), 4) for k, v in d.items()},
                   **{f"se_{k}": round(v.std() / np.sqrt(draws), 4) for k, v in d.items()})
        row["miss"] = round(float(np.sqrt(sum((row[f"d_{k}"] - BOARD[k]) ** 2 for k in BOARD))), 4)
        rows.append(row)
        print(row, flush=True)
    t = pd.DataFrame(rows).sort_values("miss")
    print(t.to_string(index=False))
    (REPORTS_DIR / f"{tag}.json").write_text(json.dumps({"board": BOARD, "draws": draws, "rows": t.to_dict("records")}, indent=1))


def grid():
    """V10 and one-setting-at-a-time moves away from it."""
    moves = [("sinkhorn", (0.0, 0.15, 0.5, 0.7)), ("native_w", (1.0, 1.5, 3.0, 4.0)), ("native_cent", (False,)),
             ("glm_cover", (0.1, 0.3, 0.4)), ("glm_w", (1.0, 3.0)), ("t_svm", (1.0, 3.0, 4.0))]
    out = [("v10", V10)]
    for name, values in moves:
        out += [(f"{name}={v}", replace(V10, **{name: v})) for v in values]
    base = np.asarray(V10.lora)
    for i, m in enumerate(MEMBERS):
        for f in (0.0, 2.0):
            w = base.copy()
            w[i] *= f
            out.append((f"{m}x{f:g}", replace(V10, lora=tuple(w))))
    out.append(("lora_all x0.5", replace(V10, lora=tuple(base * 0.5))))
    out.append(("lora_all x1.5", replace(V10, lora=tuple(base * 1.5))))
    return out


@app.command()
def tune(top: int = 4, draws: int = 60, seed: int = 1, fit_tag: str = "testlike_fit", tag: str = "testlike_tune"):
    """Each setting moved alone from v10, on the structures that best reproduce the board."""
    parts = Parts()
    fits = pd.DataFrame(json.loads((REPORTS_DIR / f"{fit_tag}.json").read_text())["rows"]).sort_values("miss")
    chosen = [Structure(r.base, r.sigma, r.grouping) for r in fits.head(top).itertuples()]
    chosen.append(Structure(1.0, 0.0, "tfidf"))  # all of train.csv, as tuned so far
    names, configs = zip(*grid())
    rows = []
    for st in chosen:
        nd = 1 if st.base >= 1 and st.sigma == 0 else draws  # all of train.csv: one deterministic draw
        r = run(parts, st, configs, nd, seed)
        d = r - r[:, :1]
        for j, n in enumerate(names):
            rows.append(dict(structure=f"{st.base}/{st.sigma}/{st.grouping}", config=n, bal_acc=round(r[:, j].mean(), 4),
                             delta=round(d[:, j].mean(), 4), se=round(d[:, j].std() / np.sqrt(nd), 4)))
        print(st, "done", flush=True)
    t = pd.DataFrame(rows)
    print(t.pivot_table(index="config", columns="structure", values="delta", sort=False).round(4).to_string())
    (REPORTS_DIR / f"{tag}.json").write_text(json.dumps({"draws": draws, "rows": rows}, indent=1))


@app.command()
def joint(top: int = 4, draws: int = 60, seed: int = 2, fit_tag: str = "testlike_fit", tag: str = "testlike_joint"):
    """The settings that moved v10 the most, varied together (fresh draws, seed 2)."""
    parts = Parts()
    fits = pd.DataFrame(json.loads((REPORTS_DIR / f"{fit_tag}.json").read_text())["rows"]).sort_values("miss")
    chosen = [Structure(r.base, r.sigma, r.grouping) for r in fits.head(top).itertuples()]
    chosen.append(Structure(1.0, 0.0, "tfidf"))
    base = np.asarray(V10.lora)
    grid = [("v10", V10)]
    for scale, sk, gw, nw in itertools.product((0.25, 0.35, 0.5, 0.7, 1.0), (0.0, 0.15, 0.3), (2.0, 3.0, 4.0), (2.0, 3.0)):
        grid.append((f"lora x{scale} sk {sk} glm_w {gw} nat {nw}",
                     replace(V10, lora=tuple(base * scale), sinkhorn=sk, glm_w=gw, native_w=nw)))
    names, configs = zip(*grid)
    rows = []
    for st in chosen:
        nd = 1 if st.base >= 1 and st.sigma == 0 else draws
        r = run(parts, st, configs, nd, seed)
        d = r - r[:, :1]
        for j, n in enumerate(names):
            rows.append(dict(structure=f"{st.base}/{st.sigma}/{st.grouping}", config=n, bal_acc=round(r[:, j].mean(), 4),
                             delta=round(d[:, j].mean(), 4), se=round(d[:, j].std() / np.sqrt(nd), 4)))
        print(st, "done", flush=True)
    (REPORTS_DIR / f"{tag}.json").write_text(json.dumps({"draws": draws, "rows": rows}, indent=1))


if __name__ == "__main__":
    app()
