"""Build the self-contained Kaggle submission notebook from this package's code.

Kaggle code competitions run a notebook offline, so the notebook cannot import this package. Instead,
this script copies the feature code (``features.py``) and the balancing functions
(``postprocess.py``) into the notebook verbatim, so the notebook and the local pipeline stay in sync.
Rebuild it after changing either file.

    uv run python -m name_that_llm.kaggle_notebook

The notebook retrains the pool model inside Kaggle (~15 min on CPU) from two inputs:
  * the competition data (/kaggle/input/competitions/name-that-llm/)
  * the public Open Pool dataset (crtproduction/name-that-llm-open-pool), attached as an input
Run outside Kaggle, it reads the same files from this project's data/ folders instead.

``--glm-cover`` adds the GLM-5.3 likelihood member (``glm_member.py``, measured by
``modeling/glm_blend.py``): after the blend, the notebook scores the test passages the blend most
suspects of being GLM under the real GLM-5.3 and GLM-5.3-Flash, as many as the cover and the time
left allow. It writes the blend without the member first, so a failed or late GLM stage in the
hidden-test rerun still leaves a submission. ``--glm-prefix`` picks the prompt prefix the GLM models
read before each passage; the train features must have been computed with the same one.
``--push-dir`` holds a pulled copy of the Kaggle
notebook (``kaggle kernels pull <id> -p <dir> -m``); the build replaces its notebook and adds the
GLM inputs to its ``kernel-metadata.json``.
"""

import inspect
import json
from pathlib import Path

import nbformat as nbf
import typer

from name_that_llm import features, glm_member, lora, native_member, native_q8
from name_that_llm.config import PROJ_ROOT
from name_that_llm.kaggle_glm_features import (
    BINARY_NOTEBOOK,
    MAX_TEXT_TOKENS,
    PREFIX,
    PREFIX_USER_TURN,
    SEGMENT_DATASETS,
    SEGMENTS,
    pipeline_source,
)
from name_that_llm.modeling import postprocess

app = typer.Typer()

NOTEBOOK_PATH = PROJ_ROOT / "notebooks" / "kaggle_submission.ipynb"


def module_body(module) -> str:
    """A module's source without its leading docstring."""
    src = inspect.getsource(module)
    if src.lstrip().startswith('"""'):
        end = src.index('"""', src.index('"""') + 3) + 3
        src = src[end:]
    return src.strip() + "\n"


# Open Pool draw shared by both notebooks: same as name_that_llm/dataset.py
POOL_SAMPLE_CODE = (
    "pool = pd.read_parquet(POOL, columns=['text', 'model', 'match_weight'])\n\n"
    "# same draw as name_that_llm/dataset.py: weighted sampling without replacement\n"
    "# (Efraimidis-Spirakis) by match_weight, the same number of passages per label\n"
    "parts = []\n"
    "for label, d in pool.groupby('model'):\n"
    "    rng = np.random.default_rng([POOL_SEED, len(parts)])\n"
    "    keys = rng.random(len(d)) ** (1.0 / d.match_weight.clip(lower=1e-6).to_numpy())\n"
    "    parts.append(d.iloc[np.argsort(-keys)[:min(N_PER_LABEL, len(d))]])\n"
    "pool_sample = pd.concat(parts, ignore_index=True)\n"
    "del pool\n"
)


LORA_SETUP_CODE = (
    "\n# LoRA members: each training notebook's output folder (lora_out*) + its base model.\n"
    "# Checkpoints saved before meta.json recorded the base model are Qwen3-4B-Base.\n"
    "DEFAULT_BASE = {'model_type': 'qwen3', 'hidden_size': 2560, 'num_hidden_layers': 36}\n\n\n"
    "def find_base(spec):\n"
    "    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):\n"
    "        try:\n"
    "            c = json.load(open(cfg))\n"
    "        except Exception:\n"
    "            continue\n"
    "        if all(c.get(k) == v for k, v in spec.items()):\n"
    "            return os.path.dirname(cfg)\n"
    "    raise FileNotFoundError(f'attach the base model {spec}')\n\n\n"
    "if ON_KAGGLE:\n"
    "    # a member uploaded as a dataset sits in a folder named after the dataset, with dashes\n"
    "    found = {os.path.basename(os.path.dirname(p)).replace('-', '_'): os.path.dirname(p)\n"
    "             for p in glob.glob('/kaggle/input/**/lora_classifier.safetensors', "
    "recursive=True)}\n"
    "else:  # NTL_LORA_DIRS='name=path,...'; one base model (NTL_LORA_BASE) for all of them\n"
    "    found = dict(kv.split('=', 1) for kv in os.environ.get(\n"
    "        'NTL_LORA_DIRS', 'lora_out=../data/interim/lora4b').split(','))\n"
    "missing = sorted(set(LORA_WEIGHTS) - set(found))\n"
    "assert not missing, f'Attach the LoRA training notebook outputs {missing}; found {found}'\n"
    "LORA_DIRS = {name: found[name] for name in LORA_WEIGHTS}\n"
    "# a 5-fold member keeps its fold adapters in fold0..fold4 next to its full fit; average them\n"
    "LORA_ADAPTERS = {name: [d] + sorted(glob.glob(os.path.join(d, 'fold[0-9]')))\n"
    "                 for name, d in LORA_DIRS.items()}\n"
    "for name, d in LORA_DIRS.items():\n"
    "    print(f'LoRA member {name} (weight {LORA_WEIGHTS[name]}): {d}, '\n"
    "          f'{len(LORA_ADAPTERS[name])} adapter(s)')\n"
)

# ---------------------------------------------------------------- GLM-5.3 likelihood member

# train.csv's likelihood features for each prompt prefix: (prefix, kind of input, slug under the
# notebook owner's account)
#  * max: our notebook's output (kaggle_glm_features.py, full mode)
#  * user-turn: José Padilla's estopadilla/ntl-glm-prefix-ab outputs (versions 1 and 2, 800 + 1,899
#    passages), in one private dataset
GLM_FEATURE_SETS = {
    "max": (PREFIX, "kernel_sources", "ntl-glm-likelihood-train-features"),
    "user-turn": (PREFIX_USER_TURN, "dataset_sources", "ntl-glm-userturn-train-features"),
}
# cost of scoring on 2x T4, both models: fitted on the train runs of 40 passages (1,512 s) and of
# 2,699 passages (14,144 s); the fixed part is mostly loading the 25 segments
GLM_FIXED_S, GLM_PER_PASSAGE_S = 1330, 4.75

GLM_SETUP_CODE = """
import shutil
import subprocess
import sys
from pathlib import Path

GLM_COVER, GLM_WEIGHT, GLM_C = {cover}, {weight}, {C}
GLM_PREFIX = {prefix!r}
GLM_MAX_TEXT_TOKENS = {max_tokens}
GLM_SEGMENTS = {segments}
GLM_TRAIN_FEATURES = {train_features!r}
GLM_FIXED_S, GLM_PER_PASSAGE_S, GLM_SAFETY = {fixed}, {per_passage}, 1.25
DEADLINE_S = 8.5 * 3600  # after T_START: Kaggle stops a submission at 9 h
GLM_CHECK_ROWS = 8  # interactive runs: train passages scored again, against their stored features
RERUN = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))
GLM_WORK = Path('/kaggle/temp/glm_ll')  # pipeline code, passage list, state between segments
GLM_BIN = Path('/kaggle/temp/glm_ll_bin')
GLM_OUT = Path('/kaggle/working/glm_ll')  # features and run logs of the passages scored
for d in (GLM_WORK, GLM_BIN, GLM_OUT):
    d.mkdir(parents=True, exist_ok=True)
"""

GLM_STAGE_CODE = """
def glm_inputs():
    # segments (exact sizes), the ntl-segrun binary, and train.csv's stored features
    flash, glm53 = [], []
    for ref, files in GLM_SEGMENTS.items():
        for name, size in files:
            hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
            assert hits, f'{name} missing: attach {ref}'
            assert os.path.getsize(hits[0]) == size, f'{hits[0]}: not {size} bytes'
            (flash if name.startswith('flash_') else glm53).append(hits[0])
    bins = sorted(glob.glob('/kaggle/input/**/ntl-bin/ntl-segrun', recursive=True))
    assert bins, 'attach the output of estopadilla/ntl-glm-likelihood-train (ntl-bin/)'
    stored = {}
    for tag in GLM_TAGS:
        hits = sorted(glob.glob(f'/kaggle/input/**/{GLM_TRAIN_FEATURES}/**/{tag}_features.csv',
                                recursive=True))
        assert hits, f'attach the output of {GLM_TRAIN_FEATURES} ({tag}_features.csv)'
        stored[tag] = pd.read_csv(hits[0], dtype={'id': str}).set_index('id')
    return sorted(flash), sorted(glm53), Path(bins[0]).parent, stored


def run_glm(csv, flash_segs, glm53_segs, bin_dir):
    # both models over the passages in csv; each run stops at the deadline
    shutil.copyfile(bin_dir / 'ntl-segrun', GLM_BIN / 'ntl-segrun')
    (GLM_BIN / 'ntl-segrun').chmod(0o755)
    # the binary was built against CUDA 13; its runtime ships next to it
    env = dict(os.environ, LD_LIBRARY_PATH=f"{bin_dir / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    for tag, segs in (('flash', flash_segs), ('glm53', glm53_segs)):
        cmd = [sys.executable, str(GLM_WORK / 'run_pipeline.py'), '--bin', str(GLM_BIN),
               '--segs', json.dumps(segs), '--csv', str(csv), '--out', str(GLM_OUT), '--tag', tag,
               '--work', str(GLM_WORK / 'state'), '--seq-batch', '8', '--state-dtype', 'f32',
               '--max-text-tokens', str(GLM_MAX_TEXT_TOKENS), '--prefix', GLM_PREFIX]
        t = time.time()
        # its own process group, so a timeout also stops ntl-segrun under it
        p = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            code = p.wait(timeout=max(60.0, DEADLINE_S - (time.time() - T_START)))
        except BaseException:
            os.killpg(p.pid, 9)
            p.wait()
            raise
        assert code == 0, f'{tag}: run_pipeline failed ({code}), see {GLM_OUT}/{tag}_run.log'
        print(f'{tag}: {time.time() - t:.0f} s', flush=True)
    return {tag: pd.read_csv(GLM_OUT / f'{tag}_features.csv', dtype={'id': str}).set_index('id')
            for tag in GLM_TAGS}


def glm_stage(logp):
    # logp plus the member's log-probabilities on the passages scored (None: no time to score any)
    t0 = time.time()
    flash_segs, glm53_segs, bin_dir, stored = glm_inputs()
    xtr = likelihood_frame(stored['flash'], stored['glm53'], train.id).to_numpy()
    ok = np.isfinite(xtr).all(axis=1)
    assert ok.mean() > 0.99, f'{(~ok).sum()} train passages have no stored likelihood features'
    member = fit_member(xtr[ok], train.model.to_numpy()[ok], GLM_C)
    assert list(member.classes_) == list(CLASSES)

    # the passages the blend most suspects of being GLM, as many as the cover and the time allow
    left = DEADLINE_S - (time.time() - T_START)
    n_cover = max(1, round(GLM_COVER * len(test)))
    n_time = int((left / GLM_SAFETY - GLM_FIXED_S) / GLM_PER_PASSAGE_S)
    n = min(n_cover, n_time)
    print(f'GLM stage: {left / 3600:.2f} h left; scoring {max(n, 0)} of {len(test)} test passages '
          f'(cover allows {n_cover}, time {n_time})', flush=True)
    if n < 1:
        return None
    sel = np.argsort(-glm_mass(logp, CLASSES), kind='stable')[:n]
    # ids by position, so that no id is read back as a number
    texts = pd.DataFrame({'id': [f'test:{i}' for i in range(n)], 'text': test.text.to_numpy()[sel]})
    k = 0 if RERUN else GLM_CHECK_ROWS
    check = pd.DataFrame({'id': [f'train:{i}' for i in range(k)], 'text': train.text.to_numpy()[:k]})
    csv = GLM_WORK / 'texts.csv'
    pd.concat([texts, check], ignore_index=True).to_csv(csv, index=False)
    if 'torch' in globals() and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    print(subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv'],
                         capture_output=True, text=True).stdout, flush=True)
    scored = run_glm(csv, flash_segs, glm53_segs, bin_dir)

    if len(check):
        # the same passages as in train.csv's run: same features, same member prediction
        a = likelihood_frame(scored['flash'], scored['glm53'], check.id)
        b = likelihood_frame(stored['flash'], stored['glm53'], train.id.iloc[:len(check)])
        gap = (a.to_numpy() - b.to_numpy())
        print('check passages, largest gap to the stored features:',
              {c: float(np.abs(gap[:, j]).max()) for j, c in enumerate(a.columns) if c.endswith('lp_mean')},
              f'all columns {np.nanmax(np.abs(gap)):.4f}; same member prediction for '
              f'{(member.predict(a.to_numpy()) == member.predict(b.to_numpy())).sum()} of {len(check)}', flush=True)
        assert np.isfinite(gap).all() and np.abs(gap[:, list(a.columns).index('flash:lp_mean')]).max() < 0.05

    x = likelihood_frame(scored['flash'], scored['glm53'], texts.id).to_numpy()
    ok = np.isfinite(x).all(axis=1)
    out = logp.copy()
    out[sel[ok]] += GLM_WEIGHT * member.predict_log_proba(x[ok])
    print(f'GLM stage: {ok.sum()} of {n} passages scored in {time.time() - t0:.0f} s', flush=True)
    return out


logp_glm = None
try:
    logp_glm = glm_stage(logp)
except Exception as e:
    if not RERUN:
        raise
    print(f'GLM stage failed; the submission stays the blend without it: {e!r}', flush=True)
if logp_glm is not None:
    before = submission.model.to_numpy()
    submission = write_submission(logp_glm)
    print(f'the GLM member changed {(submission.model.to_numpy() != before).sum()} predictions')
    print(submission.model.value_counts().to_string())
"""

GLM_BLEND_CODE = """
def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


logp = log_softmax(svm_scores * T_SVM)
for name, w_lora in LORA_WEIGHTS.items():
    logp = logp + w_lora * log_softmax(lora_logits[name])
{centre}

def write_submission(logp):
    q = sinkhorn(softmax(logp, 1.0), BALANCE_STRENGTH)
    pred = CLASSES[{decide}]
    submission = pd.DataFrame({{'id': test.id, 'model': pred}})
    assert submission.id.is_unique and len(submission) == len(test)
    assert submission.model.notna().all() and submission.model.isin(LABELS).all()
    submission.to_parquet(OUT, index=False)
    return submission


# the blend without the GLM member, written first: it stays the submission if the GLM stage fails
submission = write_submission(logp)
print(submission.model.value_counts().to_string())
"""


def blend_code(groups: dict | None) -> str:
    """The blend cell; with ``groups``, its log-probabilities are centred within the test prompt
    groups and the decision gives the passages of a group different models."""
    if not groups:
        return GLM_BLEND_CODE.format(centre="", decide="q.argmax(axis=1)")
    return GLM_BLEND_CODE.format(
        centre="logp = centre(logp, GROUPS, GROUP_CENTRE)\n",
        decide="assign(np.log(np.maximum(q, 1e-300)), GROUPS) if GROUP_ASSIGN else q.argmax(axis=1)",
    )


# ------------------------------------------------- prompt groups and the native Qwen3-Embedding expert

NATIVE_ASSETS = "lightningv08/ntl-olmo2-native24-final-inference-assets"
NATIVE_MODEL = "qwen-lm/qwen-3-embedding/Transformers/8b/1"

GROUP_SETUP_CODE = """
GROUP_CENTRE, GROUP_ASSIGN = {centre}, {assign}
t0 = time.time()
GROUPS = prompt_groups(test.text.fillna('').astype(str).to_numpy())
_size = np.bincount(GROUPS)
print(f'{{len(_size)}} prompt groups over {{len(test)}} test passages in {{time.time() - t0:.0f}} s; '
      f'passages in groups of 2..18: {{eligible(GROUPS).mean():.3f}}; group sizes {{np.bincount(_size).tolist()}}')
"""

NATIVE_SETUP_CODE = """
NATIVE_W = {weight}
# one fold over 4,000 passages on 2x T4 (estimate; interactive runs log the measured speed)
NATIVE_FOLD_S = {fold_s}
NATIVE_CHECK_ROWS = 400  # interactive runs: train passages scored for the probe check and the timing
NATIVE_MIN_AGREE = 0.35  # least agreement with the blend's decisions before the expert is trusted
NATIVE_WORK = Path('/kaggle/temp/native')
NATIVE_WORK.mkdir(parents=True, exist_ok=True)
"""

NATIVE_STAGE_CODE = """
def native_inputs():
    hits = sorted(glob.glob('/kaggle/input/**/ntl-native-q8-multidepth-fivefold-0679/manifest.json',
                            recursive=True))
    assert hits, 'attach {assets}'
    assets = os.path.dirname(hits[0])
    for k in range(5):
        assert os.path.exists(f'{{assets}}/fold{{k}}/trainables.safetensors'), k
        assert os.path.exists(f'{{assets}}/calibration/fold{{k}}.json'), k
    bases = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/model.safetensors.index.json',
                                                        recursive=True)
                   if 'qwen-3-embedding' in p.lower() and not p.startswith(assets))
    assert bases, 'attach the Kaggle model {model}'
    return assets, bases[0]


def native_stage(logp):
    # logp plus NATIVE_W times the expert's calibrated log-probabilities (None: no time to run it)
    import native_q8
    t0 = time.time()
    assets, base = native_inputs()
    # the GLM stage keeps the time it needs for its full cover
    reserve = GLM_SAFETY * (GLM_FIXED_S + GLM_PER_PASSAGE_S * max(1, round(GLM_COVER * len(test))))
    deadline = T_START + DEADLINE_S - reserve
    query = pd.DataFrame({{'id': [f'test:{{i}}' for i in range(len(test))], 'text': test.text.fillna('').astype(str)}})
    probes = None
    if not RERUN:
        # CHEN Han's numeric probes: 24 fold-0 train passages and their published plain logits
        z = np.load(f'{{assets}}/fold0/numeric_probes.npz')
        tok_ids = np.load(f'{{assets}}/tokens.npz')['ids'].astype(str)
        probe_ids = tok_ids[z['rows']]
        extra = train.id[~train.id.astype(str).isin(probe_ids)].astype(str).to_numpy()[:NATIVE_CHECK_ROWS - len(probe_ids)]
        by_id = train.set_index(train.id.astype(str)).text
        check_ids = np.concatenate([probe_ids, extra])
        query = pd.concat([query, pd.DataFrame({{'id': [f'train:{{i}}' for i in check_ids], 'text': by_id.loc[check_ids].to_numpy()}})],
                          ignore_index=True)
        probes = z['plain']
    n_fold_s = NATIVE_FOLD_S * len(query) / 4000
    left = deadline - time.time()
    print(f'native stage: {{left / 3600:.2f}} h before the GLM reserve; {{len(query)}} passages, '
          f'about {{n_fold_s / 60:.0f}} min a fold', flush=True)
    if left < 600 + n_fold_s:
        return None
    csv = NATIVE_WORK / 'query.csv'
    query.to_csv(csv, index=False)
    out = NATIVE_WORK / 'native.npz'
    if out.exists():
        out.unlink()
    if 'torch' in globals() and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    cmd = [sys.executable, '-u', str(NATIVE_WORK / 'native_q8.py'), '--base', base, '--assets', assets,
           '--query', str(csv), '--out', str(out), '--work', str(NATIVE_WORK / 'work'),
           '--deadline', str(deadline - n_fold_s)]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        code = p.wait(timeout=max(60.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        p.wait()
        code = 'killed at the deadline'
    except BaseException:
        os.killpg(p.pid, 9)
        p.wait()
        raise
    assert out.exists(), f'the native worker finished no fold ({{code}})'
    z = np.load(out, allow_pickle=False)
    folds = z['folds'].tolist()
    assert z['ids'].astype(str).tolist() == query.id.tolist(), 'native row order differs'
    labels = z['labels'].astype(str).tolist()
    order = [labels.index(c) for c in CLASSES]
    plain, cent = z['plain'][:, :, order], z['centered'][:, :, order]
    nt = len(test)
    print(f'native worker ({{code}}): folds {{folds}} in {{time.time() - t0:.0f}} s, '
          f'{{int((~z["finite"]).sum())}} passages without finite features', flush=True)
    if probes is not None and 0 in folds:
        got = plain[folds.index(0), nt:nt + len(probes)]
        want = probes[:, order]
        same = (got.argmax(1) == want.argmax(1)).sum()
        print(f'probe check: largest logit gap {{np.abs(got - want).max():.4f}}, same top label for '
              f'{{same}} of {{len(want)}}', flush=True)
        assert same >= len(want) - 1 and np.abs(got - want).max() < 0.5
    states = [json.load(open(f'{{assets}}/calibration/fold{{k}}.json'))['models']['qwen8'] for k in folds]
    lp = native_q8.calibrate(plain[:, :nt], cent[:, :nt], z['centered_bias'][:, order], GROUPS, states)
    agree = [float((lp[i].argmax(1) == logp.argmax(1)).mean()) for i in range(len(folds))]
    print('native: agreement with the blend per fold', [round(a, 3) for a in agree], flush=True)
    if min(agree) < NATIVE_MIN_AGREE and nt >= 100:
        raise RuntimeError(f'native agreement {{min(agree):.3f}} below {{NATIVE_MIN_AGREE}}')
    return logp + NATIVE_W * np.where(z['finite'][:nt, None], lp.mean(0), 0.0)


sys.path.insert(0, str(NATIVE_WORK))
try:
    logp_native = native_stage(logp)
except Exception as e:
    if not RERUN:
        raise
    logp_native = None
    print(f'native stage failed; the blend goes on without it: {{e!r}}', flush=True)
if logp_native is None:
    print('native stage skipped: not enough time before the GLM reserve', flush=True)
else:
    before = submission.model.to_numpy()
    logp = logp_native
    submission = write_submission(logp)
    print(f'the native expert changed {{(submission.model.to_numpy() != before).sum()}} predictions')
"""


def native_cells(groups: dict) -> list:
    """The native Qwen3-Embedding-8B expert: settings, its worker, then the stage."""
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md(
            "## Native Qwen3-Embedding-8B expert\n\n"
            "CHEN Han's (concyclics) multi-depth Qwen3-Embedding-8B LoRA study, run as in the 0.716 "
            "public notebook: the five fold states and their calibration from "
            "lightningv08's `ntl-olmo2-native24-final-inference-assets`, the worker `native_q8.py` "
            "copied unchanged. Its centred head reads each passage against its prompt group. The "
            "worker runs in its own process before the GLM stage and stops in time to leave that "
            f"stage its full cover; its log-probabilities are added at weight {groups['native_w']}."
        ),
        code(NATIVE_SETUP_CODE.format(weight=groups["native_w"], fold_s=NATIVE_FOLD_S).strip() + "\n"),
        code("%%writefile /kaggle/temp/native/native_q8.py\n" + inspect.getsource(native_q8)),
        code(NATIVE_STAGE_CODE.format(assets=NATIVE_ASSETS, model=NATIVE_MODEL).strip() + "\n"),
    ]


# one fold over 4,000 passages, guessed from the public notebooks' total (35-55 min for five folds)
NATIVE_FOLD_S = 900


def glm_cells(glm: dict) -> list:
    """The GLM-5.3 stage: settings, the member's code, the scoring pipeline, then the stage."""
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    prefix, _, train_features = GLM_FEATURE_SETS[glm["prefix"]]
    setup = GLM_SETUP_CODE.format(
        cover=glm["cover"],
        weight=glm["weight"],
        C=glm["C"],
        prefix=prefix,
        max_tokens=MAX_TEXT_TOKENS,
        segments=json.dumps(SEGMENTS),
        train_features=train_features,
        fixed=GLM_FIXED_S,
        per_passage=GLM_PER_PASSAGE_S,
    )
    cells = [
        md(
            "## GLM-5.3 likelihood member\n\n"
            "Scores the test passages the blend most suspects of being GLM (a share `GLM_COVER` of "
            "the test set, fewer if time is short) under the real GLM-5.3 and GLM-5.3-Flash, and adds "
            "a logistic regression on their likelihood features, fitted on train.csv's, to the "
            "blend. The scoring pipeline (`ntl-segrun`, `run_pipeline.py`, `features.py`, "
            "`segio.py`) is José Padilla's, from *0.689 | Real GLM-5.3 Likelihood on 2xT4*; the "
            "segment GGUFs are jamesliebel's mirrors of jinghengwu's quantization of Z.ai's open "
            "weights. The member's code is copied from `name_that_llm/glm_member.py`."
            + (
                "\n\nThe models read each passage after the user-turn prefix of José Padilla's "
                "*0.716 | Robust GLM-5.3 Likelihood + User Prefix* (prvsiyan's idea): a request for "
                "one paragraph of about as many words as the passage has. The train features are "
                "the outputs of his `ntl-glm-prefix-ab` notebook, scored with the same prefix."
                if glm["prefix"] == "user-turn"
                else ""
            )
        ),
        code(setup.strip() + "\n"),
        code(module_body(glm_member)),
    ]
    for name in ("segio", "features", "run_pipeline"):
        cells.append(code(f"%%writefile /kaggle/temp/glm_ll/{name}.py\n" + pipeline_source(name)))
    cells.append(code(GLM_STAGE_CODE.strip() + "\n"))
    return cells


def build(
    lora_weights: dict[str, float] | None = None,
    t_svm: float = 1.0,
    glm: dict | None = None,
    groups: dict | None = None,
    balance: float = 0.3,
) -> nbf.NotebookNode:
    """The submission notebook; ``lora_weights`` ({checkpoint folder: weight}) adds LoRA members,
    ``glm`` ({cover, weight, C, prefix}) the GLM-5.3 likelihood member after them, ``groups``
    ({centre, assign, native_w}) the test prompt groups and the native expert before it; ``balance``
    is the strength of the Sinkhorn class balancing at the end."""
    lora_weights = {k: w for k, w in (lora_weights or {}).items() if w > 0}
    use_lora = bool(lora_weights)
    assert use_lora or not glm, "the GLM stage ranks passages by the LoRA blend"
    assert glm or not groups, "the prompt-group steps are built on the GLM notebook"
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            "# Name That LLM: TF-IDF + stylometry LinearSVC with a style-matched Open Pool sample"
            + (f" + {len(lora_weights)} two-stage LoRA classifiers" if use_lora else "")
            + "\n\nGenerated by `name_that_llm/kaggle_notebook.py`; do not edit by hand.\n\n"
            "* **Model:** character 1-5-grams + word 1-2-grams + 16 punctuation/style rates, then LinearSVC.\n"
            "* **Data:** `train.csv` plus 2,000 passages per model from the public "
            "[Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool), "
            "sampled by its `match_weight` so typography and length match the real passages.\n"
            + (
                "* **LoRA members:** checkpoints from the training notebooks, blended by adding "
                "log-probabilities with weights "
                + ", ".join(f"`{k}` {w}" for k, w in lora_weights.items())
                + ".\n"
                if use_lora
                else ""
            )
            + (
                f"* **GLM-5.3 likelihood member:** the {glm['cover']:.0%} of test passages the "
                "blend most suspects of being GLM, scored under the real GLM-5.3 and "
                f"GLM-5.3-Flash; weight {glm['weight']}.\n"
                if glm
                else ""
            )
            + (
                "* **Prompt groups:** test passages grouped by TF-IDF similarity; the blend is "
                f"centred within its group (share {groups['centre']})"
                + (
                    f", CHEN Han's native Qwen3-Embedding-8B expert is added at weight "
                    f"{groups['native_w']}"
                    if groups["native_w"] > 0
                    else ""
                )
                + (", and a group's passages get different models" if groups["assign"] else "")
                + ".\n"
                if groups
                else ""
            )
            + "* **Post-processing:** partial Sinkhorn class balancing over the test set.\n"
            "* **Local grouped CV** (5 folds by `prompt_id`): balanced accuracy 0.451"
            + (" for the SVM alone.\n\n" if use_lora else ".\n\n")
            + (
                "Needs GPU T4 x2 and no internet; the GLM stage takes about 25 minutes plus "
                "5 s per passage scored, as many as fit before 8.5 h."
                if glm
                else "Needs GPU T4 and no internet; about 15 minutes for the SVM plus a few per LoRA."
                if use_lora
                else "Needs no GPU and no internet; trains in about 15 minutes."
            )
        ),
        code(
            "import gc\nimport glob\nimport json\nimport os\nimport re\nimport time\n\n"
            + (
                "T_START = time.time()  # the GLM stage stops in time for the 9 h limit\n"
                if glm
                else ""
            )
            + "import numpy as np\nimport pandas as pd\n"
            "from sklearn.pipeline import Pipeline\nfrom sklearn.svm import LinearSVC\n\n"
            "ON_KAGGLE = os.path.exists('/kaggle/input')\n"
            "if ON_KAGGLE:\n"
            "    # find the inputs wherever Kaggle mounts them; list /kaggle/input if one is missing\n"
            "    attached = sorted(glob.glob('/kaggle/input/*/*', recursive=False))\n"
            "    print('attached inputs:', attached)\n"
            "    comps = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train.csv', "
            "recursive=True)\n"
            "             if os.path.exists(os.path.join(os.path.dirname(p), 'test.csv'))]\n"
            "    assert comps, f'Attach the name-that-llm competition data; found {attached}'\n"
            "    COMP = comps[0]\n"
            "    pools = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)\n"
            "    assert pools, f'Attach crtproduction/name-that-llm-open-pool; found {attached}'\n"
            "    POOL = pools[0]\n"
            "    OUT = 'submission.parquet'\n"
            "else:  # running from this project's notebooks/ folder\n"
            "    COMP = '../data/raw'\n"
            "    POOL = '../data/external/open_pool/ntl_open_pool.parquet'\n"
            "    OUT = '../data/processed/submission_from_notebook.parquet'\n\n"
            "# settings of the committed local model (reports/pool2000_w1_cv_metrics.json)\n"
            "N_PER_LABEL, POOL_SEED, POOL_WEIGHT = 2000, 0, 1.0\n"
            "C, STYLE_WEIGHT = 0.5, 0.2\n"
            f"BALANCE_TEMPERATURE, BALANCE_STRENGTH = 1.0, {balance}\n"
            f"USE_LORA, LORA_WEIGHTS, T_SVM = {use_lora}, {lora_weights!r}, {t_svm}\n"
            "print('competition data:', COMP)\nprint('open pool:', POOL)"
            + (LORA_SETUP_CODE if use_lora else "")
        ),
        md("## Features (copied from `name_that_llm/features.py`)"),
        code(module_body(features)),
        md("## Class balancing (copied from `name_that_llm/modeling/postprocess.py`)"),
        code(
            "\n\n".join(inspect.getsource(f) for f in (postprocess.softmax, postprocess.sinkhorn))
        ),
        md("## Data: train.csv + a style-matched Open Pool sample"),
        code(
            "train = pd.read_csv(f'{COMP}/train.csv')\n"
            "test = pd.read_csv(f'{COMP}/test.csv')\n"
            + POOL_SAMPLE_CODE
            + "\nLABELS = sorted(train.model.unique())\n"
            "assert len(LABELS) == 18 and set(pool_sample.model) <= set(LABELS)\n"
            "print(f'train {len(train)}, pool sample {len(pool_sample)}, test {len(test)}')"
        ),
        md("## Train"),
        code(
            "X = pd.concat([train.text, pool_sample.text], ignore_index=True)\n"
            "y = np.concatenate([train.model.to_numpy(), pool_sample.model.to_numpy()])\n"
            "w = np.concatenate([np.ones(len(train)), np.full(len(pool_sample), POOL_WEIGHT)])\n\n"
            "model = Pipeline([\n"
            "    ('features', build_features(STYLE_WEIGHT)),\n"
            "    ('clf', LinearSVC(C=C, class_weight='balanced', max_iter=5000)),\n"
            "])\n"
            "t0 = time.time()\n"
            "model.fit(X, y, clf__sample_weight=w)\n"
            "print(f'trained in {time.time() - t0:.0f}s; in-sample balanced acc on train.csv: '\n"
            "      f'{np.mean([np.mean(model.predict(train.text[train.model == m]) == m) for m in LABELS]):.3f}')"
        ),
        md("## SVM scores for the test set"),
        code(
            "svm_scores = model.decision_function(test.text)\n"
            "CLASSES = np.asarray(model.classes_)\n"
            "del model, X, y, w, pool_sample\n"
            "gc.collect()"
        ),
    ]
    if use_lora:
        cells += [
            md("## LoRA classifier (copied from `name_that_llm/lora.py`)"),
            code(module_body(lora)),
            md("## LoRA logits for the test set, one member at a time"),
            code(
                "if torch.cuda.is_available():\n"
                "    device = torch.device('cuda:0')\n"
                "    amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 "
                "else torch.float16\n"
                "else:\n"
                "    device, amp = torch.device('cpu'), torch.float32\n"
                "def log_softmax(z):\n"
                "    z = z - z.max(axis=1, keepdims=True)\n"
                "    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))\n\n\n"
                "lora_logits = {}  # per member: the mean log-probabilities of its adapters\n"
                "for name, lora_dir in LORA_DIRS.items():\n"
                "    meta = json.load(open(f'{lora_dir}/meta.json'))\n"
                "    assert meta['labels'] == list(CLASSES), f'{name}: label order differs from "
                "the SVM'\n"
                "    cfg = meta['cfg']\n"
                "    base = find_base(cfg.get('base', DEFAULT_BASE)) if ON_KAGGLE else "
                "os.environ.get('NTL_LORA_BASE', '../data/interim/tiny_qwen3')\n"
                "    t0 = time.time()\n"
                "    tok = load_tokenizer(base)\n"
                "    # a trunk too large for one GPU (Qwen3-14B) is split over both\n"
                "    device2 = torch.device('cuda:1') if torch.cuda.device_count() > 1 else None\n"
                "    trunk = load_trunk(base, cfg['keep'], amp, device, device2=device2)\n"
                "    lora_model = LoRAClassifier(trunk, cfg['layers'], len(CLASSES), cfg['r'], "
                "cfg['alpha'], 0.0, ckpt=False)\n"
                "    logps = []\n"
                "    for adapter in LORA_ADAPTERS[name]:\n"
                "        assert json.load(open(f'{adapter}/meta.json'))['labels'] == list(CLASSES)\n"
                "        load_checkpoint(lora_model, adapter)\n"
                "        z = predict_logits(lora_model, tok, test.text, bs=16, "
                "max_len=cfg['max_len'], amp_dtype=amp)\n"
                "        assert np.isfinite(z).all(), f'{adapter}: non-finite LoRA logits'\n"
                "        logps.append(log_softmax(z))\n"
                "    lora_logits[name] = np.mean(logps, axis=0)\n"
                "    print(f'{name}: {len(logps)} adapter(s) in {time.time() - t0:.0f}s on {device} "
                "from {base}; '\n"
                '          f"held-out balanced accuracy at training: fold 0 '
                "{meta.get('holdout_balanced_accuracy', float('nan')):.4f}, \"\n"
                "          f\"all folds {meta.get('oof_balanced_accuracy', float('nan')):.4f}\")\n"
                "    del lora_model, trunk  # free the GPU for the next member\n"
                "    gc.collect()\n"
                "    if device.type == 'cuda':\n"
                "        torch.cuda.empty_cache()"
            ),
        ]
    if glm:
        if groups:
            cells += [
                md(
                    "## Test prompt groups (copied from `name_that_llm/native_member.py`)\n\n"
                    "Every train.csv prompt has one passage from each model. The test passages are "
                    "grouped by TF-IDF similarity (`capped_agglo` from the 0.716 public notebook); "
                    f"the blend's log-probabilities lose {groups['centre']} of their group mean"
                    + (", and within a group no two passages get the same model" if groups["assign"] else "")
                    + "."
                ),
                code(module_body(native_member)),
                code(GROUP_SETUP_CODE.format(centre=groups["centre"], assign=groups["assign"]).strip() + "\n"),
            ]
        glm_part = glm_cells(glm)
        cells += [
            md("## Blend, balance, write `submission.parquet`"),
            code(blend_code(groups).strip() + "\n"),
            *glm_part[:-1],
            *(native_cells(groups) if groups and groups["native_w"] > 0 else []),
            glm_part[-1],
        ]
        return nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3",
                    "language": "python",
                },
                "language_info": {"name": "python"},
            },
        )
    cells += [
        md("## Blend, balance, write `submission.parquet`"),
        code(
            "def log_softmax(z):\n"
            "    z = z - z.max(axis=1, keepdims=True)\n"
            "    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))\n\n\n"
            "if USE_LORA:\n"
            "    logp = log_softmax(svm_scores * T_SVM)\n"
            "    for name, w_lora in LORA_WEIGHTS.items():\n"
            "        logp = logp + w_lora * log_softmax(lora_logits[name])\n"
            "    proba = softmax(logp, 1.0)\n"
            "else:\n"
            "    proba = softmax(svm_scores, BALANCE_TEMPERATURE)\n"
            "pred = CLASSES[sinkhorn(proba, BALANCE_STRENGTH).argmax(axis=1)]\n"
            "submission = pd.DataFrame({'id': test.id, 'model': pred})\n\n"
            "assert submission.id.is_unique and len(submission) == len(test)\n"
            "assert submission.model.notna().all() and submission.model.isin(LABELS).all()\n"
            "submission.to_parquet(OUT, index=False)\n"
            "print(submission.model.value_counts().to_string())\n"
            "submission.head()"
        ),
    ]
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
    )


LORA_OPTION = typer.Option(
    [],
    help="A LoRA member as checkpoint-folder=weight, e.g. lora_out=0.5 (repeatable; "
    "none = SVM only).",
)
PUSH_DIR_OPT = typer.Option(
    None,
    help="A pulled copy of the Kaggle notebook (kaggle kernels pull <id> -p <dir> -m): its "
    "notebook is replaced and the GLM inputs are added to its kernel-metadata.json.",
)


def add_glm_sources(meta: dict, prefix: str) -> dict:
    """The GLM stage's inputs added to a notebook's kernel-metadata.json (each once)."""
    user = meta["id"].split("/")[0]
    _, kind, slug = GLM_FEATURE_SETS[prefix]
    extra = {"dataset_sources": [*SEGMENT_DATASETS], "kernel_sources": [BINARY_NOTEBOOK]}
    extra[kind].append(f"{user}/{slug}")
    for key, refs in extra.items():
        meta[key] = list(dict.fromkeys([*meta.get(key, []), *refs]))
    return meta


def add_native_sources(meta: dict) -> dict:
    """The native expert's fold states and its base model, added to kernel-metadata.json."""
    for key, ref in (("dataset_sources", NATIVE_ASSETS), ("model_sources", NATIVE_MODEL)):
        meta[key] = list(dict.fromkeys([*meta.get(key, []), ref]))
    return meta


@app.command()
def main(
    path: Path = NOTEBOOK_PATH,
    lora: list[str] = LORA_OPTION,
    t_svm: float = typer.Option(1.0, help="Temperature on the SVM scores in the blend."),
    glm_cover: float = typer.Option(
        0.0, help="Share of test passages the GLM-5.3 stage scores (0: no GLM stage)."
    ),
    glm_weight: float = typer.Option(2.0, help="Weight of the GLM-5.3 member in the blend."),
    glm_c: float = typer.Option(0.05, help="Inverse regularisation of the GLM-5.3 member."),
    glm_prefix: str = typer.Option(
        "max", help=f"Prompt prefix of the GLM models: {' or '.join(GLM_FEATURE_SETS)}."
    ),
    groups: bool = typer.Option(
        False, help="Group the test passages by prompt (TF-IDF) for the steps below."
    ),
    centre_f: float = typer.Option(0.5, help="Share of its group mean taken off the blend."),
    assign: bool = typer.Option(True, help="Within a group, give every passage a different model."),
    native_w: float = typer.Option(
        2.0, help="Weight of the native Qwen3-Embedding-8B expert (0: not run)."
    ),
    balance: float = typer.Option(0.3, help="Strength of the Sinkhorn class balancing (0: none)."),
    push_dir: Path = PUSH_DIR_OPT,
):
    weights = {k: float(w) for k, w in (s.split("=", 1) for s in lora)}
    assert glm_prefix in GLM_FEATURE_SETS, glm_prefix
    glm = (
        {"cover": glm_cover, "weight": glm_weight, "C": glm_c, "prefix": glm_prefix}
        if glm_cover > 0
        else None
    )
    group_steps = (
        {"centre": centre_f, "assign": assign, "native_w": native_w} if groups else None
    )
    nb = build(weights, t_svm, glm, group_steps, balance)
    nbf.write(nb, path)
    print(f"wrote {path} (lora={weights}, t_svm={t_svm}, glm={glm}, groups={group_steps}, balance={balance})")
    if push_dir is not None:
        meta_path = push_dir / "kernel-metadata.json"
        assert meta_path.exists(), f"no {meta_path}: pull the notebook first (-m for metadata)"
        meta = json.loads(meta_path.read_text())
        if glm:
            meta = add_glm_sources(meta, glm["prefix"])
        if group_steps and group_steps["native_w"] > 0:
            meta = add_native_sources(meta)
        nbf.write(nb, push_dir / meta["code_file"])
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"wrote {push_dir / meta['code_file']} and {meta_path}")


if __name__ == "__main__":
    app()
