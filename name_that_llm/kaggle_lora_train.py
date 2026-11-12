"""Build the Kaggle notebook that trains the two-stage LoRA classifier (references/lora-plan.md).

The notebook copies ``name_that_llm/lora.py`` verbatim, so rebuild it after changing that file:

    uv run python -m name_that_llm.kaggle_lora_train main --variant 4b   # or 8b

On Kaggle (GPU T4, internet off) it needs three inputs: the competition data, the Open Pool
dataset (crtproduction/name-that-llm-open-pool) and the variant's base model (see ``VARIANTS``).
``MODE`` at the top picks the run:

* ``"timing"``: 60 steps of stage 1, then prints s/step, peak GPU memory and the forecast for a
  full run. Costs ~15 min of GPU quota.
* ``"full"``: stage 1 on the pool sample once; stage 2 on 4/5 of train.csv and logits for the
  held-out fold (prompt-grouped fold 0); then stage 2 again on all of train.csv from the same
  stage-1 weights. Writes ``lora_out/`` (checkpoint, meta.json, holdout_logits.parquet).

Run outside Kaggle, it does a smoke test on a tiny random Qwen3 (see ``--smoke-model``) with a
few hundred passages on CPU, which checks the code end to end in a couple of minutes.
"""

import inspect
from pathlib import Path
import textwrap

import nbformat as nbf
import typer

from name_that_llm import lora, lora_ddp
from name_that_llm.config import PROJ_ROOT
from name_that_llm.kaggle_notebook import POOL_SAMPLE_CODE, module_body

app = typer.Typer()

NOTEBOOK_PATH = PROJ_ROOT / "notebooks" / "kaggle_lora_train.ipynb"

# Ensemble members. Each differs from the others in backbone and in its Open Pool draw, because a
# member trained on different data adds more to a blend than a stronger copy of one already in it.
VARIANTS = {
    "4b": {
        "name": "Qwen3-4B-Base",
        "source": "qwen-lm/qwen-3/Transformers/4b-base",
        "base": {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36},
        "pool_seed": 0,
        "n_per_label": 2000,
        "out": "lora_out",
        "notebook": NOTEBOOK_PATH,
    },
    "8b": {
        "name": "Qwen3-8B-Base",
        "source": "qwen-lm/qwen-3/Transformers/8b-base",
        "base": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
        "pool_seed": 1,
        "n_per_label": 2000,
        "out": "lora_out_8b",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_8b.ipynb",
    },
    # the 4B recipe with stage 1 on ~2.7x as many pool passages (references/lora-bigpool-plan.md)
    "4b-big": {
        "name": "Qwen3-4B-Base",
        "source": "qwen-lm/qwen-3/Transformers/4b-base",
        "base": {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36},
        "pool_seed": 2,
        "n_per_label": 5500,
        "out": "lora_out_4b_big",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_4b_big.ipynb",
    },
    # the 8B on the big pool; capped at 5,000 (88k passages, ~9 h) to stay clear of the
    # 12 h Kaggle session limit
    "8b-big": {
        "name": "Qwen3-8B-Base",
        "source": "qwen-lm/qwen-3/Transformers/8b-base",
        "base": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
        "pool_seed": 3,
        "n_per_label": 5000,
        "out": "lora_out_8b_big",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_8b_big.ipynb",
    },
    # both GPUs, 186k pool passages, stage 2 on all 5 folds (references/lora-2gpu-5fold-plan.md)
    "4b-full": {
        "name": "Qwen3-4B-Base",
        "source": "qwen-lm/qwen-3/Transformers/4b-base",
        "base": {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36},
        "pool_seed": 4,
        "n_per_label": 12000,
        "out": "lora_out_4b_full",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_4b_full.ipynb",
        "ddp": True,
    },
    # the same on the 8B; capped at 9,000 (151k passages, ~9.8 h) to stay clear of the 12 h
    # Kaggle session limit, since the 8B takes ~1.66x as long per step as the 4B
    "8b-full": {
        "name": "Qwen3-8B-Base",
        "source": "qwen-lm/qwen-3/Transformers/8b-base",
        "base": {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
        "pool_seed": 5,
        "n_per_label": 9000,
        "out": "lora_out_8b_full",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_8b_full.ipynb",
        "ddp": True,
    },
    # a third family for the blend (references/lora-third-member-plan.md): first 24 of Llama's 32
    # layers; ~1.13x the 8B's cost per step, so capped at 8,000 (~128k passages, ~9 h)
    "llama8b-full": {
        "name": "Llama-3.1-8B",
        "source": "metaresearch/llama-3.1/Transformers/8b",
        "base": {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32},
        "pool_seed": 6,
        "n_per_label": 8000,
        "out": "lora_out_llama8b_full",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_llama8b_full.ipynb",
        "ddp": True,
    },
    # the same plan on Mistral-7B, whose Kaggle copy needs no license request: 24 of 32 layers,
    # the Llama layer sizes but a 32k vocabulary (~15% more tokens per passage), same cap
    "mistral7b-full": {
        "name": "Mistral-7B-v0.1",
        "source": "mistral-ai/mistral/PyTorch/7b-v0.1-hf",
        "base": {"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32},
        "pool_seed": 6,
        "n_per_label": 8000,
        "out": "lora_out_mistral7b_full",
        "notebook": PROJ_ROOT / "notebooks" / "kaggle_lora_train_mistral7b_full.ipynb",
        "ddp": True,
    },
}

# Starts one training process per GPU (what torchrun does, without its launcher, which needs
# libuv on Windows) and stops all of them as soon as one fails, so a crash can't leave the others
# waiting for it until the session times out.
DDP_LAUNCH_CODE = """\
import subprocess
import sys
import threading

n_proc = torch.cuda.device_count() or 2  # 2 CPU processes for the local smoke test
env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT='29517', WORLD_SIZE=str(n_proc),
           USE_LIBUV='0', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
procs, logs = [], [open(f'rank{r}.log', 'w') for r in range(1, n_proc)]
for r in range(n_proc):
    out = subprocess.PIPE if r == 0 else logs[r - 1]
    procs.append(subprocess.Popen([sys.executable, 'lora_ddp.py', 'cfg.json'],
                                  env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
                                  stdout=out, stderr=subprocess.STDOUT, text=True))


def stream(p):
    for line in p.stdout:
        if 'UNEXPECTED' not in line:  # skip the per-layer load report of the cut-down model
            print(line, end='', flush=True)


reader = threading.Thread(target=stream, args=(procs[0],), daemon=True)
reader.start()
while any(p.poll() is None for p in procs):
    if any(p.poll() not in (None, 0) for p in procs):
        for p in procs:
            if p.poll() is None:
                p.kill()
        break
    time.sleep(5)
reader.join(timeout=30)
codes = [p.wait() for p in procs]
for f in logs:
    f.close()
for r in range(1, n_proc):
    if codes[r] != 0:
        print(f'--- rank {r} log ---')
        with open(f'rank{r}.log') as f:
            print(f.read()[-5000:])
assert codes == [0] * n_proc, f'training processes exited with {codes}'
"""


# Kaggle's TPU image may lack PyTorch/XLA or transformers; the training notebook may use the
# internet (only submission notebooks run offline), so install what is missing. torch_xla must
# match the installed torch version.
TPU_SETUP_CODE = """\
import importlib.metadata as md
import importlib.util
import os
import subprocess
import sys


def pip(*args):
    return subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args]).returncode == 0


# versions only, nothing imported: the training processes start fresh and open the TPU themselves
if os.path.exists('/kaggle/input'):
    if importlib.util.find_spec('torch_xla') is None:
        have = md.version('torch').split('+')[0] if importlib.util.find_spec('torch') else None
        # torch_xla for the installed torch, else a known torch + torch_xla pair
        if not (have and pip(f'torch_xla[tpu]=={have}')):
            assert pip('torch==2.7.0', 'torch_xla[tpu]==2.7.0'), 'could not install torch_xla'
    for module, package in [('transformers', 'transformers'), ('safetensors', 'safetensors'),
                            ('sklearn', 'scikit-learn'), ('pyarrow', 'pyarrow')]:
        if importlib.util.find_spec(module) is None:
            assert pip(package), f'could not install {package}'
    print({p: md.version(p) for p in ['torch', 'torch_xla', 'transformers']})
"""

# One process here; lora_ddp.py starts one process per TPU chip itself (xmp.spawn). This notebook
# process must not touch the TPU, or the training processes cannot open it.
TPU_LAUNCH_CODE = """\
import subprocess
import sys

env = dict(os.environ, PJRT_DEVICE='TPU', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
proc = subprocess.Popen([sys.executable, 'lora_ddp.py', 'cfg.json'], env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    if 'UNEXPECTED' not in line:  # skip the per-layer load report of the cut-down model
        print(line, end='', flush=True)
assert proc.wait() == 0, f'training exited with {proc.returncode}'
"""


def build_ddp(mode: str, variant: str, tpu: bool = False) -> nbf.NotebookNode:
    """The multi-GPU (or TPU), 5-fold training notebook: writes lora.py and lora_ddp.py next to
    it and runs one training process per GPU (or TPU chip)."""
    v = VARIANTS[variant]
    out = v["out"] + ("_tpu" if tpu else "")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    where = "TPU v5e-8" if tpu else "both GPUs"
    if tpu:
        needs = (
            "Needs **TPU VM v5e-8** and **internet on** (to install PyTorch/XLA if the image "
            "lacks it), and three inputs: the competition data, "
        )
        parallel = "data-parallel on the 8 TPU chips (bf16, static shapes)"
    else:
        needs = "Needs **GPU T4 x2**, internet off, and three inputs: the competition data, "
        parallel = "data-parallel on every GPU"
    cells = [
        md(
            f"# Name That LLM: two-stage LoRA classifier on {v['name']}, {where}, 5 folds "
            "(training)\n\n"
            "Generated by `name_that_llm/kaggle_lora_train.py`; do not edit by hand.\n\n"
            f"* **Backbone:** first 24 of {v['base']['num_hidden_layers']} layers of {v['name']}, "
            "LoRA r16 / alpha 32 on all "
            "linear layers, final norm off.\n"
            "* **Head:** mean-pooled hidden states at 8 depths, a LayerNorm each, one linear layer.\n"
            f"* **Stage 1:** 1 epoch on up to {v['n_per_label']:,} match-weighted Open Pool "
            f"passages per model (draw seed {v['pool_seed']}), {parallel} with "
            "the same global batch of 8. **Stage 2:** 2 epochs on train.csv.\n"
            "* **Folds:** from the stage-1 weights, stage 2 runs once per prompt-grouped fold "
            "(5) leaving it out and scoring it, then once on all of train.csv. Outputs: "
            "`oof_logits.parquet` for every training passage, `fold0..4/`, `stage1/`, and the "
            "full fit at the top of the folder.\n\n"
            + needs
            + f"`crtproduction/name-that-llm-open-pool` and `{v['source']}`.\n\n"
            "To switch between the timing run and the full run, change `MODE` in the next cell."
        ),
        *([code(TPU_SETUP_CODE)] if tpu else []),
        code(
            "import glob\nimport json\nimport os\nimport time\n\nimport torch\n\n"
            f"MODE = {mode!r}  # 'timing' or 'full'\n"
            "T0 = time.time()\n"
            "ON_KAGGLE = os.path.exists('/kaggle/input')\n"
            f"BASE_SPEC = {v['base']!r}  # config.json fields of {v['source']}\n\n\n"
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
            "    attached = sorted(glob.glob('/kaggle/input/*/*'))\n"
            "    print('attached inputs:', attached)\n"
            "    comps = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train.csv', "
            "recursive=True)\n"
            "             if os.path.exists(os.path.join(os.path.dirname(p), 'test.csv'))]\n"
            "    assert comps, f'Attach the name-that-llm competition data; found {attached}'\n"
            "    pools = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)\n"
            "    assert pools, f'Attach crtproduction/name-that-llm-open-pool; found {attached}'\n"
            + (
                "    # the TPU chips show up as device files (not opened here, see the launch)\n"
                "    assert glob.glob('/dev/vfio/[0-9]*') or glob.glob('/dev/accel*'), "
                "'pick the TPU VM v5e-8 accelerator for this notebook'\n"
                if tpu
                else "    assert torch.cuda.device_count() >= 2, "
                "'pick the GPU T4 x2 accelerator for this notebook'\n"
            )
            + "    paths = dict(comp=comps[0], pool=pools[0], base_dir=find_base(BASE_SPEC), "
            f"out={out!r})\n"
            "    SMOKE = False\n"
            "else:  # smoke test from this project's notebooks/ folder, 2 CPU processes\n"
            "    paths = dict(comp='../data/raw', "
            "pool='../data/external/open_pool/ntl_open_pool.parquet',\n"
            "                 base_dir=os.environ.get('NTL_SMOKE_MODEL', "
            "'../data/interim/tiny_qwen3'),\n"
            "                 out='../data/interim/lora_smoke_out_ddp')\n"
            "    SMOKE = True\n\n"
            "CFG = dict(\n"
            "    keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], r=16, alpha=32, dropout=0.05,\n"
            "    max_len=384, bs=8,  # global batch, split across the GPUs\n"
            "    stage1=dict(epochs=1, lr_lora=1e-4, lr_head=1e-3),\n"
            "    stage2=dict(epochs=2, lr_lora=5e-5, lr_head=5e-4),\n"
            f"    n_per_label={v['n_per_label']}, pool_seed={v['pool_seed']}, folds=5, seed=0,\n"
            f"    timing_steps={150 if tpu else 60}, base=BASE_SPEC, mode=MODE, smoke=SMOKE, "
            "**paths,\n"
            + (
                "    # TPU: bf16, no gradient checkpointing, 4 ranks load the trunk at a time\n"
                "    accelerator='tpu' if ON_KAGGLE else 'cpu', ckpt=False, load_parallel=4,\n"
                if tpu
                else ""
            )
            + ")\n"
            "if SMOKE:\n"
            "    CFG.update(keep=2, layers=[1, 2], max_len=64, bs=4, n_per_label=10)\n"
            "with open('cfg.json', 'w') as f:\n"
            "    json.dump(CFG, f, indent=1)\n"
            + (
                "print('mode', MODE, '| accelerator', CFG['accelerator'], '| base model', "
                "paths['base_dir'])"
                if tpu
                else "print('mode', MODE, '| GPUs', torch.cuda.device_count(), '| base model', "
                "paths['base_dir'])"
            )
        ),
        md("## Training code (copied from `name_that_llm/lora.py` and `lora_ddp.py`)"),
        code("%%writefile lora.py\n" + inspect.getsource(lora)),
        code("%%writefile lora_ddp.py\n" + inspect.getsource(lora_ddp)),
        md(
            f"## Train: one process per {'TPU chip' if tpu else 'GPU'}\n\n"
            f"In `timing` mode stage 1 runs only {150 if tpu else 60} steps and prints a "
            "forecast for a full run; "
            "then every later step runs briefly (10 stage-2 steps per fold, 48 held-out "
            "passages scored per fold), so a bug there shows up in minutes, not hours."
            + (
                " The first steps of each new sequence length are slow while XLA compiles; "
                "the forecast uses the steps after the first 20."
                if tpu
                else ""
            )
        ),
        code(
            # off Kaggle the TPU notebook runs the CPU smoke test through the DDP launcher
            "if CFG.get('accelerator') == 'tpu':\n"
            + textwrap.indent(TPU_LAUNCH_CODE, "    ")
            + "else:\n"
            + textwrap.indent(DDP_LAUNCH_CODE, "    ")
            if tpu
            else DDP_LAUNCH_CODE
        ),
        md("## Summary"),
        code(
            "for f in ['lora.py', 'lora_ddp.py', 'cfg.json'] + glob.glob('rank*.log'):\n"
            "    os.remove(f)  # keep the output folder to the checkpoints\n"
            "if MODE == 'timing':\n"
            "    print('timing run: these scores come from 10 stage-2 steps per fold and only '\n"
            "          'show that every step runs; set MODE to full for the real run')\n"
            "meta = json.load(open(f\"{CFG['out']}/meta.json\"))\n"
            "for fold in meta['folds']:\n"
            "    print(f\"fold {fold['fold']}: {fold['balanced_accuracy']:.4f} ({fold['n']} passages)\")\n"
            "print(f\"out-of-fold: {meta['oof_balanced_accuracy']:.4f}\")\n"
            "print('outputs:', sorted(os.listdir(CFG['out'])))\n"
            "print(f'done in {(time.time() - T0) / 60:.1f} min')"
        ),
    ]
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
    )


def build(mode: str, variant: str = "4b", tpu: bool = False) -> nbf.NotebookNode:
    v = VARIANTS[variant]
    if v.get("ddp"):
        return build_ddp(mode, variant, tpu)
    if tpu:
        raise ValueError(f"variant {variant} is single-GPU; --tpu needs a multi-process variant")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md(
            f"# Name That LLM: two-stage LoRA classifier on {v['name']} (training)\n\n"
            "Generated by `name_that_llm/kaggle_lora_train.py`; do not edit by hand.\n\n"
            f"* **Backbone:** first 24 of {v['base']['num_hidden_layers']} layers of {v['name']}, "
            "LoRA r16 / alpha 32 on all "
            "linear layers, final norm off.\n"
            "* **Head:** mean-pooled hidden states at 8 depths, a LayerNorm each, one linear layer.\n"
            f"* **Stage 1:** 1 epoch on up to {v['n_per_label']:,} match-weighted Open Pool "
            "passages per model "
            f"(draw seed {v['pool_seed']}). **Stage 2:** 2 epochs on train.csv.\n"
            "* **Check:** stage 2 is run once without prompt-grouped fold 0, to score that fold, "
            "and once on all of train.csv for the final checkpoint.\n\n"
            "Needs GPU T4 (one is used), internet off, and three inputs: the competition data, "
            f"`crtproduction/name-that-llm-open-pool` and `{v['source']}`.\n\n"
            "To switch between the timing run and the full run, change `MODE` in the next cell."
        ),
        code(
            "import gc\nimport glob\nimport json\nimport os\nimport time\n\n"
            "import numpy as np\nimport pandas as pd\nimport torch\n"
            "from sklearn.metrics import balanced_accuracy_score\n"
            "from sklearn.model_selection import GroupKFold\n\n"
            f"MODE = {mode!r}  # 'timing' or 'full'\n"
            "T0 = time.time()\n"
            "ON_KAGGLE = os.path.exists('/kaggle/input')\n"
            f"BASE_SPEC = {v['base']!r}  # config.json fields of {v['source']}\n\n\n"
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
            "    attached = sorted(glob.glob('/kaggle/input/*/*'))\n"
            "    print('attached inputs:', attached)\n"
            "    comps = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/train.csv', "
            "recursive=True)\n"
            "             if os.path.exists(os.path.join(os.path.dirname(p), 'test.csv'))]\n"
            "    assert comps, f'Attach the name-that-llm competition data; found {attached}'\n"
            "    COMP = comps[0]\n"
            "    pools = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)\n"
            "    assert pools, f'Attach crtproduction/name-that-llm-open-pool; found {attached}'\n"
            "    POOL = pools[0]\n"
            "    BASE = find_base(BASE_SPEC)\n"
            f"    OUT = {v['out']!r}\n"
            "    SMOKE = False\n"
            "else:  # smoke test from this project's notebooks/ folder\n"
            "    COMP, POOL = '../data/raw', '../data/external/open_pool/ntl_open_pool.parquet'\n"
            "    BASE = os.environ.get('NTL_SMOKE_MODEL', '../data/interim/tiny_qwen3')\n"
            "    OUT = '../data/interim/lora_smoke_out'\n"
            "    SMOKE = True\n\n"
            "CFG = dict(\n"
            "    keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], r=16, alpha=32, dropout=0.05,\n"
            "    max_len=384, bs=8,\n"
            "    stage1=dict(epochs=1, lr_lora=1e-4, lr_head=1e-3),\n"
            "    stage2=dict(epochs=2, lr_lora=5e-5, lr_head=5e-4),\n"
            f"    n_per_label={v['n_per_label']}, pool_seed={v['pool_seed']}, holdout_fold=0, "
            "folds=5, seed=0,\n"
            "    base=BASE_SPEC,\n"
            ")\n"
            "if SMOKE:\n"
            "    CFG.update(keep=2, layers=[1, 2], max_len=64, bs=4, n_per_label=10)\n"
            "N_PER_LABEL, POOL_SEED = CFG['n_per_label'], CFG['pool_seed']\n\n"
            "if torch.cuda.is_available():\n"
            "    DEVICE = torch.device('cuda:0')\n"
            "    AMP = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 "
            "else torch.float16\n"
            "    print(torch.cuda.get_device_name(0), 'amp', AMP)\n"
            "else:\n"
            "    DEVICE, AMP = torch.device('cpu'), torch.float32\n"
            "print('mode', MODE, '| base model', BASE)"
        ),
        md("## LoRA classifier (copied from `name_that_llm/lora.py`)"),
        code(module_body(lora)),
        md("## Data"),
        code(
            "train = pd.read_csv(f'{COMP}/train.csv')\n" + POOL_SAMPLE_CODE + "\nif SMOKE:\n"
            "    train = train.groupby('model').head(12).reset_index(drop=True)\n"
            "LABELS = sorted(train.model.unique())\n"
            "L2I = {m: i for i, m in enumerate(LABELS)}\n"
            "assert len(LABELS) == 18 and set(pool_sample.model) <= set(LABELS)\n"
            "y_train = train.model.map(L2I).to_numpy()\n"
            "y_pool = pool_sample.model.map(L2I).to_numpy()\n\n"
            "folds = list(GroupKFold(n_splits=CFG['folds']).split(train, y_train, "
            "train.prompt_id))\n"
            "fit_idx, hold_idx = folds[CFG['holdout_fold']]\n"
            "print(f'train {len(train)} (holdout {len(hold_idx)}), pool sample "
            "{len(pool_sample)}')"
        ),
        md("## Model"),
        code(
            "tok = load_tokenizer(BASE)\n"
            "trunk = load_trunk(BASE, CFG['keep'], AMP if DEVICE.type == 'cuda' else "
            "torch.float32, DEVICE)\n"
            "model = LoRAClassifier(trunk, CFG['layers'], len(LABELS), CFG['r'], CFG['alpha'], "
            "CFG['dropout'])\n"
            "n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
            "print(f'{model.n_lora} LoRA layers, {n_train / 1e6:.1f}M trainable parameters, '\n"
            "      f'loaded in {time.time() - T0:.0f}s')\n\n"
            "common = dict(bs=CFG['bs'], max_len=CFG['max_len'], n_classes=len(LABELS), "
            "amp_dtype=AMP)"
        ),
        md(
            "## Stage 1: the Open Pool sample\n\n"
            "In `timing` mode this runs only 60 steps and forecasts a full run."
        ),
        code(
            "if DEVICE.type == 'cuda':\n"
            "    torch.cuda.reset_peak_memory_stats()\n"
            "timing = MODE == 'timing'\n"
            "s1 = train_stage(model, tok, pool_sample.text, y_pool, **CFG['stage1'], **common,\n"
            "                 seed=CFG['seed'], max_steps=60 if timing else None, "
            "log_every=20 if timing else 200)\n"
            "print('stage 1:', s1)\n"
            "if DEVICE.type == 'cuda':\n"
            "    print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')\n"
            "if timing:\n"
            "    per_step = s1['seconds'] / s1['steps']\n"
            "    bs = CFG['bs']\n"
            "    steps = {\n"
            "        'stage 1': -(-len(pool_sample) // bs),\n"
            "        'stage 2 without holdout': 2 * -(-len(fit_idx) // bs),\n"
            "        'stage 2 on all': 2 * -(-len(train) // bs),\n"
            "    }\n"
            "    for k, v in steps.items():\n"
            "        print(f'{k}: {v} steps, ~{v * per_step / 60:.0f} min')\n"
            "    print(f'forecast for MODE=full: ~{sum(steps.values()) * per_step / 3600:.1f} h '\n"
            "          f'of training plus loading and prediction')\n"
            "else:\n"
            "    stage1_state = {k: v.clone() for k, v in model.trainable_state().items()}"
        ),
        md("## Stage 2 without the held-out fold, then score that fold"),
        code(
            "if MODE == 'full':\n"
            "    s2 = train_stage(model, tok, train.text.iloc[fit_idx], y_train[fit_idx], "
            "**CFG['stage2'], **common,\n"
            "                     seed=CFG['seed'] + 1, log_every=100)\n"
            "    logits = predict_logits(model, tok, train.text.iloc[hold_idx], bs=2 * CFG['bs'],\n"
            "                            max_len=CFG['max_len'], amp_dtype=AMP)\n"
            "    pred = logits.argmax(1)\n"
            "    hold_bacc = balanced_accuracy_score(y_train[hold_idx], pred)\n"
            "    print(f'stage 2: {s2}\\nheld-out fold: balanced accuracy {hold_bacc:.4f} '\n"
            "          f'({len(hold_idx)} passages, {train.prompt_id.iloc[hold_idx].nunique()} "
            "prompts)')\n"
            "    os.makedirs(OUT, exist_ok=True)\n"
            "    hold = pd.DataFrame(logits, columns=LABELS)\n"
            "    hold.insert(0, 'id', train.id.iloc[hold_idx].to_numpy())\n"
            "    hold.insert(1, 'model', train.model.iloc[hold_idx].to_numpy())\n"
            "    hold.to_parquet(f'{OUT}/holdout_logits.parquet', index=False)"
        ),
        md("## Stage 2 on all of train.csv, from the same stage-1 weights; save"),
        code(
            "if MODE == 'full':\n"
            "    model.load_state_dict({k: v.to(DEVICE) for k, v in stage1_state.items()}, "
            "strict=False)\n"
            "    s2_all = train_stage(model, tok, train.text, y_train, **CFG['stage2'], "
            "**common,\n"
            "                         seed=CFG['seed'] + 2, log_every=100)\n"
            "    save_checkpoint(model, OUT, dict(labels=LABELS, cfg=CFG, "
            "holdout_balanced_accuracy=hold_bacc,\n"
            "                                     stage1=s1, stage2_holdout=s2, "
            "stage2_all=s2_all))\n"
            "    print('saved', sorted(os.listdir(OUT)))\n"
            "print(f'done in {(time.time() - T0) / 60:.1f} min')"
        ),
    ]
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
    )


@app.command()
def main(
    mode: str = "timing",
    variant: str = typer.Option("4b", help=f"Ensemble member: {', '.join(VARIANTS)}."),
    path: Path | None = None,  # default: the variant's own notebook
    tpu: bool = typer.Option(False, help="Train on a Kaggle TPU v5e-8 (multi-process variants)."),
):
    if mode not in ("timing", "full"):
        raise typer.BadParameter("mode is 'timing' or 'full'")
    if variant not in VARIANTS:
        raise typer.BadParameter(f"variant is one of {', '.join(VARIANTS)}")
    path = path or VARIANTS[variant]["notebook"]
    if tpu and path == VARIANTS[variant]["notebook"]:
        path = path.with_name(path.stem + "_tpu.ipynb")
    nbf.write(build(mode, variant, tpu), path)
    print(f"wrote {path} (variant={variant}, MODE={mode!r}, tpu={tpu})")


@app.command()
def smoke_model(
    out: Path | None = None,
    arch: str = typer.Option(
        "qwen3", help="qwen3, or llama / mistral (tiny models with a BOS token)."
    ),
):
    """A tiny random model with the real Qwen3 tokenizer, for the local smoke test. The llama
    one gets a BOS token, as Llama's tokenizer has, so the BOS path is tested too."""
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaModel,
        MistralConfig,
        MistralModel,
        Qwen3Config,
        Qwen3Model,
    )

    out = out or PROJ_ROOT / "data" / "interim" / f"tiny_{arch}"
    tok_dir = snapshot_download(
        "Qwen/Qwen3-4B-Base", allow_patterns=["tokenizer*", "vocab.json", "merges.txt"]
    )
    tok = AutoTokenizer.from_pretrained(tok_dir)
    small = {
        "vocab_size": len(tok),
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    }
    if arch in ("llama", "mistral"):
        tok.bos_token = "<|im_start|>"  # stands in for <|begin_of_text|> / <s>
        config, model = (
            (LlamaConfig, LlamaModel) if arch == "llama" else (MistralConfig, MistralModel)
        )
        model(config(**small, bos_token_id=tok.bos_token_id)).save_pretrained(out)
    else:
        Qwen3Model(Qwen3Config(**small)).save_pretrained(out)
    tok.save_pretrained(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
