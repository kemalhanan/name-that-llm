"""Train a 5-fold LoRA member on any CUDA machine (a rented A100/H100, Colab), not a Kaggle
notebook, so it uses no Kaggle GPU quota. Only the submission has to run on Kaggle: upload the
output folder as a private dataset and attach it to the submission notebook.

On the GPU machine (Linux, Python 3.12, uv)::

    git clone https://github.com/kemalhanan/name-that-llm.git && cd name-that-llm
    uv sync --group lora
    export KAGGLE_USERNAME=... KAGGLE_KEY=...          # or ~/.kaggle/kaggle.json
    uv run python -m name_that_llm.remote_train fetch mistral7b-full
    uv run python -m name_that_llm.remote_train run mistral7b-full --mode timing   # ~10 min
    uv run python -m name_that_llm.remote_train run mistral7b-full --mode full
    uv run python -m name_that_llm.remote_train package mistral7b-full --user <kaggle user>

Outputs go to data/remote/out/ (git ignores data/).

``fetch`` downloads the competition data, the Open Pool and the variant's base model from Kaggle
(kagglehub; downloads use no quota). ``run`` writes the same cfg.json as the training notebook and
starts one training process per GPU (lora_ddp.py). The global batch stays 8 whatever the number of
GPUs, so the recipe matches the Kaggle-trained members. On an A100, H100 or L4 the trunk trains in
bf16 instead of the T4s' fp16 (lora_ddp.Comm picks it from the GPU).
"""

import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import typer

from name_that_llm.config import PROJ_ROOT
from name_that_llm.kaggle_lora_train import VARIANTS

app = typer.Typer()

COMPETITION = "name-that-llm"
POOL_DATASET = "crtproduction/name-that-llm-open-pool"
REMOTE_DIR = PROJ_ROOT / "data" / "remote"  # downloads, cfg.json and logs
MODELS_DIR = REMOTE_DIR / "out"  # under data/, which git ignores


def find_base(root: str, spec: dict) -> str:
    """The folder under ``root`` whose config.json matches ``spec`` (as the notebooks do)."""
    for cfg in sorted(glob.glob(f"{root}/**/config.json", recursive=True)):
        try:
            with open(cfg) as f:
                c = json.load(f)
        except Exception:  # noqa: BLE001, S112
            continue
        if all(c.get(k) == v for k, v in spec.items()):
            return os.path.dirname(cfg)
    raise FileNotFoundError(f"no base model {spec} under {root}; run fetch first")


def train_cfg(variant: str, mode: str, paths: dict, smoke: bool = False) -> dict:
    """The training notebook's CFG (kaggle_lora_train.build_ddp); keep the two in step."""
    v = VARIANTS[variant]
    cfg = dict(
        keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], r=16, alpha=32, dropout=0.05,
        max_len=384, bs=8,  # global batch, split across the GPUs
        stage1={"epochs": 1, "lr_lora": 1e-4, "lr_head": 1e-3},
        stage2={"epochs": 2, "lr_lora": 5e-5, "lr_head": 5e-4},
        n_per_label=v["n_per_label"], pool_seed=v["pool_seed"], folds=5, seed=0,
        timing_steps=60, base=v["base"], mode=mode, smoke=smoke, **paths,
    )  # fmt: skip
    if smoke:
        cfg.update(keep=2, layers=[1, 2], max_len=64, bs=4, n_per_label=10)
    return cfg


def launch(cfg_path: Path, n_proc: int, log_dir: Path) -> None:
    """One lora_ddp process per GPU, as the notebook starts them; rank 0 prints here, the others
    log to rank<N>.log. Stops every process as soon as one fails."""
    env = dict(
        os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="29517", WORLD_SIZE=str(n_proc),
        USE_LIBUV="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
    )  # fmt: skip
    cmd = [sys.executable, "-m", "name_that_llm.lora_ddp", str(cfg_path)]
    logs = [open(log_dir / f"rank{r}.log", "w") for r in range(1, n_proc)]  # noqa: SIM115
    procs = []
    for r in range(n_proc):
        out = subprocess.PIPE if r == 0 else logs[r - 1]
        procs.append(
            subprocess.Popen(
                cmd, env=dict(env, RANK=str(r), LOCAL_RANK=str(r)), cwd=PROJ_ROOT,
                stdout=out, stderr=subprocess.STDOUT, text=True,
            )
        )  # fmt: skip

    def stream(p):
        with open(log_dir / "rank0.log", "w") as f:
            for line in p.stdout:
                f.write(line)
                if "UNEXPECTED" not in line:  # the per-layer load report of the cut-down model
                    print(line, end="", flush=True)

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
            print(f"--- rank {r} log ---")
            print((log_dir / f"rank{r}.log").read_text()[-5000:])
    if codes != [0] * n_proc:
        raise SystemExit(f"training processes exited with {codes}")


@app.command()
def fetch(variant: str, root: Path = REMOTE_DIR):
    """Download the competition data, the Open Pool and the variant's base model."""
    import kagglehub

    root.mkdir(parents=True, exist_ok=True)
    v = VARIANTS[variant]
    found = {
        "comp": kagglehub.competition_download(COMPETITION),
        "pool": kagglehub.dataset_download(POOL_DATASET),
        "model": kagglehub.model_download(v["source"]),
    }
    find_base(found["model"], v["base"])  # fail now, not after the pool sample is drawn
    (root / "inputs.json").write_text(json.dumps(found, indent=1))
    print(json.dumps(found, indent=1))


@app.command()
def run(
    variant: str,
    mode: str = typer.Option("timing", help="timing (60 stage-1 steps and a forecast) or full."),
    gpus: int = typer.Option(0, help="Training processes; 0 = every visible GPU."),
    root: Path = REMOTE_DIR,
    out: Path | None = None,  # default data/remote/out/<the variant's out>
    smoke: bool = typer.Option(False, help="Tiny model, local data, 2 CPU processes."),
):
    """Write cfg.json and train, one process per GPU."""
    import torch

    assert mode in ("timing", "full"), mode
    v = VARIANTS[variant]
    out = out or MODELS_DIR / (v["out"] + ("_smoke" if smoke else ""))
    if smoke:
        base = os.environ.get("NTL_SMOKE_MODEL", str(PROJ_ROOT / "data/interim/tiny_qwen3"))
        paths = {
            "comp": str(PROJ_ROOT / "data/raw"),
            "pool": str(PROJ_ROOT / "data/external/open_pool/ntl_open_pool.parquet"),
            "base_dir": base,
        }
        n_proc = 2
    else:
        found = json.loads((root / "inputs.json").read_text())
        pools = glob.glob(f"{found['pool']}/**/ntl_open_pool.parquet", recursive=True)
        assert pools, f"no ntl_open_pool.parquet under {found['pool']}"
        base = find_base(found["model"], v["base"])
        paths = {"comp": found["comp"], "pool": pools[0], "base_dir": base}
        n_proc = gpus or torch.cuda.device_count()
        assert n_proc >= 1, "no GPU visible; pass --smoke for the CPU test"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; move it or pass --out")
    root.mkdir(parents=True, exist_ok=True)
    cfg = train_cfg(variant, mode, dict(paths, out=str(out)), smoke=smoke)
    cfg_path = root / f"cfg_{variant}_{mode}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"mode {mode} | {n_proc} processes | base model {paths['base_dir']} | out {out}")

    t0 = time.time()
    launch(cfg_path, n_proc, root)
    meta = json.loads((out / "meta.json").read_text())
    if mode == "timing":
        print("timing run: the scores below come from 10 stage-2 steps per fold and only show "
              "that every step runs")  # fmt: skip
    for fold in meta["folds"]:
        print(f"fold {fold['fold']}: {fold['balanced_accuracy']:.4f} ({fold['n']} passages)")
    print(f"out-of-fold: {meta['oof_balanced_accuracy']:.4f}")
    print(f"outputs in {out}: {sorted(os.listdir(out))}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


@app.command()
def package(
    variant: str,
    user: str = typer.Option(..., help="Your Kaggle username (owner of the new dataset)."),
    out: Path | None = None,
):
    """Wrap the output folder for upload as a private Kaggle dataset. The submission notebook finds
    a member by its folder name (lora_out*), so the folder keeps the training notebook's name."""
    v = VARIANTS[variant]
    out = out or MODELS_DIR / v["out"]
    assert (out / "meta.json").exists(), f"no finished run in {out}"
    stage = MODELS_DIR / "upload" / v["out"]
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(out, stage / v["out"])
    slug = v["out"].replace("_", "-")
    meta = {"title": f"ntl {slug}", "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"staged in {stage}. Upload (private by default) with:\n"
          f"  kaggle datasets create -p {stage} --dir-mode tar\n"
          f"then attach {user}/{slug} to the submission notebook.")  # fmt: skip


REPO = "kemalhanan/name-that-llm"


def colab_cells(variant: str) -> list:
    """The cells of the Colab notebook: clone this repo, then run fetch, run and package."""
    import nbformat as nbf

    v = VARIANTS[variant]
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md(
            f"# Name That LLM: train the {v['name']} LoRA member on Colab\n\n"
            "Generated by `name_that_llm/remote_train.py colab`; do not edit by hand.\n\n"
            "Trains the same 5-fold, two-stage LoRA member as the Kaggle training notebook, "
            "without using Kaggle GPU quota, then stages the output for upload as a private "
            "Kaggle dataset.\n\n"
            "**Before running:**\n"
            "1. **Runtime → Change runtime type → A100 GPU** (an L4 works, at about the speed of "
            "Kaggle's two T4s).\n"
            "2. **Secrets** (key icon on the left): add `KAGGLE_USERNAME` and `KAGGLE_KEY` and "
            "give this notebook access. You must have joined the competition on Kaggle.\n"
            "3. Run the timing run first (`MODE = 'timing'`, ~10 min), check its s/step, then "
            "set `MODE = 'full'` and run all again.\n\n"
            "Keep this tab open while the full run trains: a disconnect stops it, and stage 1 has "
            "no checkpoint part-way, so it would start over."
        ),
        code(
            f"VARIANT = {variant!r}\n"
            "MODE = 'timing'  # 'timing' (~10 min speed check) or 'full'\n"
            "KAGGLE_USER = ''  # your Kaggle username, for the upload step\n"
            "# the full run writes to Google Drive, so the result survives a disconnect\n"
            f"DRIVE_OUT = '/content/drive/MyDrive/ntl/{v['out']}'"
        ),
        md("## Setup"),
        code(
            "import os\n"
            "import subprocess\n\n"
            "from google.colab import drive, userdata\n\n"
            "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', "
            "'--format=csv'],\n"
            "                     capture_output=True, text=True).stdout)\n"
            "drive.mount('/content/drive')\n"
            "os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\n"
            "os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')"
        ),
        code(
            "%cd /content\n"
            f"!test -d ntl || git clone https://github.com/{REPO}.git ntl\n"
            "%cd /content/ntl\n"
            "!git pull -q\n"
            "!pip install -q uv\n"
            "!uv sync -q --group lora"
        ),
        md(
            "## Inputs\n\n"
            "The competition data, the Open Pool and the base model, downloaded with kagglehub "
            "to this runtime's disk (about 15 GB for a 7-8B model)."
        ),
        code("!uv run python -m name_that_llm.remote_train fetch {VARIANT}"),
        md(
            "## Train\n\n"
            "Timing runs write to the runtime's disk and start fresh each time. The full run "
            "writes to `DRIVE_OUT` and refuses to start if that folder holds files, for example "
            "from a run that was cut off: delete it first."
        ),
        code(
            "if MODE == 'full':\n"
            "    OUT = DRIVE_OUT\n"
            "else:\n"
            "    OUT = '/content/timing_out'\n"
            "    !rm -rf {OUT}\n"
            "!uv run python -m name_that_llm.remote_train run {VARIANT} --mode {MODE} --out {OUT}"
        ),
        md(
            "## Upload\n\n"
            "Stages the full run's output with a `dataset-metadata.json`. The folder keeps the "
            "training notebook's name, which is how the submission notebook finds the member. "
            "Then upload it (the Kaggle CLI makes new datasets private) and attach it to the "
            "submission notebook."
        ),
        code(
            "assert MODE == 'full', 'upload only a full run'\n"
            "assert KAGGLE_USER, 'set KAGGLE_USER in the first cell'\n"
            "!uv run python -m name_that_llm.remote_train package {VARIANT} --user {KAGGLE_USER} "
            "--out {DRIVE_OUT}\n"
            f"STAGE = '/content/ntl/data/remote/out/upload/{v['out']}'\n"
            "!uv run --with kaggle kaggle datasets create -p {STAGE} --dir-mode tar"
        ),
    ]


@app.command()
def colab(variant: str, path: Path | None = None):
    """Write the Colab notebook that runs fetch, run and package for ``variant``."""
    import nbformat as nbf

    path = path or PROJ_ROOT / "notebooks" / f"colab_lora_train_{variant.replace('-', '_')}.ipynb"
    nb = nbf.v4.new_notebook(
        cells=colab_cells(variant),
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
        },
    )
    nbf.write(nb, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    app()
