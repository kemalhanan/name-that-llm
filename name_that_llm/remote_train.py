"""Train a 5-fold LoRA member on any CUDA machine (a rented A100/H100, Colab), not a Kaggle
notebook, so it uses no Kaggle GPU quota. Only the submission has to run on Kaggle: upload the
output folder as a private dataset and attach it to the submission notebook.

On the GPU machine (Linux, Python 3.12, uv)::

    git clone https://github.com/kemalhanan/name-that-llm.git && cd name-that-llm
    uv sync --group lora
    export KAGGLE_API_TOKEN=...   # or the legacy KAGGLE_USERNAME + KAGGLE_KEY
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

On a machine that can go away mid-run (Colab), keep the run's progress on Kaggle and resume it::

    uv run python -m name_that_llm.remote_train restore mistral7b-full --user <kaggle user>
    uv run python -m name_that_llm.remote_train sync mistral7b-full --user <kaggle user> &
    uv run python -m name_that_llm.remote_train run mistral7b-full --mode full --resume
    uv run python -m name_that_llm.remote_train sync mistral7b-full --user <kaggle user> --once

``run`` saves stage 1 every ``--save-every-min`` minutes and each fold as it finishes; ``sync``
uploads the output folder as a new version of the private dataset whenever it has changed;
``restore`` downloads the latest version into the output folder, and ``run --resume`` then carries
on from it instead of starting over. ``unattended`` does those four steps in one command that is
safe to start again at any time; ``scripts/colab/`` starts it on a Colab runtime with no browser
tab (push.py copies this working tree and the Kaggle credential there, ensure.sh starts or
reports the run). Nothing is written to Google Drive.
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


def train_cfg(
    variant: str, mode: str, paths: dict, smoke: bool = False, save_every_min: float = 0
) -> dict:
    """The training notebook's CFG (kaggle_lora_train.build_ddp); keep the two in step."""
    v = VARIANTS[variant]
    cfg = dict(
        keep=24, layers=[2, 4, 6, 8, 12, 16, 20, 24], r=16, alpha=32, dropout=0.05,
        max_len=384, bs=8,  # global batch, split across the GPUs
        stage1={"epochs": 1, "lr_lora": 1e-4, "lr_head": 1e-3},
        stage2={"epochs": 2, "lr_lora": 5e-5, "lr_head": 5e-4},
        n_per_label=v["n_per_label"], pool_seed=v["pool_seed"], folds=5, seed=0,
        timing_steps=60, base=v["base"], mode=mode, smoke=smoke,
        save_every_min=save_every_min, **paths,
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
                f.flush()
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


def download_model(source: str) -> str:
    """The base model's folder. kagglehub can fail on one small file after fetching the rest (it
    raised ``KaggleApiHTTPError: 1994`` on Qwen3-14B's vocab.json, although the server answered
    200). The download is then still usable if the config, the tokenizer and every weight file
    named in the index are there, since the tokenizer loads from tokenizer.json alone."""
    import kagglehub

    try:
        return kagglehub.model_download(source)
    except Exception as e:  # noqa: BLE001
        cache = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub"))
        for d in sorted((cache / "models" / source).glob("*"), reverse=True):
            index = d / "model.safetensors.index.json"
            if not ((d / "config.json").exists() and (d / "tokenizer.json").exists()):
                continue
            if index.exists():
                shards = set(json.loads(index.read_text())["weight_map"].values())
            else:
                shards = {"model.safetensors"}
            if all((d / s).exists() for s in shards):
                print(f"kagglehub failed on a file the model does not need ({e!s:.80}); using {d}")
                return str(d)
        raise


@app.command()
def fetch(variant: str, root: Path = REMOTE_DIR):
    """Download the competition data, the Open Pool and the variant's base model."""
    import kagglehub

    root.mkdir(parents=True, exist_ok=True)
    v = VARIANTS[variant]
    found = {
        "comp": kagglehub.competition_download(COMPETITION),
        "pool": kagglehub.dataset_download(POOL_DATASET),
        "model": download_model(v["source"]),
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
    resume: bool = typer.Option(
        False, help="Continue the run already in --out (from its last save) instead of refusing."
    ),
    save_every_min: float = typer.Option(
        20, help="Save stage 1 this often, so --resume can continue it (0: no saves)."
    ),
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
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f"{out} is not empty; move it, pass --out, or pass --resume")
    if resume and (out / "oof_logits.parquet").exists():
        raise SystemExit(f"the run in {out} already finished")
    root.mkdir(parents=True, exist_ok=True)
    cfg = train_cfg(
        variant, mode, dict(paths, out=str(out)), smoke=smoke, save_every_min=save_every_min
    )
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


def dataset_ref(variant: str, user: str) -> str:
    return f"{user}/{VARIANTS[variant]['out'].replace('_', '-')}"


def kaggle_cli(*args: str) -> subprocess.CompletedProcess:
    """The Kaggle CLI (``uv run --with kaggle``: it is not a project dependency)."""
    cmd = ["uv", "run", "--with", "kaggle", "kaggle", *args]
    return subprocess.run(cmd, check=False, cwd=PROJ_ROOT, capture_output=True, text=True)


PROGRESS = "progress.bin"  # a tar of the output folder; not named .tar, so Kaggle keeps it whole


def upload(variant: str, user: str, out: Path, note: str, progress: bool = False) -> bool:
    """Upload ``out`` as a new version of the private dataset, or as the dataset itself the first
    time. A finished run goes up as its folder (see ``package``); a run in progress as the single
    file ``progress.bin``, which ``restore`` can fetch by name. Returns whether Kaggle took it."""
    import tarfile

    if progress:
        stage = MODELS_DIR / "upload" / (VARIANTS[variant]["out"] + "_progress")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        with tarfile.open(stage / PROGRESS, "w") as t:
            t.add(out, arcname=".", filter=lambda i: None if i.name.endswith(".tmp") else i)
        meta = {
            "title": f"ntl {VARIANTS[variant]['out'].replace('_', '-')}",
            "id": dataset_ref(variant, user),
            "licenses": [{"name": "CC0-1.0"}],
        }
        (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    else:
        stage = package(variant, user=user, out=out, quiet=True)
    r = kaggle_cli(
        "datasets", "version", "-p", str(stage), "-m", note, "--dir-mode", "tar",
        "--delete-old-versions",
    )  # fmt: skip
    text = (r.stdout + r.stderr).lower()
    # Kaggle answers 403, not 404, for a dataset of yours that does not exist yet
    missing = any(k in text for k in ("not found", "404", "403", "forbidden"))
    if (r.returncode != 0 or "error" in text) and missing:
        r = kaggle_cli("datasets", "create", "-p", str(stage), "--dir-mode", "tar")
    text = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and "error" not in text.lower()
    print(f"[sync {time.strftime('%H:%M:%S')}] {note}: {'uploaded' if ok else 'FAILED'}\n{text[-800:]}",
          flush=True)  # fmt: skip
    return ok


def newest_change(out: Path) -> float:
    return max((f.stat().st_mtime for f in out.rglob("*") if f.is_file()), default=0.0)


def sync_loop(variant: str, user: str, out: Path, every_min: float, stop=None) -> None:
    """Upload ``out`` whenever it has changed, until the run has finished or ``stop`` is set."""
    sent = 0.0
    while not (stop and stop.is_set()):
        if (out / "oof_logits.parquet").exists():
            return  # the final upload is the caller's
        changed = out.exists() and newest_change(out) > sent
        # wait until nothing has been written for a minute, so no file is caught half-written
        if changed and time.time() - newest_change(out) > 60:
            stamp = newest_change(out)
            parts = sorted(p.name for p in out.iterdir())
            note = f"progress: {', '.join(parts)}"[:200]
            if upload(variant, user, out, note, progress=True):
                sent = stamp
        if stop:
            stop.wait(60 * every_min)
        else:
            time.sleep(60 * every_min)


@app.command()
def sync(
    variant: str,
    user: str = typer.Option(..., help="Your Kaggle username (owner of the private dataset)."),
    out: Path | None = None,
    every_min: float = typer.Option(10, help="How often to look for changes."),
    once: bool = typer.Option(False, help="Upload the finished run now and stop."),
):
    """Upload the output folder to the private Kaggle dataset whenever it changes, until the run
    has finished (``oof_logits.parquet`` written), so a lost machine loses at most one save."""
    v = VARIANTS[variant]
    out = out or MODELS_DIR / v["out"]
    if once:
        raise SystemExit(0 if upload(variant, user, out, "final") else 1)
    sync_loop(variant, user, out, every_min)


@app.command()
def restore(
    variant: str,
    user: str = typer.Option(..., help="Your Kaggle username (owner of the private dataset)."),
    out: Path | None = None,
) -> str:
    """Download the run's last backup into the output folder, for ``run --resume``. Returns
    "none" (no dataset yet), "restored", or "finished" (the dataset holds a finished run; nothing
    is downloaded). Fails, instead of letting the run start over, if a backup exists but cannot
    be fetched."""
    import tarfile

    v = VARIANTS[variant]
    out = out or MODELS_DIR / v["out"]
    ref = dataset_ref(variant, user)
    for attempt in range(4):
        r = kaggle_cli("datasets", "files", ref, "--csv", "--page-size", "200")
        text = r.stdout + r.stderr
        if "name,size" in r.stdout:
            break
        if any(k in text.lower() for k in ("403", "404", "not found", "have access")):
            print(f"nothing to restore: no dataset {ref} yet")
            return "none"
        time.sleep(30 * (attempt + 1))  # a network error must not look like "no backup"
    else:
        raise SystemExit(f"cannot list {ref}:\n{text[-600:]}")
    names = [line.split(",")[0] for line in r.stdout.strip().splitlines()[1:]]
    if any(n.endswith("oof_logits.parquet") for n in names):
        print(f"{ref} holds a finished run")
        return "finished"
    if PROGRESS not in names:
        raise SystemExit(f"{ref} has neither {PROGRESS} nor a finished run: {names}")
    got = REMOTE_DIR / "restore"
    if got.exists():
        shutil.rmtree(got)
    for attempt in range(4):
        r = kaggle_cli("datasets", "download", ref, "-f", PROGRESS, "-p", str(got))
        if (got / PROGRESS).exists() and tarfile.is_tarfile(got / PROGRESS):
            break
        time.sleep(30 * (attempt + 1))
    else:
        raise SystemExit(f"cannot download {PROGRESS} from {ref}:\n{(r.stdout + r.stderr)[-600:]}")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(got / PROGRESS) as t:
        t.extractall(out, filter="data")
    shutil.rmtree(got)
    print(f"restored {ref} into {out}: {sorted(p.name for p in out.iterdir())}")
    return "restored"


STATE_FILE = REMOTE_DIR / "state.txt"  # one line: where the unattended run is


def mark(state: str) -> None:
    REMOTE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(f"{state} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[unattended] {state}", flush=True)


@app.command()
def unattended(
    variant: str,
    user: str = typer.Option("", help="Your Kaggle username; default: the token's owner."),
    out: Path | None = None,
    save_every_min: float = typer.Option(20, help="How often stage 1 saves itself."),
    sync_every_min: float = typer.Option(5, help="How often the backup looks for new saves."),
):
    """The whole full run for a machine that can go away (Colab), safe to start again at any
    time: restore the last backup from the private Kaggle dataset, train from it while backing
    up every new save, then upload the result. Nothing is written outside this machine's disk
    and Kaggle. The last line of data/remote/state.txt says how far it is."""
    import kagglehub

    v = VARIANTS[variant]
    out = out or MODELS_DIR / v["out"]
    user = user or kagglehub.whoami()["username"]
    me = [sys.executable, "-m", "name_that_llm.remote_train"]
    if not (out.exists() and any(out.iterdir())):  # a fresh machine
        mark("RESTORING")
        if restore(variant, user=user, out=out) == "finished":
            mark("DONE_UPLOADED")
            return
    if not (out / "oof_logits.parquet").exists():
        stop = threading.Event()
        syncer = threading.Thread(
            target=sync_loop, args=(variant, user, out, sync_every_min, stop), daemon=True
        )
        syncer.start()
        mark("TRAINING")
        try:
            r = subprocess.run(
                [*me, "run", variant, "--mode", "full", "--out", str(out), "--resume",
                 "--save-every-min", str(save_every_min)],
                check=False, cwd=PROJ_ROOT,
            )  # fmt: skip
        finally:
            stop.set()
            syncer.join()  # let an upload under way finish
        if r.returncode != 0:
            mark("FAILED")
            raise SystemExit(r.returncode)
    mark("UPLOADING")
    for attempt in range(5):
        if upload(variant, user, out, "final"):
            mark("DONE_UPLOADED")
            print(f"the member is in the private dataset {dataset_ref(variant, user)}")
            return
        time.sleep(60 * (attempt + 1))
    mark("UPLOAD_FAILED")
    raise SystemExit(1)


@app.command()
def package(
    variant: str,
    user: str = typer.Option(..., help="Your Kaggle username (owner of the new dataset)."),
    out: Path | None = None,
    quiet: bool = typer.Option(False, hidden=True),
) -> Path:
    """Wrap the output folder for upload as a private Kaggle dataset. The submission notebook finds
    a member by its folder name (lora_out*), so the folder keeps the training notebook's name."""
    v = VARIANTS[variant]
    out = out or MODELS_DIR / v["out"]
    if not quiet:
        assert (out / "meta.json").exists(), f"no finished run in {out}"
    stage = MODELS_DIR / "upload" / v["out"]
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(out, stage / v["out"], ignore=shutil.ignore_patterns("*.tmp"))
    slug = v["out"].replace("_", "-")
    meta = {
        "title": f"ntl {slug}",
        "id": dataset_ref(variant, user),
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    if quiet:
        return stage
    print(f"staged in {stage}. Upload (private by default) with:\n"
          f"  kaggle datasets create -p {stage} --dir-mode tar\n"
          f"then attach {user}/{slug} to the submission notebook.")  # fmt: skip
    return stage


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
            "2. **Secrets** (key icon on the left): add `KAGGLE_API_TOKEN` (a token from "
            "kaggle.com/settings → API Tokens; or the legacy `KAGGLE_USERNAME` and `KAGGLE_KEY`) and "
            "give this notebook access. You must have joined the competition on Kaggle.\n"
            "3. Run the timing run first (`MODE = 'timing'`, ~10 min), check its s/step, then "
            "set `MODE = 'full'` and run all again.\n\n"
            "Nothing is written to Google Drive. The full run backs itself up to a private "
            "Kaggle dataset (stage 1 every 20 minutes, each fold as it finishes) and, when "
            "started again after a disconnect, restores that backup and continues from it. "
            "`scripts/colab/` starts and watches the same run without this tab."
        ),
        code(
            f"VARIANT = {variant!r}\n"
            "MODE = 'timing'  # 'timing' (~10 min speed check) or 'full'"
        ),
        md("## Setup"),
        code(
            "import os\n"
            "import subprocess\n\n"
            "from google.colab import userdata\n\n"
            "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', "
            "'--format=csv'],\n"
            "                     capture_output=True, text=True).stdout)\n"
            "# a new-style API token (KGAT_...); else the legacy username + key pair\n"
            "try:\n"
            "    os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN').strip()\n"
            "except userdata.SecretNotFoundError:\n"
            "    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')\n"
            "    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')"
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
            "The timing run writes to the runtime's disk and starts fresh each time. The full "
            "run restores the last backup from your private Kaggle dataset "
            f"`{v['out'].replace('_', '-')}` (if there is one), continues from it, uploads every "
            "new save, and uploads the finished member at the end. Rerun this cell after a "
            "disconnect; attach the dataset to the submission notebook when it is done."
        ),
        code(
            "if MODE == 'full':\n"
            "    !uv run python -m name_that_llm.remote_train unattended {VARIANT}\n"
            "else:\n"
            "    !rm -rf /content/timing_out\n"
            "    !uv run python -m name_that_llm.remote_train run {VARIANT} --mode timing "
            "--out /content/timing_out"
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
