# AGENTS.md

Guidance for coding agents working in this repository. Read [README.md](README.md) first: it holds the
approach, the leaderboard history and the exact commands for every pipeline.

## What this is

A solution to the Kaggle code competition **Name That LLM**: say which of 18 LLMs wrote a passage.
The metric is **balanced accuracy**. The submission is a Kaggle notebook that runs offline on two T4s in
at most 9 hours and writes `submission.parquet`.

The submitted model is a blend of log-probabilities:

- a TF-IDF + stylometry LinearSVC (`features.py`, `modeling/train.py`)
- 5-fold LoRA classifiers on Qwen3 4B/8B/14B and Mistral-7B (`lora.py`, `lora_ddp.py`)
- a GLM-5.3 likelihood member, scored only on the passages the blend thinks are most GLM-like
  (`glm_member.py`, `glm_ll/`)
- CHEN Han's native embedding experts (`native_member.py`, `native_q8.py`, `native_wemm4.py`)

Sinkhorn class balancing (`modeling/postprocess.py`) runs on the blend at the end.

## Layout

```
name_that_llm/              the package
  features.py, dataset.py     SVM features; the Open Pool sample
  lora.py, lora_ddp.py        LoRA model and multi-GPU/TPU training (copied into notebooks)
  glm_member.py               GLM likelihood member (copied into the submission notebook)
  native_*.py                 prompt groups and embedding experts (copied into the notebook)
  glm_ll/                     THIRD-PARTY ntl-segrun pipeline, copied unchanged
  kaggle_notebook.py          builds the submission notebook
  kaggle_lora_train.py        builds the LoRA training notebooks (VARIANTS)
  kaggle_glm_features.py      builds the GLM-likelihood-on-train.csv notebooks (GPU or --tpu)
  remote_train.py             LoRA training off Kaggle (no longer used, see Compute below)
  modeling/                   out-of-fold measurement: train, blend, lora_blend, glm_blend,
                              native_blend, testlike (test-like validation), glm_compare
notebooks/                  GENERATED Kaggle notebooks (+ 1.0-eda.ipynb)
reports/                    metrics JSON/CSV per run, figures
references/                 design notes and plans, one per step
gui/                        static browser GUI explaining the blend (build_data.py -> data.js)
scripts/colab/              Colab runtime helpers (no longer used)
data/, models/*.joblib      not tracked: raw data, Open Pool, interim logits, trained models
```

`plots.py` and `docs/` are untouched Cookiecutter template stubs.

## Setup and commands

Python 3.12 with uv, on Windows (Git Bash or PowerShell). The venv interpreter is `.venv/Scripts/python.exe`.

```bash
uv sync                      # base deps
uv sync --group lora         # + torch/transformers, only needed to run LoRA code locally
make lint                    # ruff format --check && ruff check
make format                  # ruff check --fix && ruff format
```

- Ruff: line length 99, isort with `force-sort-within-sections`, scope `name_that_llm/**/*.py`.
  `glm_ll/` is excluded and must stay that way.
- There is no pytest suite. To check changes:
  - **Builders:** rebuild the notebook and look at the diff.
  - **LoRA code:** run the generated training notebook off Kaggle. It does a CPU smoke test on a tiny
    random model (`data/interim/tiny_*`).
  - **Modelling changes:** run the matching `modeling/*` script and compare its out-of-fold number
    with the one in `reports/`.
- The GUI preview is the `gui` entry in `.claude/launch.json` (http.server on port 8765).

Data the scripts expect (none of it is tracked):

- `data/raw/`: `train.csv`, `test.csv`, `sample_submission.parquet`
- `data/external/open_pool/ntl_open_pool.parquet`
- `data/external/native_q8/`: CHEN Han's out-of-fold logits
- `data/interim/<run>/`: each training run's `oof_logits.parquet` / `holdout_logits.parquet`
- `data/interim/glm_ll*`: GLM features

## Rules that matter

### Validation

- **All CV is `GroupKFold(5)` by `prompt_id`.** Every model answered every prompt, so any other split
  leaks topic between folds. The LoRA folds, the GLM member's folds and CHEN Han's published folds all
  hold out whole prompts. Out-of-fold numbers are only comparable when they use those folds.
- **Tune blend weights on out-of-fold predictions, never on the public leaderboard alone.** Report a
  gain together with its paired prompt-bootstrap interval, as `glm_blend` and `native_blend` do.
- **Out-of-fold gains from prompt-group steps did not transfer.** Centring the blend within a group and
  giving each passage in a group a different model both gained out-of-fold and lost on the public
  board (−0.006 and −0.021). The hidden test is built differently from `train.csv`.
  `modeling/testlike.py` simulates that test. Check any group-dependent idea there before trusting it.
- Write each measurement to `reports/` under a descriptive `--tag`, and keep the README results table
  current when a submission is scored.

### Code copied into notebooks

Kaggle runs notebooks offline, so they cannot import this package. The builders copy source into the
notebook with `inspect.getsource`:

- `features.py`, the softmax and Sinkhorn functions of `postprocess.py`, `glm_member.py`,
  `native_member.py`, `native_q8.py`, `native_wemm4.py` and `glm_ll/*` go into the submission notebook.
- `lora.py` and `lora_ddp.py` go into the training notebooks.

Consequences:

- Keep these modules self-contained: stdlib, numpy/pandas/sklearn, torch and transformers only (no
  peft, no `name_that_llm` imports). `lora_ddp.py` imports `lora` as a sibling module.
- **After changing any of them, rebuild the notebooks that use them.** Never hand-edit a notebook in
  `notebooks/`: it is regenerated from its builder, and hand edits are lost.
- Name a submission notebook after its members and steps, as the existing ones are named, and keep each
  version that was submitted.

### Third-party code

Some files are copied unchanged from public Kaggle work. Each one says so in its header:

- `glm_ll/*` and `glm_ll/segrun/*` (José Padilla's ntl-segrun)
- `glm_ll/token_features.py` (lumibee)
- `native_q8.py` and `native_wemm4.py` (CHEN Han's experts, via lightningv08)
- `native_member.capped_agglo`

Rules for these files:

- Do not reformat, lint-fix or "clean up" them.
- `glm_ll/segrun/**` must keep **LF** line endings (`.gitattributes`). `kaggle_glm_features.py` checks
  their sha256 and fails on CRLF.
- Credit the author by name in the file header and in the README when you add or use someone's work.

### Kaggle runtime

- **Time limit:** the submission has 9 h. The notebook stops new GLM work at `DEADLINE_S = 8.5 h` and
  writes a submission without the GLM member before the GLM stage starts. Keep that fallback whenever
  you add a slow stage.
- **Inputs:** the notebooks find their inputs by globbing `/kaggle/input/**`. Make each pattern specific
  enough to match one thing. Qwen3-Embedding-8B once matched the 8B LoRA's base-model lookup and was
  loaded in its place for four versions (see `find_base`).
- **Attaching inputs:** `--push-dir <dir>` takes a pulled notebook (`kaggle kernels pull <id> -p <dir> -m`).
  It writes the new notebook there and adds the required datasets, models and kernels to its
  `kernel-metadata.json`.
- **Runs longer than one session:** they save progress and stop cleanly before 12 h (`stop_after_min`,
  `progress.bin`). Wait for a version to finish before pushing the next one, because a new version does
  not cancel a running one.
- **Drift:** GLM-5.3 features computed at test time drift slightly from the stored train features,
  because its sparse attention depends on which passages share a batch. Flash does not drift. The
  interactive check tolerates this; do not "fix" it by tightening the asserts.

## Compute and storage (user preferences)

- **Training runs on Kaggle GPU T4 x2 only.** Using the weekly GPU quota freely is fine. Do not use Colab
  or create Colab runtimes. `remote_train.py` and `scripts/colab/` remain only as a record.
- **Never use Google Drive.** Back up run state and outputs to **private Kaggle datasets**.
- Push notebooks with the Kaggle CLI, for example `uvx --with six kaggle kernels push -p <dir>`.
- **Ask before submitting to the competition.** Daily submissions are limited, and each one is a
  public-board data point the user wants to choose.
- Never read, print or commit credentials: `.env`, `~/.kaggle/kaggle.json`, `KAGGLE_API_TOKEN`.
- Never commit competition data. `data/` and `gui/passages.js` are git-ignored for this reason.

## Writing style

- Each module starts with a docstring that says what the module does and why, and gives the exact
  `uv run python -m ...` command that runs it. Keep that docstring current.
- Comments and docs are plain, specific prose with the measured numbers (scores, timings, intervals).
- Commit subjects are imperative and name the public LB score when there is one, for example
  "Add the 5-fold Qwen3-14B LoRA to the blend; scores 0.629 public". The body lists what changed and
  what was measured.
- Plans and design notes go in `references/` as dated markdown files.
