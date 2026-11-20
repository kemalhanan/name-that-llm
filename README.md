# name-that-llm

My solution for the Kaggle code competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). The task is to tell which of 18 LLMs wrote a passage of text, scored by balanced accuracy.

**Best public LB: 0.629**. That submission blends a TF-IDF LinearSVC with four LoRA classifiers (Qwen3 4B, 8B and 14B, and Mistral-7B), each trained 5-fold.

## Approach

1. **TF-IDF + stylometry SVM** ([features.py](name_that_llm/features.py), [modeling/train.py](name_that_llm/modeling/train.py))
   - Character 1-5-grams and word 1-2-grams, plus 16 punctuation and style rates per 100 words. A LinearSVC classifies them.
   - Punctuation and quote style separate model families almost on their own, so the text keeps its punctuation and unicode.
   - Trained on `train.csv` plus a class-balanced, style-matched sample of the community [Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) ([dataset.py](name_that_llm/dataset.py)).
2. **LoRA classifiers on Qwen3 and Mistral-7B** ([lora.py](name_that_llm/lora.py), [lora_ddp.py](name_that_llm/lora_ddp.py))
   - Only the first 24 decoder layers of Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-14B-Base or Mistral-7B-v0.1 are kept. Every linear layer gets a LoRA adapter (r=16, alpha=32).
   - A linear head reads mean-pooled hidden states from several depths.
   - Training has two stages: 1 epoch on a large Open Pool sample (186k passages for the 4B, 152k for the 8B, 193k for the 14B, 136k for the Mistral-7B), then 2 epochs on `train.csv`.
   - The Qwen3 4B and 8B run in fp16 with DDP on Kaggle's two T4 GPUs; the Mistral-7B ran in bf16 on one Colab A100.
   - The 14B's 24-layer trunk is 16.2 GiB in fp16, more than one T4 holds. It is split instead: embeddings and layers 0-9 on the first GPU, layers 10-23 on the second, in a single process. Its run began on a Colab A100 and finished on Kaggle.
   - Each model has 6 adapters: one fitted on all of `train.csv` plus one per prompt-grouped fold. At test time their log-probabilities are averaged.
3. **Blend** ([modeling/lora_blend.py](name_that_llm/modeling/lora_blend.py), [modeling/postprocess.py](name_that_llm/modeling/postprocess.py))
   - The models' log-probabilities are added: `log_softmax(2·svm) + 0.25·log_softmax(4B) + 0.5·log_softmax(8B) + 0.5·log_softmax(Mistral-7B) + 1.0·log_softmax(14B)`.
   - Sinkhorn class balancing at strength 0.3 then evens out how often each label is predicted.
   - Weights are picked on out-of-fold predictions.

All cross-validation is `GroupKFold(5)` by `prompt_id`. Every model answered every prompt, so a random split would leak topic between folds.

## Results

| Submission | Public LB |
|---|---|
| SVM + LoRA 4B | 0.537 |
| SVM + LoRA 4B + LoRA 8B | 0.544 |
| + a 4B LoRA trained on a 96k pool sample | 0.563 |
| + an 8B LoRA trained on an 88k pool sample (4 LoRA members) | 0.569 |
| + a 5-fold 4B LoRA trained on a 186k pool sample | 0.589 |
| SVM + 5-fold 4B (186k) + 5-fold 8B (152k) only | 0.607 |
| + a 5-fold Mistral-7B LoRA (136k) | 0.611 |
| **+ a 5-fold Qwen3-14B LoRA (193k)** | **0.629** |

The final blend scores 0.6531 balanced accuracy out-of-fold (0.6361 without the 14B, 0.6305 without the Mistral-7B as well). The best single model is the 5-fold 14B LoRA at 0.630; the 8B scores 0.615, the 4B 0.597 and the Mistral-7B 0.595. Per-run metrics are in [reports/](reports/).

## Reproducing

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Put the competition files (`train.csv`, `test.csv`, `sample_submission.parquet`) in `data/raw/`. Put `ntl_open_pool.parquet` from the Open Pool dataset in `data/external/open_pool/`.

Local SVM pipeline:

```bash
uv run python -m name_that_llm.dataset --n-per-label 2000
uv run python -m name_that_llm.modeling.train
uv run python -m name_that_llm.modeling.postprocess --tag pool2000_w1
```

The LoRA models train on Kaggle (GPU T4 x2, internet off). These scripts build self-contained notebooks into [notebooks/](notebooks/):

```bash
uv run python -m name_that_llm.kaggle_lora_train main --variant 8b-full
uv run python -m name_that_llm.kaggle_notebook --lora lora_out_4b_full=0.25 --lora lora_out_8b_full=0.5 --lora lora_out_mistral7b_full=0.5 --lora lora_out_14b_full=1.0 --t-svm 2.0 --path notebooks/kaggle_submission_lora_4bfull_8bfull_mistral7bfull_14bfull.ipynb
```

- **Training notebook:** set `MODE = "timing"` for a 15-minute speed check, or `"full"` for the real run (about 9 h for the 8B). It needs the competition data, the Open Pool and the base model as inputs.
- **Runs longer than one session:** the `14b-full` variant needs about 30 h on two T4s, more than Kaggle's 12-hour session. Its notebook saves stage 1 every 20 minutes and each fold as it finishes, and stops cleanly after 11 hours so Kaggle keeps the output. To continue, attach the previous version's output (or a dataset holding its `progress.bin`) as an input and run it again; build it with `--need-saved` to make it fail at the start if no saved state is attached. Wait for one version to finish before pushing the next: a new version does not cancel a running one.
- **Submission notebook:** needs the training runs' outputs and their base models attached as inputs. A member trained off Kaggle is attached as its private dataset.
- **TPU:** add `--tpu` to build a training notebook for Kaggle's TPU v5e-8 instead of the two T4s, for example `kaggle_lora_train_4b_full_tpu.ipynb`. It trains in bf16 with fixed tensor shapes and writes the same outputs to a folder ending in `_tpu`. It needs internet on, so it can install PyTorch/XLA if the image lacks it.
- **Off Kaggle:** `name_that_llm/remote_train.py` runs the same training on any CUDA machine, such as a rented A100 or H100, without using Kaggle GPU quota. It trains in bf16 on those GPUs and keeps the global batch at 8. Upload the output as a private dataset and attach it to the submission notebook:

  ```bash
  uv sync --group lora
  uv run python -m name_that_llm.remote_train fetch mistral7b-full
  uv run python -m name_that_llm.remote_train run mistral7b-full --mode timing
  uv run python -m name_that_llm.remote_train run mistral7b-full --mode full
  uv run python -m name_that_llm.remote_train package mistral7b-full --user <kaggle user>
  ```

  `fetch` needs a Kaggle API token in `KAGGLE_API_TOKEN` (or the legacy `KAGGLE_USERNAME` and `KAGGLE_KEY`). Outputs go to `data/remote/out/`.
- **Colab:** [notebooks/colab_lora_train_mistral7b_full.ipynb](notebooks/colab_lora_train_mistral7b_full.ipynb) runs the full run on a Colab A100 with `remote_train unattended` ([open in Colab](https://colab.research.google.com/github/kemalhanan/name-that-llm/blob/main/notebooks/colab_lora_train_mistral7b_full.ipynb)). Nothing is written to Google Drive: stage 1 saves every 20 minutes and each fold as it finishes, every save is backed up to a private Kaggle dataset, and a restarted run restores the last backup and continues. Build it for another variant with `uv run python -m name_that_llm.remote_train colab <variant>`.
- **Colab without a browser tab:** [scripts/colab/](scripts/colab/) starts the same run on a runtime created through the Colab API. `push.py <runtime url> <proxy token>` copies the working tree and your Kaggle token to the runtime, `ensure.sh` reports the run and starts it if it is not running, and `boot.sh` installs and runs it.

GLM-5.3 likelihood features (two of the 18 labels are open-weight models, scored under their real weights with José Padilla's `ntl-segrun`, see [name_that_llm/glm_ll/](name_that_llm/glm_ll/)):

```bash
uv run python -m name_that_llm.kaggle_glm_features --mode timing
uv run python -m name_that_llm.kaggle_glm_features --tpu --mode timing
uv run python -m name_that_llm.modeling.glm_compare <gpu output dir> <tpu output dir>
uv run python -m name_that_llm.modeling.glm_blend <features dir>
```

- The first notebook runs on two T4s with the author's CUDA build. The `--tpu` one spends TPU quota instead: llama.cpp cannot use the TPU chips, so it builds `ntl-segrun` for the TPU VM's host CPUs (internet on), checks the build with the author's segmented-vs-whole validation, and runs one thread per core. `probe` mode only checks the machine and the inputs; `SHARD = (k, n)` splits a full run over sessions.
- `glm_compare` checks that CPU and GPU runs give the same features on the 40 passages both timing runs score, since a submission would score the test on GPUs.
- `glm_blend` fits the likelihood member fold by fold and reports its gain on the out-of-fold blend.

Each training run writes held-out logits. Compare and blend them with:

```bash
uv run python -m name_that_llm.modeling.lora_blend <oof_logits.parquet>... --names <output dirs> --tag <name>
```

Design notes for each step are in [references/](references/).

## Model explorer

[gui/](gui/) is a small browser GUI that explains the blend in five steps: a guessing game against the model, a step-by-step walk-through of one passage, a line-up showing how often each of the 18 models is caught, a mixer that recomputes the out-of-fold score as you move the judges' weights, and a list of next moves. Build its data from the reports and out-of-fold predictions, then serve the folder:

```bash
uv run python gui/build_data.py
uv run python -m http.server 8765 --directory gui
```

Open http://localhost:8765. `gui/passages.js` holds a sample of competition passages and is not tracked.

## Layout

```
gui/             browser GUI: game, walk-through, blind spots, mixer, next moves
data/            raw competition data, external Open Pool, interim logits (not tracked)
name_that_llm/   package: features, SVM training, LoRA model, blend, notebook builders
notebooks/       EDA and the generated Kaggle training / submission notebooks
references/      plans and notes on approaches
reports/         CV and blend metrics as JSON
```

Project template: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).
