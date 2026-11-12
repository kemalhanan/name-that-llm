# name-that-llm

My solution for the Kaggle code competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). The task is to tell which of 18 LLMs wrote a passage of text, scored by balanced accuracy.

**Best public LB: 0.607**. That submission blends a TF-IDF LinearSVC with two Qwen3 LoRA classifiers (4B and 8B), each trained 5-fold.

## Approach

1. **TF-IDF + stylometry SVM** ([features.py](name_that_llm/features.py), [modeling/train.py](name_that_llm/modeling/train.py))
   - Character 1-5-grams and word 1-2-grams, plus 16 punctuation and style rates per 100 words. A LinearSVC classifies them.
   - Punctuation and quote style separate model families almost on their own, so the text keeps its punctuation and unicode.
   - Trained on `train.csv` plus a class-balanced, style-matched sample of the community [Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) ([dataset.py](name_that_llm/dataset.py)).
2. **LoRA classifiers on Qwen3** ([lora.py](name_that_llm/lora.py), [lora_ddp.py](name_that_llm/lora_ddp.py))
   - Only the first 24 decoder layers of Qwen3-4B-Base or Qwen3-8B-Base are kept. Every linear layer gets a LoRA adapter (r=16, alpha=32).
   - A linear head reads mean-pooled hidden states from several depths.
   - Training has two stages: 1 epoch on a large Open Pool sample (186k passages for the 4B, 152k for the 8B), then 2 epochs on `train.csv`.
   - Runs in fp16 with DDP on Kaggle's two T4 GPUs.
   - Each model has 6 adapters: one fitted on all of `train.csv` plus one per prompt-grouped fold. At test time their log-probabilities are averaged.
3. **Blend** ([modeling/lora_blend.py](name_that_llm/modeling/lora_blend.py), [modeling/postprocess.py](name_that_llm/modeling/postprocess.py))
   - The models' log-probabilities are added: `log_softmax(2·svm) + 1.0·log_softmax(4B) + 2.0·log_softmax(8B)`.
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
| **SVM + 5-fold 4B (186k) + 5-fold 8B (152k) only** | **0.607** |

The final blend scores 0.6305 balanced accuracy out-of-fold. The best single model is the 5-fold 8B LoRA at 0.615. Per-run metrics are in [reports/](reports/).

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
uv run python -m name_that_llm.kaggle_notebook --lora lora_out_4b_full=1.0 --lora lora_out_8b_full=2.0 --t-svm 2.0 --path notebooks/kaggle_submission_lora_4bfull_8bfull.ipynb
```

- **Training notebook:** set `MODE = "timing"` for a 15-minute speed check, or `"full"` for the real run (about 9 h for the 8B). It needs the competition data, the Open Pool and the Qwen3 base model as inputs.
- **Submission notebook:** needs the training runs' outputs attached as inputs.
- **TPU:** add `--tpu` to build a training notebook for Kaggle's TPU v5e-8 instead of the two T4s, for example `kaggle_lora_train_4b_full_tpu.ipynb`. It trains in bf16 with fixed tensor shapes and writes the same outputs to a folder ending in `_tpu`. It needs internet on, so it can install PyTorch/XLA if the image lacks it.
- **Off Kaggle:** `name_that_llm/remote_train.py` runs the same training on any CUDA machine, such as a rented A100 or H100, without using Kaggle GPU quota. It trains in bf16 on those GPUs and keeps the global batch at 8. Upload the output as a private dataset and attach it to the submission notebook:

  ```bash
  uv sync --group lora
  uv run python -m name_that_llm.remote_train fetch mistral7b-full
  uv run python -m name_that_llm.remote_train run mistral7b-full --mode timing
  uv run python -m name_that_llm.remote_train run mistral7b-full --mode full
  uv run python -m name_that_llm.remote_train package mistral7b-full --user <kaggle user>
  ```

  `fetch` needs Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`). Outputs go to `data/remote/out/`.
- **Colab:** [notebooks/colab_lora_train_mistral7b_full.ipynb](notebooks/colab_lora_train_mistral7b_full.ipynb) runs those steps on a Colab A100 and saves the full run to Google Drive ([open in Colab](https://colab.research.google.com/github/kemalhanan/name-that-llm/blob/main/notebooks/colab_lora_train_mistral7b_full.ipynb)). Build it for another variant with `uv run python -m name_that_llm.remote_train colab <variant>`.

Each training run writes held-out logits. Compare and blend them with:

```bash
uv run python -m name_that_llm.modeling.lora_blend <oof_logits.parquet>... --names <output dirs> --tag <name>
```

Design notes for each step are in [references/](references/).

## Layout

```
data/            raw competition data, external Open Pool, interim logits (not tracked)
name_that_llm/   package: features, SVM training, LoRA model, blend, notebook builders
notebooks/       EDA and the generated Kaggle training / submission notebooks
references/      plans and notes on approaches
reports/         CV and blend metrics as JSON
```

Project template: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).
