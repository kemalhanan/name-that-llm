# Plan: two-GPU training, 186k pool passages, 5-fold stage 2

Written 2026-11-07, after the SVM + 8B + 4B big-pool + 8B big-pool blend scored **0.569** on the
public LB (held-out fold 0: 0.6185).

## Why

Two findings from the last two rounds:

1. **More pool data is the lever that works.** Going from 36k to 96k pool passages raised the 4B
   from 0.543 to 0.587 on fold 0, and 36k to 88k raised the 8B from 0.535 to 0.594. We still leave
   out more than half of the 211k-passage Open Pool, because one T4 can't get through more in a
   12 h session.
2. **One held-out fold is too noisy to choose blends.** On fold 0 (540 passages) the 4-member blend
   tied the 3-member one (0.6185 each), yet it scored +0.006 higher on the public LB. Blend
   weights fit on 540 passages are close to guesswork at this scale.

This run tackles both: use both T4s to train on more data in the same time, and train stage 2 on
all 5 folds to get held-out scores for all 2,699 training passages.

## The change

Same recipe as before (Qwen3-4B-Base, first 24 layers, LoRA r16 / alpha 32, same learning rates
and epochs). What changes:

- **Both T4s, data-parallel.** The training code is written to a script in `/kaggle/working` and
  started as one process per GPU (DDP, NCCL; the notebook sets the same environment variables
  torchrun would, since torchrun's launcher doesn't run on Windows for the local test). Each GPU takes half of every batch, so
  the global batch stays 8 and the recipe is unchanged. Kaggle counts session time, not time per
  GPU, so this should nearly halve the cost of a run.
- **Pool cap 12,000 per model: 186,299 passages** (88% of the pool). Uncapped would be 211k, with
  gemini-3.8-flash at 22k against gemini-3.7-flash at 3.5k. The cap keeps the largest class at
  3.4x the smallest instead of 6.3x; the class-weighted loss handles the rest. Draw seed 4.
- **Stage 1 saved.** The stage-1 weights go to the output, so more stage-2 variants can be trained
  later without repeating stage 1.
- **Stage 2 on 5 folds plus a full fit.** From the same stage-1 weights: 5 runs, each leaving out
  one prompt-grouped fold and scoring it, then one run on all of train.csv. The folds are
  `GroupKFold(5)` by `prompt_id`, the same split `train.py` uses for the SVM.

Outputs (`lora_out_4b_full/`):

| File | What |
|---|---|
| `oof_logits.parquet` | held-out logits for all 2,699 passages, each from the fold that left it out |
| `fold{0..4}/lora_classifier.safetensors` | the 5 fold adapters |
| `full/lora_classifier.safetensors` | the adapter trained on all of train.csv |
| `stage1/lora_classifier.safetensors` | the stage-1 weights |
| `meta.json` | config, per-fold scores, timings |

## Steps

1. **Generator:** a new `4b-full` variant in `kaggle_lora_train.py` that writes `lora.py` and a
   training script to `/kaggle/working` and runs it with `torchrun`. In the training loop:
   - Every rank builds the same batch list from the same seed and takes its half of each batch.
   - **Overflow check across GPUs:** the "skip this batch" decision for a non-finite fp16 loss is
     shared between the GPUs (`all_reduce`). Otherwise one GPU skips its backward pass while the
     other waits for it, and the run hangs.
   - Rank 0 alone scores held-out passages and writes files.
2. **Local test without GPUs:** the same script with `torchrun --nproc_per_node=2` on CPU (gloo
   backend) and the tiny model. This checks the DDP code, the fold loop and the output files. The
   local GTX 1050 can't run two-GPU NCCL, so the first real two-GPU test is on Kaggle.
3. **Timing run on Kaggle** (MODE `timing`, ~20 min, GPU T4 x2): 60 steps of stage 1 on both GPUs,
   then a forecast. Decision:
   - forecast ≤ 10 h: run the full job as planned;
   - over 10 h: lower the cap to 10,000 (166k) or 8,000 (136k) to fit.
   If the per-step speedup is poor (under 1.4x), try 8 per GPU (global batch 16) in the same
   timing run before deciding.
4. **Full run** (MODE `full`). You attach the same three inputs as before and pick GPU T4 x2.
5. **Blend on all 2,699 passages:** extend `lora_blend` to use every passage that all members
   have scores for. With the SVM and the new 4B, that's all 2,699, so their weights are fit on 5x
   more data. The older LoRA members only have fold 0, so a blend that includes them is still
   checked on fold 0 alone; I'll report both.
6. **Submission notebook:** support a member made of several adapters (average the log-probabilities
   of the 5 fold adapters and the full one). Push, you run it, I ask before submitting.

## Budget

Assuming 2 GPUs give ~1.7x the speed of one (the timing run measures it):

| Item | GPU hours |
|---|---|
| Timing run | ~0.3 |
| Stage 1: 23,288 steps × ~0.87 s | ~5.6 |
| Stage 2: 5 folds × 540 steps + full fit 676 steps, × ~0.87 s | ~0.8 |
| Loading, scoring held-out folds | ~0.2 |
| **Training run** | **~6.6** |
| Submission run (6 adapters for this member) | ~0.4 |

About 21 h of quota are left this week (resets 2026-11-14), so this fits with room for one more
run.

## Go / no-go

- **New 4B on its own folds, mean over 5 folds:** compare fold 0 with the 96k-pool 4B (0.587). If
  it's ≥ 0.60, more data still helps and the 8B is worth the same treatment next week.
- **Fold-0 blend with all members:** if it beats 0.6185, submit it.
- **If two-GPU training fails on Kaggle** (hang, NCCL error), fall back to one GPU with a 5,500 cap
  and keep only the 5-fold part. That still gives the 5x larger blend check.

## Risks

- **First two-GPU run:** DDP can't be tested locally on GPUs. The CPU test covers the logic, and
  the timing run covers NCCL on Kaggle before any long run.
- **Global batch kept at 8 (4 per GPU):** small batches use each GPU less efficiently, so the
  speedup may be well under 2x. The timing run decides whether to switch to 16.
- **A crash late in the run** costs up to ~7 h of quota. The code paths are the same as the
  single-GPU runs that already completed four times, apart from the DDP parts the timing run
  exercises.
- **Inference cost:** the member has 6 adapters instead of 1, so it scores the hidden test set 6
  times. We haven't measured one pass on the hidden test set; the whole 3-LoRA submission already
  fits comfortably, but if time gets tight, drop the full-fit adapter and average the 5 folds.
