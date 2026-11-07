# Plan: retrain the 4B LoRA on a larger Open Pool sample

Written 2026-11-05, after the SVM + 4B + 8B blend scored **0.544** on the public LB (held-out 0.585).

## Why

Our LoRA models train stage 1 on 2,000 pool passages per model (36k), about 17% of the 211k
Open Pool we already have locally. The public evidence says that data, not the backbone or the
blend, is what's missing:

| Source | Single model | Score |
|---|---|---|
| Forum (GoPain), Qwen3-8B LoRA, older smaller pool | held-out prompts | 0.589 |
| same recipe, Open Pool v10 capped per label | held-out prompts | 0.622 (public LB 0.628) |
| alexchilton, own 5-backbone two-stage LoRA blend | public LB | 0.56–0.59 |
| us, Qwen3-4B / Qwen3-8B on 36k | held-out fold 0 | 0.543 / 0.535 |

The forum author's words: nearly all of the gain came from the data. Adding more LoRA models to a
blend moved alexchilton's score by at most 0.004, which matches our +0.007 from adding the 8B.

## The change: one variable

Same recipe, same base (Qwen3-4B-Base, first 24 layers), same hyperparameters, same held-out fold.
Only the stage-1 data changes:

- **Cap 5,500 per model** instead of 2,000: **96,107 passages**. Two models have fewer and give
  all they have: qwen3.8-max (4,566) and gemini-3.7-flash (3,541). Still drawn by `match_weight`, so
  the sample leans to passages whose typography matches the official rows. Class-weighted loss
  already handles the uneven counts.
- **Draw seed 2**, so the sample differs from the 4B (seed 0) and 8B (seed 1) runs.
- **Keep the `flag_kb_proxy` rows** (72k, generated through proxies rather than official APIs), as
  the public notebooks do. Dropping them would halve the GPT classes (48–54% of their rows).
- 1 epoch of stage 1 as before; stage 2 unchanged (2 epochs on train.csv).

The held-out check stays fair: the pool has no passages from train prompts
(`flag_near_train_prompt` and `flag_train_prompt_regen` are 0 on every row), so fold 0's 30 prompts
are unseen in both stages.

## Steps

1. **Generator:** add `n_per_label` to `VARIANTS` and a new variant `4b-big` (cap 5,500, seed 2,
   output `lora_out_4b_big`, notebook `notebooks/kaggle_lora_train_4b_big.ipynb`). Smoke-test
   locally with the tiny model. No timing run: the 4B speed is already measured (1.54 s/step).
2. **Push** as a new Kaggle notebook (quick save, not run). You attach the inputs (competition data,
   Open Pool, Qwen3 4b-base) and start it with MODE `full`.
3. **Meanwhile, on CPU (free):** grouped CV of the SVM with the larger pool (cap 5,500) against the
   current 0.451, to see whether the SVM member gains from the same change. About 3 h locally.
4. **When training finishes:** download the held-out logits, compare the new 4B with the old one on
   the same 540 passages, then rerun the blend search with all members (SVM, old 4B, 8B, new 4B).
5. **Submission notebook** with the picked weights; check the run; submit with your OK.

## Go / no-go after step 4

- **New 4B ≥ 0.58 held out** (old 0.543): data is the lever. Next week, retrain the 8B the same way
  (12k steps × 2.59 s ≈ 8.6 h + stage 2 ≈ 9.7 h, inside the 12 h session limit but tight; a cap of
  5,000 would give more margin).
- **0.55–0.58:** some gain. Blend it in and decide on the 8B from the blend gain.
- **Below 0.55:** data alone doesn't fix our recipe. Look at what else the strong single models do
  differently (fold-averaged adapters, the Qwen3-14B backbone) before spending more quota.

## Budget

| Item | GPU hours |
|---|---|
| Stage 1: 12,014 steps × 1.54 s | 5.1 |
| Stage 2 twice, held-out scoring, model load | ~0.6 |
| **Training run** | **~5.8** |
| Submission run (Kaggle reruns it when scoring) | ~0.3 |

Quota: 22.7 h left this week (7.3 of 30 used), resetting 2026-11-07 00:00 UTC.

## Risks

- **A crash late in the run loses it:** Kaggle keeps outputs only from a finished run, so a failure at
  hour 5 costs 5 h of quota. The same code already ran twice without error, and memory per step
  doesn't change with more data, so the risk is low.
- **One held-out fold is noisy** (540 passages, about ±0.02). Fine for the go/no-go above, not for
  fine distinctions.
- **The public LB is noisy too** (identical code varies by up to 0.009), so judge the new member
  mainly on the held-out fold.
