# Name That LLM: what the top public approaches do

Notes from the competition pages, the top public notebooks and the main discussion threads, read on
2026-11-03. Leaderboard numbers are the **public** LB unless stated otherwise.

## The competition in brief

- 18 classes, balanced accuracy. Hidden test of ~4,000 passages, **all from prompts not in train.csv**;
  the class mix differs between the public and private LB.
- **Notebook-only submissions**: no internet, ≤ 9 h on CPU or GPU (Kaggle gives 2× T4), output
  `submission.parquet`. Trained weights are attached as Kaggle datasets/models.
- External data and freely available pretrained models are allowed (subject to a vague
  "reasonable cost" clause). 5 submissions/day, 2 final picks. Final deadline 2027-01-13.

## Where the scores are

| Approach | Public LB |
|---|---|
| Leaderboard top (Hod Carriers) | 0.747 |
| Best public notebooks: big ensembles of LoRA-tuned LLMs + external pool | 0.660–0.667 |
| nlm-v27 (3–4 LoRA members retrained on a 120k pool) | 0.651 |
| Single two-stage LoRA, Qwen3-8B (stage 1 on pool, stage 2 on train) + Sinkhorn | 0.549 |
| Qwen3-14B-Base LoRA, train only | 0.466 |
| 4-view n-gram SVM, train + pool | 0.423 |
| Char + word TF-IDF + LR/SVM, train only (≈ our baseline) | 0.353–0.361 |
| **Ours: TF-IDF + style blend, train only (grouped CV)** | **0.366 (CV)** |

The gap to the top is not one trick. It comes from three things, roughly in order of size:

1. **External data** in the host's style: worth about +0.06 for an n-gram model and +0.11–0.12 for
   LLM-based models, by several competitors' estimates.
2. **Fine-tuned LLM encoders** instead of n-grams: about +0.1 at equal data.
3. **Diverse ensembles** plus a few post-processing steps: the last +0.05–0.1.

## 1. External data

- **[Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)** (crtproduction)
  is the community's merged pool: v12 has 211k passages from 59 public source datasets
  (thread 752766). Columns that matter:
  - `match_weight`: reweights passages within each label so typography and length match the real
    passages. Sampling by it gave +0.026 in the author's test.
  - `prompt_canon`: a recovered prompt id for 99% of rows, for grouped splits.
  - `panel_gaps.csv`: prompts missing some labels.
- **The style mismatch matters more than volume.** Raw external answers have line breaks (94%) and
  bold (60%), and a median of 174 words against 101. Added naively, they *hurt* a TF-IDF model
  (0.326 → 0.279, thread 749447). Filter to single plain paragraphs and match length/typography.
- **Label noise:** since 2026-10-08, non-Vertex APIs answer `gemini-3.7` requests with 3.8. Open Pool
  keeps only Vertex-AI-routed 3.7 rows in the main file. Most under-covered labels:
  `gemini-3.7-flash`, `qwen3.8-max`, `kimi-k3`, `claude-fable-5`.
- Generating your own data through the APIs is allowed, and the top teams are almost certainly doing it.
  Kaggle Benchmarks gives some free API credit.

## 2. Models

- **Two-stage LoRA classifiers** are the backbone of every strong notebook:
  - Base models: Qwen3-8B/14B-Base, Qwen2.5-7B/14B, Llama-3.1-8B. Often truncated to the first ~24–32
    layers ("LoRA24").
  - LoRA r16, alpha 32 on all linear layers. Mean-pool the hidden states of several layers, then add
    a classification head.
  - Stage 1: one epoch on the pool. Stage 2: two epochs on train.csv. `max_len` 384 is enough.
  - Training was done on H100s outside Kaggle; inference fits 2× T4 in fp16 (no bf16 on T4).
- **Earlier layers carry more authorship signal than late ones** (thread 750462: 0.39 vs 0.31 with
  frozen Qwen features).
- **The n-gram SVM still earns its place in the ensemble**: removing it cost −0.011 on the LB, even
  next to members with much higher OOF scores.

## 3. Ensembling and post-processing

- **Diversity beats member strength.** A member trained on a *different* data mix adds more than a
  stronger clone. Retraining a member on the same pool as the others raised its OOF from 0.569 to
  0.605 but *lowered* the LB from 0.651 to 0.647 (thread 753177).
- **Sinkhorn class balancing** of the test predictions: about +0.01 on the LB with no change in CV.
  It makes sense for balanced accuracy.
- **Prompt-centred "twin" heads** (subtracting the prompt-cluster mean): worth up to +0.017 inside the
  big ensembles. But the plain version (clustering + Hungarian assignment) *lost* every time on the LB.
  The hidden test seems to have only a few passages per prompt, so tricks that need full 18-answer
  prompt groups don't transfer.

## Validation lessons

- **Grouped CV by prompt** tracks the LB for train-only models (TF-IDF 0.343 → 0.353; ours should behave
  the same). Random K-fold is badly misleading (0.217 vs 0.369).
- **Pool-trained models read 0.03–0.04 high in grouped CV**, so compare them with each other, not with
  train-only models.
- **One public score is noisy**: identical code has scored 0.643 and 0.651. A single gain under ~0.01 is
  not evidence. The sd of a *difference* is about √(f / 1200), where f is the share of rows on which
  exactly one of the two models is right.
- To judge anything prompt-dependent, score the OOF with prompts cut to **2 answers per prompt**. That
  tracked the LB far better (r ≈ 0.8–0.9) than full 18-answer prompts.
- **Short passages (≤ 60 words) are the hardest.** Training on truncated copies did not help.

## Implications for this project

1. **Cheap, CPU only:** add Open Pool rows (single paragraph, sampled by `match_weight`, Vertex-only
   gemini-3.7) to our TF-IDF/SVM blend, and add Sinkhorn balancing. Expected LB about 0.42–0.45.
2. **First GPU step:** a frozen-feature member (mean-pooled early/middle layers of a Qwen3 base model,
   then logistic regression), which runs on Kaggle's T4s with no fine-tuning.
3. **Main step:** a two-stage LoRA classifier (stage 1 on the pool, stage 2 on train). This needs more
   GPU than Kaggle's free quota comfortably allows for 8B+ models; a smaller base (Qwen3-1.7B/4B) is
   feasible there.
4. Alternatively, start from the public 0.667 notebook and add a member trained on data it hasn't seen.
   Diversity is what moves that ensemble.
