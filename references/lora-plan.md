# Plan: a fine-tuned LoRA classifier as the next model

Written 2026-11-04, after the TF-IDF + Open Pool notebook scored **0.414** on the public LB
(grouped CV 0.451).

## Where the gain should come from

The public notes (see `competition-approaches.md`) and the "Two-stage LoRA on the public pool"
notebook (nishantdahal, LB 0.592) agree on the recipe and its payoff:

| Model (that notebook's grouped CV) | CV |
|---|---|
| n-gram SVC, train + pool (≈ ours) | 0.435 |
| Qwen3-8B LoRA, pool only (stage 1) | 0.483 |
| Qwen3-8B LoRA, stage 1 + stage 2 | 0.569 |
| 8B + 14B + n-gram blend, balanced | 0.593 (LB 0.592) |

So one two-stage LoRA model should take us from ~0.41 to ~0.50–0.55 on the LB, and blending it
with the SVM we already have adds a little more.

## Constraints

- **Local GPU is a GTX 1050 (4 GB)**: too small to train or even run a 4B+ model. All GPU work
  runs on Kaggle.
- **Kaggle GPU quota: 30 h/week** (resets weekly; 0 h used). 2× T4 (16 GB each), **fp16 only, no
  bf16**, sessions up to 12 h. The submission notebook must finish in ≤ 9 h with internet off.
- Base models are attached as Kaggle Models (`qwen-lm/qwen-3/Transformers/...-base`), so no
  internet is needed.
- The public recipe trained on an RTX 5090 (30 min for 8B). A T4 is roughly 8–10× slower, so an 8B
  checkpoint would be ~4–5 h per run. That's why we start smaller.

## The recipe ("LoRA24", two-stage)

- **Backbone:** Qwen3-**4B**-Base first (≈ 2 h per run on a T4, by estimate; measured in step 1), keeping only
  the first ~24 of 36 layers. Style lives in early and middle layers; late layers mostly encode
  topic, which doesn't transfer to unseen prompts. Move to 8B only if 4B works and quota allows.
- **Adapter:** LoRA r16, alpha 32, dropout 0.05 on all linear layers.
- **Head:** mean-pool the hidden states at ~8 depths (skipping the first token), one LayerNorm per
  depth, concatenate, then a linear layer to 18 classes. Final RMSNorm off.
- **max_len 384** tokens (passages are ~100 words).
- **Precision on T4:** fp16 autocast with fp32 LoRA/head weights and a GradScaler, plus gradient
  checkpointing. Guard against fp16 overflow (the public notebook redoes a batch in bf16; on T4 we
  skip/scale instead).
- **Stage 1:** 1 epoch on our match-weighted Open Pool sample (2,000/label = 36k passages, the same
  draw as the SVM). Class-weighted CE, AdamW lr 1e-4 (LoRA) / 1e-3 (head), 10% warm-up, cosine.
- **Stage 2:** 2 epochs on train.csv, lr 5e-5 / 5e-4. (The public notes: more epochs overfit, and
  mixing pool rows back in hurts.)
- Skip the "centred twin" head for now. It adds complexity, and its gain depends on how many test
  passages share a prompt.

## Steps

1. **Training notebook** (`name_that_llm/kaggle_lora_train.py` generates it, like the submission
   notebook). On Kaggle GPU T4×2, it:
   - holds out 1 prompt-grouped fold (not 5, to save quota) for a CV estimate and blend weights;
   - runs stage 1 + stage 2 on the other 4/5, saves holdout probabilities;
   - retrains stage 2 on all of train.csv, saves `adapter/`, `heads.safetensors`, `meta.json`.
   - A short timing run (~200 steps) comes first, to measure T4 speed before committing hours.
2. **Check:** holdout balanced accuracy vs the SVM on the same fold. Go on only if the LoRA is
   clearly ahead (≥ +0.05).
3. **Submission notebook v2:** attach the training notebook's output and the Qwen3-4B base model.
   Add the LoRA member, blend its log-probabilities with the SVM's (weight fit on the holdout fold),
   then Sinkhorn. Inference on ~4,000 passages should take well under an hour on T4.
4. **Submit** (with your OK), compare with 0.414.
5. **If quota is left:** an 8B run, or a second 4B with a different pool draw. Diversity adds more
   than a stronger clone.

## Budget

| Item | GPU hours (estimate) |
|---|---|
| Timing run | 0.3 |
| Train with holdout + final stage 2 | 3–4 |
| Submission run (Kaggle reruns it on scoring) | ~1 per submission |
| **Total for one 4B member** | **~5** of 30 per week |

## Risks

- fp16 instability on T4: the backbone is pretrained in bf16, so activations can overflow in fp16.
  Mitigation: overflow check, loss scaling, keep pooled features in fp32.
- T4 speed could be worse than the estimate: the timing run decides 4B vs a smaller model
  (Qwen3-1.7B).
- One holdout fold is noisy (~540 passages, ±0.02). Fine for go/no-go, not for fine tuning.
