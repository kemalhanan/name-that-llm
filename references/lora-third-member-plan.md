# Plan: a third LoRA member on a different base model (Llama-3.1-8B)

Written 2026-11-11, after SVM + 4B full + 8B full scored **0.607** on the public LB (out-of-fold
0.6305 on all 2,699 passages).

## Why another family

- **Diversity is what has moved the blend.** Adding the 5-fold 8B to the 4B gave +0.018 on the
  public LB. Older Qwen members trained on other pool draws added only noise on fold 0 (±0.006),
  so a third Qwen would likely add little.
- **Another family differs where it matters.** Its pretraining data, tokenizer and the features it
  learned are all different. Its mistakes should overlap less with the two Qwen members than a
  third Qwen's would.

## Candidates on Kaggle Models

Checked 2026-11-11. The submission runs offline, so the base model must be a Kaggle input and
must load with the transformers in the Kaggle image.

| Base | Kaggle source | For | Against |
|---|---|---|---|
| **Llama-3.1-8B (base)** | `metaresearch/llama-3.1/Transformers/8b` v2 | Plain decoder (`LlamaModel`), same layer names as Qwen, so `lora.py` works as is; 128k tokenizer; long-standing transformers support; usually stable in fp16 | Llama 3.1 license: you accept it once on the model page; trunk ~1 GB larger than the Qwen3-8B one |
| Gemma 3 4B / 12B (pt) | `google/gemma-3/Transformers/gemma-3-*-pt` | Different family, Google data | Gemma models are known to overflow in fp16, and T4s have no bf16; Gemma license |
| Gemma 4 12B / E4B | `google/gemma-4/Transformers/*` | Newest; Apache 2.0 | Multimodal "unified" models from June 2026: they may need a newer transformers than the offline image has; same fp16 risk as Gemma 3 |
| Ministral 3 8B base | `mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512` | Apache 2.0 | Multimodal (`Mistral3ForConditionalGeneration`); its card asks for `transformers==5.0.0rc0` |
| Mistral 7B | `mistral-ai/mistral` | Plain decoder, Apache 2.0 | 2023 model, weaker than Llama 3.1; fallback if Llama fails |

**Update 2026-11-11: switched to Mistral-7B-v0.1.** Kaggle has temporarily disabled access
requests for Llama 3.1, so the license can't be accepted for now. The fallback is
`mistral-ai/mistral/PyTorch/7b-v0.1-hf` (variant `mistral7b-full`): Apache 2.0, no license
request, a plain decoder with the same layer sizes as Llama-3.1-8B (32 layers, hidden 4,096, MLP
14,336) and a BOS token (`<s>`). Its 32k vocabulary gives roughly 15% more tokens per passage, so
expect each step to be a little slower than the Llama estimate below. Its trunk is smaller,
~10.7 GB in fp16 (embeddings 0.13B against Llama's 0.53B), so memory is less of a concern. The
`llama8b-full` variant stays in the code for when access reopens. The rest of this plan applies
to Mistral as written.

**Pick: Llama-3.1-8B base.** It has the fewest unknowns on T4s and offline inference. A Gemma
member would make more sense once the TPU path works (bf16), but its inference would still run
in fp16 on the T4s in the submission, where Gemma tends to overflow.

## Recipe

The same as the Qwen members, with these changes:

- **Backbone:** first 24 of Llama's 32 layers, so 75% of the depth (the Qwen members keep 24 of
  36). Hidden states pooled at the same depths [2, 4, 6, 8, 12, 16, 20, 24].
- **BOS token:** Llama expects `<|begin_of_text|>` first, and the classifier already leaves the
  first token out of the mean, since it acts as an attention sink. `tokenize` will prepend BOS
  when the tokenizer defines one. Qwen's tokenizer defines none, so the existing members and
  their submission code tokenize exactly as before (checked: `bos_token` is `None` for Qwen3).
- **Pool:** draw seed 6, cap **8,000 per model (~128k passages)**. The 24 Llama layers cost about
  1.13x the Qwen3-8B ones (MLP width 14,336 vs 12,288), so ~1.55 s/step on both T4s against the
  8B's 1.37. That is about 16k stage-1 steps, ~6.9 h, plus ~1.5 h for the 6 stage-2 runs: ~9 h,
  clear of the 12 h limit. The timing run will refine the cap.
- **Memory:** the 24-layer trunk is ~11.5 GB in fp16 against the 8B's ~10.5 GB, on 15 GB T4s.
  If the timing run's peak is too close to the limit, keep 22 layers instead.

## Steps

1. **Code** (here):
   - `lora.tokenize`: prepend BOS when the tokenizer has one. Add a test that the Qwen token ids
     don't change.
   - New variant `llama8b-full` in `kaggle_lora_train.py`: spec
     `{model_type: llama, hidden_size: 4096, num_hidden_layers: 32}`, pool seed 6, cap 8,000,
     out `lora_out_llama8b_full`.
   - Smoke test: a tiny random Llama with a BOS token, through the two-process CPU path and
     through the submission notebook's member loading.
2. **License:** you accept the Llama 3.1 license on its Kaggle model page, and check that the
   competition rules allow it as an external model.
3. **Timing run** on GPU T4 x2, ~15 min. About 4.9 h of GPU quota is left until it resets on
   2026-11-14. The run gives s/step, peak memory, the number of skipped fp16 batches, and a
   forecast for the full run.
4. **Full run** after the quota reset, ~9 h. Or on the TPU, if the TPU timing run works and the
   queue allows.
5. **Blend check** with `lora_blend.py` on all 2,699 out-of-fold passages: SVM + 4B + 8B + Llama
   against the current 0.6305. Adopt it only if the gain is clearly above noise, at least ~+0.004.
6. **Submission:** a third member with 6 adapters adds roughly a third to the LoRA scoring time.
   Check it stays within the competition's run time limit before submitting.

## Risks

- **fp16 overflow.** Llama 3 is usually fine in fp16, but the timing run will show skipped
  batches. More than a few percent means changing the recipe, for example fewer layers.
- **Run time at submission.** Three 8B-class members × 6 adapters on the hidden test set. If it
  runs too long, use fewer fold adapters for the new member.
- **No gain.** If Llama's out-of-fold score lands well below the Qwen 8B's 0.615, it may add
  nothing. Step 5 decides, before any submission is spent.
