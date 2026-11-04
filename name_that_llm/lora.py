"""Two-stage LoRA classifier on the first layers of a decoder LLM ("LoRA24" recipe).

Self-contained (torch + transformers only, no peft) so that kaggle_lora_train.py can copy it
verbatim into a Kaggle notebook. See references/lora-plan.md for the recipe and why.

* The backbone keeps only its first ``keep`` decoder layers; the final norm is switched off.
* Every linear layer inside them gets a LoRA adapter (fp32 parameters on frozen fp16 weights).
* A passage's features are its mean hidden state (first token left out) at several depths, each
  through its own LayerNorm, concatenated, then one linear layer to the classes.
* Training runs in fp16 autocast with loss scaling on T4s (no bf16 there); batches whose loss is
  not finite are skipped and counted.
"""

import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update ``B @ A * alpha / r``."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        delta = self.drop(x).to(self.lora_A.dtype) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + (delta * self.scale).to(x.dtype)


def add_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> int:
    """Wrap every target linear layer of ``model`` in LoRALinear; returns how many."""
    targets = [
        (parent, name)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Linear) and name in LORA_TARGETS
    ]
    for parent, name in targets:
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha, dropout))
    return len(targets)


def load_trunk(base_dir: str, keep: int, dtype, device) -> nn.Module:
    """The first ``keep`` decoder layers of the base model, frozen, final norm switched off."""
    cfg = AutoConfig.from_pretrained(base_dir)
    cfg.num_hidden_layers, cfg.use_cache = keep, False
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[:keep]
    trunk = AutoModel.from_pretrained(base_dir, config=cfg, dtype=dtype).to(device)
    trunk.norm = nn.Identity()
    trunk.requires_grad_(False)
    return trunk


class LoRAClassifier(nn.Module):
    def __init__(
        self, trunk: nn.Module, layers, n_classes: int, r=16, alpha=32, dropout=0.05, ckpt=True
    ):
        super().__init__()
        self.trunk, self.layers = trunk, list(layers)
        self.n_lora = add_lora(trunk, r, alpha, dropout)
        if ckpt:  # recompute activations in backward: far less memory for ~30% more time
            trunk.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        H = trunk.config.hidden_size
        dev = trunk.embed_tokens.weight.device
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in self.layers]).to(dev)
        self.head = nn.Linear(H * len(self.layers), n_classes).to(dev)

    def forward(self, input_ids, attention_mask):
        hs = self.trunk(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states
        m = attention_mask.unsqueeze(-1).float()
        m[:, 0] = 0  # the first token is an attention sink; leave it out of the mean
        den = m.sum(1).clamp(min=1)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            z = torch.cat(
                [nm((hs[l].float() * m).sum(1) / den) for l, nm in zip(self.layers, self.norms)],
                -1,
            )
            return self.head(z)

    def trainable_state(self) -> dict:
        """Only what training changes: LoRA matrices, LayerNorms and the head."""
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}


def tokenize(tok, texts, max_len: int) -> list[list[int]]:
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max_len - 1)
    return [x + [tok.eos_token_id] for x in ids["input_ids"]]


def batches(ids, bs: int, shuffle: bool, rng=None):
    """Index batches of similar length (less padding); batch order shuffled when training."""
    if not shuffle:
        order = np.argsort([len(x) for x in ids], kind="stable")
    else:  # sort within random chunks of 50 batches so batches still differ between epochs
        order = rng.permutation(len(ids))
        chunk = bs * 50
        order = np.concatenate(
            [
                sorted(order[i : i + chunk], key=lambda j: len(ids[j]))
                for i in range(0, len(order), chunk)
            ]
        )
    out = [order[i : i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def collate(tok, ids, idx, device):
    enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def train_stage(
    model,
    tok,
    texts,
    y,
    *,
    epochs: float,
    lr_lora: float,
    lr_head: float,
    bs: int,
    max_len: int,
    n_classes: int,
    amp_dtype,
    seed: int = 0,
    max_steps: int | None = None,
    log_every: int = 50,
    log=print,
):
    """One training stage; class-weighted CE, AdamW, 10% warm-up then cosine decay."""
    device = model.head.weight.device
    rng = np.random.default_rng(seed)
    ids = tokenize(tok, texts, max_len)
    y = torch.tensor(np.array(y), device=device)
    counts = torch.bincount(y, minlength=n_classes).float()
    class_w = (counts.sum() / (n_classes * counts.clamp(min=1))).to(device)

    per_epoch = math.ceil(len(ids) / bs)
    total = math.ceil(epochs * per_epoch)
    if max_steps is not None:
        total = min(total, max_steps)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW(
        [{"params": lora, "lr": lr_lora}, {"params": rest, "lr": lr_head}], weight_decay=0.01
    )
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        ),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    model.train()
    step, skipped, run_loss, run_n, t0 = 0, 0, 0.0, 0, time.time()
    while step < total:
        for idx in batches(ids, bs, shuffle=True, rng=rng):
            if step >= total:
                break
            input_ids, mask = collate(tok, ids, idx, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids, mask)
            loss = nn.functional.cross_entropy(logits.float(), y[idx], weight=class_w)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1  # fp16 overflow in the backbone; drop this batch
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(lora + rest, 1.0)
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item()
                run_n += 1
            sched.step()
            step += 1
            if step % log_every == 0 or step == total:
                el = time.time() - t0
                log(
                    f"step {step}/{total} loss {run_loss / max(1, run_n):.4f} "
                    f"{el / step:.2f}s/step, ~{el / step * (total - step) / 60:.0f} min left, "
                    f"skipped {skipped}"
                )
                run_loss, run_n = 0.0, 0
    model.eval()
    return {"steps": total, "skipped": skipped, "seconds": time.time() - t0}


@torch.no_grad()
def predict_logits(model, tok, texts, *, bs: int, max_len: int, amp_dtype) -> np.ndarray:
    device = model.head.weight.device
    ids = tokenize(tok, texts, max_len)
    out = np.zeros((len(ids), model.head.out_features), np.float32)
    model.eval()
    for idx in batches(ids, bs, shuffle=False):
        input_ids, mask = collate(tok, ids, idx, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
        ):
            out[idx] = model(input_ids, mask).float().cpu().numpy()
    return out


def load_tokenizer(base_dir: str):
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def save_checkpoint(model, out_dir: str, meta: dict):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in model.trainable_state().items()},
        os.path.join(out_dir, "lora_classifier.safetensors"),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(model, ckpt_dir: str):
    from safetensors.torch import load_file

    state = load_file(os.path.join(ckpt_dir, "lora_classifier.safetensors"))
    missing = model.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return model
