"""Two-stage LoRA training on several GPUs, with stage 2 on every prompt-grouped fold.

Run with ``torchrun --standalone --nproc_per_node=N lora_ddp.py cfg.json`` (references/
lora-2gpu-5fold-plan.md). kaggle_lora_train.py copies this file and ``lora.py`` verbatim into the
notebook's working folder, so it imports ``lora`` as a sibling module there.

* Stage 1: 1 epoch on the Open Pool sample, data-parallel (DDP): every GPU builds the same batch
  list from the same seed and takes its share of each batch, so the global batch and the recipe
  are the same as on one GPU.
* Stage 2: from the stage-1 weights, once per fold leaving that fold out (its held-out logits make
  ``oof_logits.parquet``), then once on all of train.csv.
* A batch whose fp16 loss is not finite on any GPU is skipped on all of them, so no GPU waits
  forever for another one's gradients.
* Without a GPU it runs on CPU with the gloo backend (the local smoke test).
"""

from datetime import timedelta
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # flat files on Kaggle
    from lora import (
        LoRAClassifier,
        batches,
        collate,
        load_tokenizer,
        load_trunk,
        predict_logits,
        save_checkpoint,
        tokenize,
    )
except ImportError:  # inside this package
    from name_that_llm.lora import (
        LoRAClassifier,
        batches,
        collate,
        load_tokenizer,
        load_trunk,
        predict_logits,
        save_checkpoint,
        tokenize,
    )


def sample_pool(pool: pd.DataFrame, n_per_label: int, seed: int) -> pd.DataFrame:
    """Same draw as name_that_llm/dataset.py: weighted sampling without replacement by
    match_weight (Efraimidis-Spirakis), up to ``n_per_label`` passages per label."""
    parts = []
    for _, d in pool.groupby("model"):
        rng = np.random.default_rng([seed, len(parts)])
        keys = rng.random(len(d)) ** (1.0 / d.match_weight.clip(lower=1e-6).to_numpy())
        parts.append(d.iloc[np.argsort(-keys)[: min(n_per_label, len(d))]])
    return pd.concat(parts, ignore_index=True)


def log0(*args):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, flush=True)


def train_stage_ddp(
    ddp,
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
):
    """``lora.train_stage`` on several processes: same schedule, global batch ``bs``."""
    model = ddp.module
    rank, world = dist.get_rank(), dist.get_world_size()
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

    ddp.train()
    step, skipped, run_loss, run_n, t0 = 0, 0, 0.0, 0, time.time()
    while step < total:
        for idx in batches(ids, bs, shuffle=True, rng=rng):  # same list on every rank
            if step >= total:
                break
            mine = idx[rank::world]
            if len(mine) == 0:  # a last batch smaller than the number of GPUs: all skip it
                continue
            input_ids, mask = collate(tok, ids, mine, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = ddp(input_ids, mask)
            loss = nn.functional.cross_entropy(logits.float(), y[mine], weight=class_w)
            opt.zero_grad(set_to_none=True)
            ok = torch.tensor(float(torch.isfinite(loss)), device=device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() < 1:
                skipped += 1  # fp16 overflow on some GPU; every GPU drops this batch
            else:
                scaler.scale(loss).backward()  # DDP averages the gradients here
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
                log0(
                    f"step {step}/{total} loss {run_loss / max(1, run_n):.4f} "
                    f"{el / step:.2f}s/step, ~{el / step * (total - step) / 60:.0f} min left, "
                    f"skipped {skipped}"
                )
                run_loss, run_n = 0.0, 0
    ddp.eval()
    return {"steps": total, "skipped": skipped, "seconds": time.time() - t0}


def main(cfg_path: str):
    with open(cfg_path) as f:
        C = json.load(f)
    use_cuda = torch.cuda.is_available()
    # a GPU waits at most 30 min for another one (rank 0 alone scores each held-out fold)
    dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        device = torch.device("cuda", torch.cuda.current_device())
        amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        device, amp = torch.device("cpu"), torch.float32
    T0 = time.time()
    log0(f"{world} processes on {device.type}, amp {amp}, mode {C['mode']}")

    train = pd.read_csv(f"{C['comp']}/train.csv")
    pool = pd.read_parquet(C["pool"], columns=["text", "model", "match_weight"])
    pool_sample = sample_pool(pool, C["n_per_label"], C["pool_seed"])
    del pool
    if C.get("smoke"):
        train = train.groupby("model").head(12).reset_index(drop=True)
    labels = sorted(train.model.unique())
    l2i = {m: i for i, m in enumerate(labels)}
    assert len(labels) == 18 and set(pool_sample.model) <= set(labels)
    y_train = train.model.map(l2i).to_numpy()
    y_pool = pool_sample.model.map(l2i).to_numpy()
    folds = list(GroupKFold(n_splits=C["folds"]).split(train, y_train, train.prompt_id))
    log0(f"train {len(train)} in {C['folds']} folds, pool sample {len(pool_sample)}")

    torch.manual_seed(C["seed"])  # same LoRA and head init on every rank (DDP broadcasts it too)
    tok = load_tokenizer(C["base_dir"])
    trunk = load_trunk(C["base_dir"], C["keep"], amp if use_cuda else torch.float32, device)
    model = LoRAClassifier(trunk, C["layers"], len(labels), C["r"], C["alpha"], C["dropout"])
    ddp = DDP(model, device_ids=[device.index] if use_cuda else None)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log0(
        f"{model.n_lora} LoRA layers, {n_train / 1e6:.1f}M trainable, loaded in "
        f"{time.time() - T0:.0f}s"
    )
    common = {"bs": C["bs"], "max_len": C["max_len"], "n_classes": len(labels), "amp_dtype": amp}
    out = C["out"]

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    timing = C["mode"] == "timing"
    s1 = train_stage_ddp(
        ddp,
        tok,
        pool_sample.text,
        y_pool,
        **C["stage1"],
        **common,
        seed=C["seed"],
        max_steps=C["timing_steps"] if timing else None,
        log_every=20 if timing else 200,
    )
    log0("stage 1:", s1)
    if use_cuda:
        log0(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if timing:
        per_step = s1["seconds"] / s1["steps"]
        bs = C["bs"]
        steps = {"stage 1": -(-len(pool_sample) // bs)}
        steps["stage 2, 5 folds"] = sum(2 * -(-len(f) // bs) for f, _ in folds)
        steps["stage 2 on all"] = 2 * -(-len(train) // bs)
        for k, v in steps.items():
            log0(f"{k}: {v} steps, ~{v * per_step / 60:.0f} min")
        log0(
            f"forecast for mode full: ~{sum(steps.values()) * per_step / 3600:.1f} h of "
            "training plus loading and scoring"
        )
        # then a short pass through everything mode full does, so a bug there shows up now
        log0("timing mode: a few stage-2 steps per fold to check the rest of the run")

    stage1_state = {k: v.clone() for k, v in model.trainable_state().items()}
    meta = {"labels": labels, "cfg": C, "world_size": world, "stage1": s1, "folds": []}
    if rank == 0:
        save_checkpoint(model, f"{out}/stage1", {"labels": labels, "cfg": C, "stage1": s1})

    s2_steps = 10 if timing else None
    oof = np.zeros((len(train), len(labels)), np.float32)
    fold_of = np.full(len(train), -1)
    for k, (fit_idx, hold_idx) in enumerate(folds):
        if timing:
            hold_idx = hold_idx[:48]
        model.load_state_dict({n: v.to(device) for n, v in stage1_state.items()}, strict=False)
        s2 = train_stage_ddp(
            ddp,
            tok,
            train.text.iloc[fit_idx],
            y_train[fit_idx],
            **C["stage2"],
            **common,
            seed=C["seed"] + 1 + k,
            max_steps=s2_steps,
            log_every=100,
        )
        if rank == 0:
            z = predict_logits(
                model,
                tok,
                train.text.iloc[hold_idx],
                bs=2 * C["bs"],
                max_len=C["max_len"],
                amp_dtype=amp,
            )
            oof[hold_idx], fold_of[hold_idx] = z, k
            bacc = balanced_accuracy_score(y_train[hold_idx], z.argmax(1))
            meta["folds"].append(dict(fold=k, balanced_accuracy=bacc, n=len(hold_idx), **s2))
            save_checkpoint(
                model,
                f"{out}/fold{k}",
                {"labels": labels, "cfg": C, "fold": k, "holdout_balanced_accuracy": bacc},
            )
            log0(
                f"fold {k}: balanced accuracy {bacc:.4f} ({len(hold_idx)} passages) "
                f"after {(time.time() - T0) / 60:.0f} min"
            )
        dist.barrier()

    model.load_state_dict({n: v.to(device) for n, v in stage1_state.items()}, strict=False)
    s2_all = train_stage_ddp(
        ddp,
        tok,
        train.text,
        y_train,
        **C["stage2"],
        **common,
        seed=C["seed"] + 1 + C["folds"],
        max_steps=s2_steps,
        log_every=100,
    )
    if rank == 0:
        done = fold_of >= 0  # every passage in mode full
        oof_bacc = balanced_accuracy_score(y_train[done], oof[done].argmax(1))
        meta.update(
            stage2_all=s2_all,
            oof_balanced_accuracy=oof_bacc,
            holdout_balanced_accuracy=meta["folds"][0]["balanced_accuracy"],
        )
        # the full fit sits at the top of the output folder, like the single-GPU runs
        save_checkpoint(model, out, meta)
        frame = pd.DataFrame(oof, columns=labels)
        frame.insert(0, "id", train.id.to_numpy())
        frame.insert(1, "model", train.model.to_numpy())
        frame.insert(2, "fold", fold_of)
        frame.to_parquet(f"{out}/oof_logits.parquet", index=False)
        frame[frame.fold == 0].drop(columns="fold").to_parquet(
            f"{out}/holdout_logits.parquet", index=False
        )
        log0(
            f"out-of-fold balanced accuracy {oof_bacc:.4f} over {done.sum()} passages; "
            f"per fold {[round(f['balanced_accuracy'], 4) for f in meta['folds']]}"
        )
        log0(f"saved to {out} in {(time.time() - T0) / 60:.1f} min")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
