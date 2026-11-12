"""Two-stage LoRA training on several GPUs or TPU chips, with stage 2 on every prompt-grouped fold.

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

With ``"accelerator": "tpu"`` in cfg.json, run it once as ``python lora_ddp.py cfg.json``: it starts
one PyTorch/XLA process per TPU chip itself (``xmp.spawn``). The recipe is the same; what changes:

* The trunk is bf16 and runs without autocast or loss scaling (bf16 does not overflow like fp16),
  and gradient checkpointing is off (a TPU v5e chip has 16 GB, enough for the 8B trunk at batch 1).
* Static shapes: every rank pads to the same length, the longest passage of the whole global
  batch rounded up to 64 tokens, and a last batch smaller than ``bs`` is dropped. XLA then compiles
  one training program per length (at most 6 at max_len 384) instead of one per batch.
* Gradients are averaged with ``xm.reduce_gradients``, and the loss is read back to the host only
  when it is logged, since every read waits for the TPU.
* Every rank builds its model on the CPU from the same seed, a few ranks at a time so host RAM
  holds only a few copies of the trunk at once, then moves it to its chip.
"""

from datetime import timedelta
import gc
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
        static_len,
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
        static_len,
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


RANK = 0  # this process's rank, set in run()


def log0(*args):
    if RANK == 0:
        print(*args, flush=True)


class Comm:
    """What training needs from the process group: torch.distributed (NCCL on GPU, gloo on CPU)
    or PyTorch/XLA on TPU."""

    def __init__(self, xla: bool):
        self.xla = xla
        if xla:
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr

            self.xm = xm
            self.rank, self.world = xr.global_ordinal(), xr.world_size()
            self.device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
            self.sync = getattr(torch_xla, "sync", None) or xm.mark_step
            self.amp = torch.bfloat16  # the trunk's dtype; no autocast on XLA
            return
        use_cuda = torch.cuda.is_available()
        # a GPU waits at most 30 min for another one (rank 0 alone scores each held-out fold)
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=30))
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if use_cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", self.rank)))
            self.device = torch.device("cuda", torch.cuda.current_device())
            major = torch.cuda.get_device_capability()[0]
            self.amp = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.device, self.amp = torch.device("cpu"), torch.float32

    def barrier(self, tag: str):
        if self.xla:
            self.xm.rendezvous(tag)
        else:
            dist.barrier()

    def close(self):
        if not self.xla:
            dist.destroy_process_group()


def train_stage_ddp(
    comm,
    fwd,
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
):
    """``lora.train_stage`` on several processes: same schedule, global batch ``bs``.

    ``fwd`` is the DDP wrapper on GPU/CPU (it averages the gradients in backward) and the bare
    model on XLA (gradients averaged by ``xm.reduce_gradients``)."""
    rank, world, xla = comm.rank, comm.world, comm.xla
    device = model.head.weight.device
    rng = np.random.default_rng(seed)
    ids = tokenize(tok, texts, max_len)
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), device=device)

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
    # every rank needs a share of each batch; on XLA every batch also needs the same shape
    min_batch = bs if xla else world
    # the steps after the first few (and, on XLA, after most compiling) give the steady s/step
    warm_steps = min(20, total // 2)

    fwd.train()
    step, skipped, run_n, t0, t_warm = 0, 0, 0, time.time(), None
    run_loss = torch.zeros((), device=device) if xla else 0.0
    while step < total:
        for idx in batches(ids, bs, shuffle=True, rng=rng):  # same list on every rank
            if step >= total:
                break
            if len(idx) < min_batch:  # the last batch of an epoch; the same on every rank
                continue
            mine = idx[rank::world]
            length = static_len(ids, idx, max_len) if xla else None
            input_ids, mask = collate(tok, ids, mine, device, length)
            target = torch.as_tensor(y[mine]).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = fwd(input_ids, mask)
            loss = nn.functional.cross_entropy(logits.float(), target, weight=class_w)
            opt.zero_grad(set_to_none=True)
            if xla:
                loss.backward()
                comm.xm.reduce_gradients(opt)
                torch.nn.utils.clip_grad_norm_(lora + rest, 1.0)
                opt.step()
                comm.sync()  # run this step's graph on the TPU
                run_loss = run_loss + loss.detach()
                run_n += 1
            else:
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
            if step == warm_steps:
                t_warm = time.time()
            if step % log_every == 0 or step == total:
                el = time.time() - t0
                mean = float(run_loss) / max(1, run_n)  # on XLA this waits for the TPU
                extra = ""
                if xla:
                    import torch_xla.debug.metrics as met

                    compiles = met.metric_data("CompileTime")
                    extra = f", {compiles[0] if compiles else 0} XLA compilations"
                log0(
                    f"step {step}/{total} loss {mean:.4f} "
                    f"{el / step:.2f}s/step, ~{el / step * (total - step) / 60:.0f} min left, "
                    f"skipped {skipped}{extra}"
                )
                run_loss = torch.zeros((), device=device) if xla else 0.0
                run_n = 0
    fwd.eval()
    t_end = time.time()
    out = {"steps": total, "skipped": skipped, "seconds": t_end - t0}
    if t_warm is not None and total > warm_steps:
        out["steady_s_per_step"] = (t_end - t_warm) / (total - warm_steps)
    return out


def build_model(comm, C: dict, n_classes: int):
    """The classifier on this rank's device with the same initial weights on every rank, and
    what to call for the forward pass (DDP on GPU/CPU, the bare model on XLA)."""
    if not comm.xla:
        torch.manual_seed(C["seed"])  # same LoRA and head init on every rank (DDP broadcasts too)
        dtype = comm.amp if comm.device.type == "cuda" else torch.float32
        trunk = load_trunk(C["base_dir"], C["keep"], dtype, comm.device)
        model = LoRAClassifier(trunk, C["layers"], n_classes, C["r"], C["alpha"], C["dropout"])
        on_gpu = comm.device.type == "cuda"
        return model, DDP(model, device_ids=[comm.device.index] if on_gpu else None)
    # XLA: build on the CPU (CPU RNG, so the same init everywhere), a few ranks at a time
    wave = C.get("load_parallel", 4)
    model = None
    for first in range(0, comm.world, wave):
        if first <= comm.rank < first + wave:
            torch.manual_seed(C["seed"])
            trunk = load_trunk(
                C["base_dir"], C["keep"], comm.amp, "cpu", attn_implementation="eager"
            )
            model = LoRAClassifier(
                trunk,
                C["layers"],
                n_classes,
                C["r"],
                C["alpha"],
                C["dropout"],
                ckpt=C.get("ckpt", False),
            ).to(comm.device)
            comm.sync()
            gc.collect()
        comm.barrier(f"load{first}")
    return model, model


def main(cfg_path: str):
    with open(cfg_path) as f:
        C = json.load(f)
    if C.get("accelerator") == "tpu":
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(run, args=(C,))  # one process per TPU chip
    else:
        run(None, C)


def run(index, C: dict):
    global RANK
    comm = Comm(xla=C.get("accelerator") == "tpu")
    RANK = comm.rank
    rank, world, device, amp = comm.rank, comm.world, comm.device, comm.amp
    use_cuda = device.type == "cuda"
    T0 = time.time()
    log0(f"{world} processes on {device.type}, dtype {amp}, mode {C['mode']}")

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

    tok = load_tokenizer(C["base_dir"])
    model, fwd = build_model(comm, C, len(labels))
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
        comm,
        fwd,
        model,
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
    if comm.xla and rank == 0:
        try:
            log0(f"TPU memory: {comm.xm.get_memory_info(device)}")
        except Exception as e:  # noqa: BLE001  (API differs between torch_xla versions)
            log0(f"TPU memory: not available ({e})")
    if timing:
        per_step = s1.get("steady_s_per_step", s1["seconds"] / s1["steps"])
        log0(f"steady {per_step:.2f}s/step (after the first steps and compiles)")
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
            comm,
            fwd,
            model,
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
                static=comm.xla,
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
        comm.barrier(f"fold{k}")

    model.load_state_dict({n: v.to(device) for n, v in stage1_state.items()}, strict=False)
    s2_all = train_stage_ddp(
        comm,
        fwd,
        model,
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
    comm.barrier("done")
    comm.close()


if __name__ == "__main__":
    main(sys.argv[1])
