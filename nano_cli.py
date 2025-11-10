#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nano_cli.py — DDP-friendly training entrypoint (torchrun --standalone --nproc_per_node=N)

Refactor highlights:
  • Replaces any prior linear schedule with: linear warmup (default 500 steps, CLI-overridable)
    followed by cosine decay to min_lr (default 0.0).
  • Optimizer-agnostic: works with standard PyTorch optimizers AND optimizers that implement
    special train()/eval() semantics (e.g., Schedule-Free variants using iterate averaging).
    >>> IMPORTANT: For optimizers whose stepping behavior differs in train vs eval mode,
    >>> this script calls optimizer.eval() before validation & checkpointing and then
    >>> optimizer.train() before resuming training. (This is crucial for iterate-averaging
    >>> style optimizers like AdamWScheduleFree.)
  • Two-optimizer split: opt1 handles scalar+head+embed; opt2 handles hidden matrices + gate.
  • Full, reproducible RUN CONFIG header printed BEFORE any other logs so runs are self-contained.

Usage examples (8 GPUs):

torchrun --standalone --nproc_per_node=8 nano_cli.py \
  --num-devices 8 --batch-size 256 --num-iterations 5100 \
  --scheduler cosine --warmup-steps 500 --min-lr 0.0 \
  --opt1 AdamW \
  --opt1-kwargs lr=0.0036,betas=[0.9,0.95],eps=1e-8,weight_decay=0.0 \
  --opt2 Muon \
  --opt2-kwargs lr=0.02,momentum=0.95,nesterov=true,backend=newtonschulz5,backend_steps=5

torchrun --standalone --nproc_per_node=8 nano_cli.py \
  --num-devices 8 --batch-size 256 --num-iterations 5100 \
  --scheduler cosine --warmup-steps 500 --min-lr 0.0 \
  --opt1 AdamW \
  --opt1-kwargs '{"lr":0.0036,"betas":[0.9,0.95],"eps":1e-8,"weight_decay":0.0}' \
  --opt2 Muon \
  --opt2-kwargs "{\"lr\":0.02,\"momentum\":0.95,\"nesterov\":true,\"backend\":\"newtonschulz5\",\"backend_steps\":5}"


Schedule‑Free example (opt2 warmup→constant internally; opt1 uses external cosine):
  --opt2 AdamWScheduleFree --opt2-kwargs '{"lr":0.0036,"betas":[0.9,0.999],"eps":1e-8}'

This file assumes the following utilities are available (as in your original project):
  from utils import DistributedDataLoader, GPT, GPTConfig
  and any custom optimizers like NorMuon, MuonEMA, AdamWScheduleFree, etc.

"""

import os
import sys
import json
import re
import time
import math
import uuid
import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

# --- Project utilities & (optional) custom optimizers ------------------------
# Expect these to be present in the repo
from data_loader import DistributedDataLoader
from gpt2_model import GPT, GPTConfig
from optimizers import Muon

# ----------------------------------------------------------------------------
# CLI & defaults
# ----------------------------------------------------------------------------
@dataclass
class Hyperparameters:
    # data/model
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    model_type: str = "gpt2"
    vocab_size: int = 50304  # Modified to fit the GPU bette r
    # Parameters for GPT 2 Small 
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    sequence_len: int = 1024

    # devices & batching
    num_devices: int = 8
    device_batch_size: int = 64   # tokens per device per step = device_batch_size*sequence_len
    batch_size: int = 8*64        # global tokens per step: batch_size*sequence_len

    # iterations
    num_iterations: int = 5100

    # legacy (unused by new schedule) kept for compatibility toggles
    weight_decay: float = 0.0

    # evaluation & logging
    val_loss_every: int = 125
    val_tokens: int = 10_485_760
    save_every: int = 0
    first_below: float = 0.0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nano trainer")
    # data/model
    p.add_argument("--input-bin", type=str, default=Hyperparameters.input_bin)
    p.add_argument("--input-val-bin", type=str, default=Hyperparameters.input_val_bin)
    p.add_argument("--model-type", type=str, default=Hyperparameters.model_type)
    p.add_argument("--vocab-size", type=int, default=Hyperparameters.vocab_size)
    p.add_argument("--n-layer", type=int, default=Hyperparameters.n_layer)
    p.add_argument("--n-head", type=int, default=Hyperparameters.n_head)
    p.add_argument("--n-embd", type=int, default=Hyperparameters.n_embd)
    p.add_argument("--sequence-len", type=int, default=Hyperparameters.sequence_len)

    # devices & batching
    p.add_argument("--num-devices", type=int, default=Hyperparameters.num_devices)
    p.add_argument("--device-batch-size", type=int, default=Hyperparameters.device_batch_size)
    p.add_argument("--batch-size", type=int, default=Hyperparameters.batch_size)

    # iterations
    p.add_argument("--num-iterations", type=int, default=Hyperparameters.num_iterations)

    # schedule (new)
    p.add_argument("--scheduler", type=str, choices=["cosine", "none"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr", type=float, default=0.0)

    # two-optimizer split & kwargs
    p.add_argument("--opt1", type=str, default="AdamW")
    p.add_argument("--opt2", type=str, default="AdamW")
    p.add_argument("--opt1-kwargs", type=str, default="")
    p.add_argument("--opt2-kwargs", type=str, default="")

    # evaluation & logging
    p.add_argument("--val-loss-every", type=int, default=Hyperparameters.val_loss_every)
    p.add_argument("--val-tokens", type=int, default=Hyperparameters.val_tokens)
    p.add_argument("--save-every", type=int, default=Hyperparameters.save_every)
    p.add_argument("--first-below", type=float, default=Hyperparameters.first_below)

    return p


# ----------------------------------------------------------------------------
# Helpers: parsing, schedule-free detection, factories, etc.
# ----------------------------------------------------------------------------

def _safe_parse_kwargs(s: str) -> Dict[str, Any]:
    import json, ast
    if not s:
        return {}
    # Try JSON first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: key=val,key=val with lists allowed (e.g., betas=[0.65,0.95])
    out: Dict[str, Any] = {}
    # split on commas only at top level (not inside [] or {})
    parts, buf, depth = [], [], 0
    for ch in s.strip():
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())

    for kv in parts:
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Bad kwarg token: {kv!r}")
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        # Try to literal-eval lists/numbers/bools; fall back to string
        try:
            val = ast.literal_eval(v)
        except Exception:
            if v.lower() in ("true", "false"):
                val = (v.lower() == "true")
            else:
                # bare words (e.g., backend=newtonschulz5)
                val = v
        out[k] = val
    return out


def _is_schedule_free_name(name: str) -> bool:
    return "schedulefree" in name.replace("_", "").replace("-", "").lower()


def _is_schedule_free_opt(opt) -> bool:
    return _is_schedule_free_name(opt.__class__.__name__)


def _count_params(ps: Iterable[torch.Tensor]) -> int:
    return sum(int(p.numel()) for p in ps)

# Auto-fill rank/world_size for Muon if missing
def _maybe_fill_dist(optimizer_name: str, kw: Dict[str, Any]):
    name = optimizer_name.lower().replace("_","")
    if name == "muon":
        if "rank" not in kw:
            kw["rank"] = int(os.environ.get("LOCAL_RANK", "0"))
        if "world_size" not in kw:
            kw["world_size"] = int(os.environ.get("WORLD_SIZE", os.environ.get("NNODES","1")))
    return kw

def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], kw: Dict[str, Any]):
    # Prefer torch.optim first, then fall back to globals (custom optimizers coming from utils)
    if hasattr(torch.optim, name):
        cls = getattr(torch.optim, name)
    elif name in globals():
        cls = globals()[name]
    else:
        raise ValueError(f"Optimizer '{name}' not found in torch.optim or globals().")
    opt = cls(params, **kw)
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])
    return opt


def make_cosine_with_warmup_lambda(total_steps: int, warmup: int, min_lr_ratio: float):
    warm = max(0, min(warmup, max(0, total_steps - 1)))
    def lr_mult(step: int):
        if step < warm:
            return (step + 1) / float(max(1, warm))
        decay_steps = max(1, total_steps - warm)
        pct = min(1.0, max(0.0, (step - warm) / decay_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * pct))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    return lr_mult


def _optimizers_eval(optimizers: List[torch.optim.Optimizer]):
    for _opt in optimizers:
        if hasattr(_opt, "eval") and callable(_opt.eval):
            _opt.eval()


def _optimizers_train(optimizers: List[torch.optim.Optimizer]):
    for _opt in optimizers:
        if hasattr(_opt, "train") and callable(_opt.train):
            _opt.train()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

# --- Parse args & seed ---------------------------------------------------
parser = build_argparser()
args = parser.parse_args()

torch.manual_seed(1337)
np.random.seed(1337)

# --- torch.distributed init (torchrun) ----------------------------------
assert torch.cuda.is_available(), "CUDA required"
dist.init_process_group(backend="nccl")
ddp_rank = dist.get_rank()
ddp_world_size = dist.get_world_size()
ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(ddp_local_rank)
master_process = (ddp_rank == 0)

# --- Repro header (print BEFORE anything else) --------------------------
def tokens_per_step():
    return args.batch_size * args.sequence_len * getattr(args, "num_devices", 1)

opt1_kwargs = _safe_parse_kwargs(getattr(args, "opt1_kwargs", ""))
opt2_kwargs = _safe_parse_kwargs(getattr(args, "opt2_kwargs", ""))

opt1_kwargs = _maybe_fill_dist(args.opt1, opt1_kwargs)
opt2_kwargs = _maybe_fill_dist(args.opt2, opt2_kwargs)

print("=== RUN CONFIG (repro header) ===")
print(f"world_size={ddp_world_size} devices={getattr(args,'num_devices',1)}")
print(f"batch_size={args.batch_size} seq_len={args.sequence_len} tokens_per_step={tokens_per_step()}")
print(f"num_iterations={args.num_iterations} val_loss_every={args.val_loss_every} val_tokens={args.val_tokens}")
print(f"save_every={args.save_every} first_below={args.first_below}")
print(f"scheduler={args.scheduler} warmup_steps={args.warmup_steps} min_lr={args.min_lr}")
print(f"opt1={args.opt1} opt1_kwargs={opt1_kwargs}")
print(f"opt2={args.opt2} opt2_kwargs={opt2_kwargs}")
print("=================================")

# --- Build model ---------------------------------------------------------
gptconf = GPTConfig(
    vocab_size=args.vocab_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
)

print("[init] building/compiling model...", flush=True)
_m = GPT(gptconf).cuda()
_m = torch.compile(_m)            # like your original
model = DDP(_m, device_ids=[ddp_local_rank])
raw_model = model.module          # unwrapped
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
print("[init] model ready", flush=True)

# --- Data ---------------------------------------------------------------
B, T = args.device_batch_size, args.sequence_len
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

print("[init] building dataloaders...", flush=True)
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader   = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: tokens={train_loader.ntok_total} files={len(train_loader.files)}", flush=True)
    print(f"Validation DataLoader: tokens={val_loader.ntok_total} files={len(val_loader.files)}", flush=True)

# seed batch (match old code: do NOT .to(device) here; buffers stay pinned by loader)
x, y = train_loader.next_batch()

# --- Parameter split (no gates; honor weight tying) ---------------------
m = raw_model
tied = (m.lm_head.weight is m.transformer.wte.weight)

hidden_matrix_params = [
    p for n, p in m.named_parameters()
    if p.ndim >= 2 and n != "lm_head.weight" and (tied or n != "transformer.wte.weight")
]
embed_params  = [] if tied else [m.transformer.wte.weight]
scalar_params = [p for p in m.parameters() if p.ndim < 2]
head_params   = [m.lm_head.weight]

params_opt1 = scalar_params + head_params + embed_params
params_opt2 = hidden_matrix_params

if master_process:
    def _count_params(ps): return sum(int(p.numel()) for p in ps)
    print("=== PARAM GROUP SIZES ===", flush=True)
    print(f"opt1_params (scalar+head+embed): {len(params_opt1)} tensors, {_count_params(params_opt1):,} params", flush=True)
    print(f"opt2_params (hidden_mats):       {len(params_opt2)} tensors, {_count_params(params_opt2):,} params", flush=True)
    print("=========================", flush=True)

# --- Optimizers ----------------------------------------------------------
opt1_is_sf = _is_schedule_free_name(args.opt1)
opt2_is_sf = _is_schedule_free_name(args.opt2)

if opt1_is_sf: opt1_kwargs.setdefault("warmup_steps", args.warmup_steps)
if opt2_is_sf: opt2_kwargs.setdefault("warmup_steps", args.warmup_steps)

optimizer1 = build_optimizer(args.opt1, params_opt1, opt1_kwargs)
optimizer2 = build_optimizer(args.opt2, params_opt2, opt2_kwargs)
optimizers = [optimizer1, optimizer2]

# iterate-averaging style contract: keep opts in train() during stepping
_optimizers_train(optimizers)

if (opt1_is_sf or opt2_is_sf) and master_process:
    print("[info] Schedule-Free optimizer detected:", flush=True)
    if opt1_is_sf: print(f"  - opt1={args.opt1}: internal warmup={args.warmup_steps}", flush=True)
    if opt2_is_sf: print(f"  - opt2={args.opt2}: internal warmup={args.warmup_steps}", flush=True)
    if args.scheduler != "none":
        print("[info] External cosine applies ONLY to non-SF opts.", flush=True)

# --- LR schedulers (cosine warmup->decay) --------------------------------
from torch.optim.lr_scheduler import LambdaLR

def _ratio_for_opt(opt: torch.optim.Optimizer) -> float:
    if args.min_lr <= 0: return 0.0
    base_lrs = [g.get("initial_lr", g["lr"]) for g in opt.param_groups]
    blr_max = max(base_lrs) if base_lrs else 1.0
    return min(1.0, float(args.min_lr) / max(1e-12, blr_max))

def make_cosine_with_warmup_lambda(total_steps: int, warmup: int, min_lr_ratio: float):
    warm = max(0, min(warmup, max(0, total_steps - 1)))
    def lr_mult(step: int):
        if step < warm:  # linear warmup
            return (step + 1) / float(max(1, warm))
        decay_steps = max(1, total_steps - warm)
        pct = min(1.0, max(0.0, (step - warm) / decay_steps))
        import math
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * pct))
    return lr_mult

schedulers: List[LambdaLR] = []
if args.scheduler == "cosine":
    if not opt1_is_sf:
        schedulers.append(LambdaLR(optimizer1, lr_lambda=make_cosine_with_warmup_lambda(
            args.num_iterations, args.warmup_steps, _ratio_for_opt(optimizer1))))
    if not opt2_is_sf:
        schedulers.append(LambdaLR(optimizer2, lr_lambda=make_cosine_with_warmup_lambda(
            args.num_iterations, args.warmup_steps, _ratio_for_opt(optimizer2))))
else:
    if not opt1_is_sf: schedulers.append(LambdaLR(optimizer1, lr_lambda=lambda _: 1.0))
    if not opt2_is_sf: schedulers.append(LambdaLR(optimizer2, lr_lambda=lambda _: 1.0))

if master_process:
    print("=== LR MULTIPLIER PREVIEW ===", flush=True)
    preview = [0, max(0, args.warmup_steps//2), max(0, args.warmup_steps-1),
               args.warmup_steps, args.num_iterations//2, max(0, args.num_iterations-1)]
    def _mult_for(opt_is_sf: bool, opt: Optional[torch.optim.Optimizer], s: int) -> float:
        if args.scheduler != "cosine" or opt_is_sf or opt is None: return 1.0
        r = _ratio_for_opt(opt); return make_cosine_with_warmup_lambda(args.num_iterations, args.warmup_steps, r)(s)
    for s in preview:
        s = max(0, min(s, args.num_iterations-1))
        print(f"step={s} opt1_mult={_mult_for(opt1_is_sf, optimizer1, s):.6f} "
              f"opt2_mult={_mult_for(opt2_is_sf, optimizer2, s):.6f}", flush=True)
    print("================================", flush=True)

# --- Logging setup -------------------------------------------------------
out_dir = os.path.join("logs", time.strftime("%Y%m%d-%H%M%S"))
if master_process: os.makedirs(out_dir, exist_ok=True)
logfile = os.path.join(out_dir, "train.log")

def _lr_string(opt: torch.optim.Optimizer) -> str:
    lrs = []
    for g in opt.param_groups:
        lr = g.get("scheduled_lr", g["lr"])
        lrs.append(f"{lr:.6g}")
    return "[" + ",".join(lrs) + "]"

# --- Training loop (old style; no scaler) --------------------------------
print("[run] starting loop...", flush=True)
t0 = time.time()
training_time_ms = 0.0
timed_steps = 0
best_ckpt_written = False

for step in range(args.num_iterations):
    last_step = (step == args.num_iterations - 1)

    # === Validation =========================================================
    do_val = (args.val_loss_every > 0 and (step % args.val_loss_every == 0)) or last_step
    if do_val:
        torch.cuda.synchronize()
        training_time_ms += 1000.0 * (time.time() - t0)

        _optimizers_eval(optimizers)  # only change vs original

        model.eval()
        val_loader.reset()
        val_loss = torch.zeros((), device=x.device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()  # keep old semantics (no .to(device))
            with ctx:                               # no torch.no_grad() to avoid compile quirk
                _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.detach()
            del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps

        if master_process:
            print(f"VAL step:{step}/{args.num_iterations} val_loss:{val_loss:.4f}", flush=True)
            with open(logfile, "a") as f:
                f.write(f"VAL step:{step}/{args.num_iterations} val_loss:{val_loss:.4f}\n")

        # back to train
        model.train()
        _optimizers_train(optimizers)

        t0 = time.time()  # restart timer

    # === Train ==============================================================
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()

        # next batch (match old code: do not .to(device); loader hands CUDA tensors)
        x, y = train_loader.next_batch()

        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    # average grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps

    # step opts, then scheds; zero grads like before
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    for g in optimizer1.param_groups: g["scheduled_lr"] = g["lr"]
    for g in optimizer2.param_groups: g["scheduled_lr"] = g["lr"]
    for sch in schedulers: sch.step()

    timed_steps += 1

    if master_process:
        approx = training_time_ms + 1000.0 * (time.time() - t0)
        print(
            f"step:{step+1}/{args.num_iterations} "
            f"train_loss:{float(train_loss):.4f} "
            f"lr1:{_lr_string(optimizer1)} lr2:{_lr_string(optimizer2)} "
            f"train_time:{approx:.0f}ms step_avg:{approx/max(1,timed_steps):.2f}ms",
            flush=True
        )
        with open(logfile, "a") as f:
            f.write(
                f"step:{step+1}/{args.num_iterations} train_loss:{float(train_loss):.4f} "
                f"lr1:{_lr_string(optimizer1)} lr2:{_lr_string(optimizer2)} "
                f"train_time:{approx:.0f}ms step_avg:{approx/max(1,timed_steps):.2f}ms\n"
            )

    if last_step:
        break

if master_process:
    peak = torch.cuda.max_memory_allocated() // 1024 // 1024
    print(f"peak memory consumption: {peak} MiB", flush=True)