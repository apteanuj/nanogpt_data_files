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
  --num-devices 8 --batch-size 512 --num-iterations 5100 \
  --scheduler cosine --warmup-steps 500 --min-lr 0.0 \
  --opt1 AdamW \
  --opt1-kwargs lr=0.0036,betas=[0.9,0.95],eps=1e-8,weight_decay=0.0 \
  --opt2 Muon \
  --opt2-kwargs lr=0.02,momentum=0.95,nesterov=true,backend=newtonschulz5,backend_steps=5

torchrun --standalone --nproc_per_node=8 nano_cli.py \
  --num-devices 8 --batch-size 512 --num-iterations 5100 \
  --scheduler cosine --warmup-steps 250 --min-lr 0.0 \
  --opt1 AdamW \
  --opt1-kwargs lr=0.0036,betas=[0.9,0.95],eps=1e-8,weight_decay=0.0 \
  --opt2 MuonIterAvg \
  --opt2-kwargs lr=0.02,momentum=0.95,nesterov=true,backend=newtonschulz5,backend_steps=5,ia_beta=0.99

Schedule‑Free example (opt1 and opt2 warmup→constant internally):

torchrun --standalone --nproc_per_node=8 nano_cli.py \
  --num-devices 8 --batch-size 512 --num-iterations 5100 \
  --scheduler cosine --warmup-steps 250 --min-lr 0.0 \
  --opt1 AdamWScheduleFree \
  --opt1-kwargs lr=0.0036,betas=[0.9,0.99],eps=1e-8 \
  --opt2 AdamWScheduleFree \
  --opt2-kwargs lr=0.0036,betas=[0.9,0.99],eps=1e-8

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
import subprocess, textwrap
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
from optimizers import Muon, MuonIterAvg
from adamw_schedulefree import AdamWScheduleFree

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

if master_process:
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

# print("[init] building/compiling model...", flush=True)
_m = GPT(gptconf).cuda()
_m = torch.compile(_m)            # like your original
model = DDP(_m, device_ids=[ddp_local_rank])
raw_model = model.module          # unwrapped
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# print("[init] model ready", flush=True)

# --- Data ---------------------------------------------------------------
B, T = args.device_batch_size, args.sequence_len
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# print("[init] building dataloaders...", flush=True)
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

# Force schedule-free optimizers to have *no internal warmup*
if opt1_is_sf:
    opt1_kwargs.pop("warmup_steps", None)
if opt2_is_sf:
    opt2_kwargs.pop("warmup_steps", None)

optimizer1 = build_optimizer(args.opt1, params_opt1, opt1_kwargs)
optimizer2 = build_optimizer(args.opt2, params_opt2, opt2_kwargs)
optimizers = [optimizer1, optimizer2]

# iterate-averaging style contract: keep opts in train() during stepping
_optimizers_train(optimizers)

if (opt1_is_sf or opt2_is_sf) and master_process:
    print("[info] Schedule-Free optimizer detected:", flush=True)
    if opt1_is_sf: print(f"  - opt1={args.opt1}: linear warmup={args.warmup_steps}", flush=True)
    if opt2_is_sf: print(f"  - opt2={args.opt2}: linear warmup={args.warmup_steps}", flush=True)
    if args.scheduler != "none":
        print("[info] External cosine applies ONLY to non-SF opts.", flush=True)

def _ratio_for_opt(opt: torch.optim.Optimizer) -> float:
    # r = min_lr / base_lr over param groups; clamp to [0,1]
    if args.min_lr is None:
        return 0.0
    base_lrs = [pg.get("initial_lr", pg["lr"]) for pg in opt.param_groups]
    blr_max = max(base_lrs) if base_lrs else 1.0
    return float(max(0.0, min(1.0, args.min_lr / max(1e-12, blr_max))))

def make_linear_warmup_lambda(total_steps: int, warmup_steps: int):
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps == 0:
        return lambda s: 1.0
    denom = float(warmup_steps)
    return lambda s: min(1.0, max(0.0, (min(s, warmup_steps - 1) + 1) / denom))

def make_cosine_lambda(total_steps: int, r: float):
    # cosine from 1.0 down to r over [0, total_steps)
    T = max(1, int(total_steps))
    return lambda s: r + 0.5 * (1 - r) * (1 + math.cos(math.pi * min(s, T - 1) / (T - 1)))

schedulers: List[LambdaLR] = []

warm = make_linear_warmup_lambda(args.num_iterations, args.warmup_steps)

def _final_lambda_for(opt, is_sf: bool):
    if is_sf or args.scheduler != "cosine":
        # Schedule-free (or no cosine requested): warmup only, then constant
        return warm
    # Non-SF + cosine: warmup × cosine
    r = _ratio_for_opt(opt)
    cos = make_cosine_lambda(args.num_iterations, r)
    return lambda s, w=warm, c=cos: w(s) * c(s)

if optimizer1 is not None:
    lam1 = _final_lambda_for(optimizer1, opt1_is_sf)
    schedulers.append(LambdaLR(optimizer1, lr_lambda=lam1, last_epoch=-1))

if optimizer2 is not None:
    lam2 = _final_lambda_for(optimizer2, opt2_is_sf)
    schedulers.append(LambdaLR(optimizer2, lr_lambda=lam2, last_epoch=-1))

#####################################################################################
# --- Logging setup (classic header, consolidated to "begin logging") ---------
def _run_config_text():
    lines = []
    lines.append("=== RUN CONFIG (repro header) ===")
    lines.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"world_size={ddp_world_size} local_rank={ddp_local_rank} device={device}")
    lines.append(f"num_devices={getattr(args,'num_devices', ddp_world_size)}")
    lines.append(f"batch_size={args.batch_size} device_batch_size={args.device_batch_size} seq_len={args.sequence_len}")
    lines.append(f"num_iterations={args.num_iterations} val_loss_every={args.val_loss_every} val_tokens={args.val_tokens}")
    lines.append(f"scheduler={args.scheduler} warmup_steps={args.warmup_steps} min_lr={args.min_lr}")
    lines.append(f"opt1={args.opt1} opt1_kwargs={opt1_kwargs}")
    lines.append(f"opt2={args.opt2} opt2_kwargs={opt2_kwargs}")
    lines.append(f"param_group_opt1_count={len(params_opt1)} param_group_opt1_params={_count_params(params_opt1):,}")
    lines.append(f"param_group_opt2_count={len(params_opt2)} param_group_opt2_params={_count_params(params_opt2):,}")
    lines.append("=================================")
    return "\n".join(lines)

def _env_block_text():
    try:
        smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        smi_out = smi.stdout
    except Exception as e:
        smi_out = f"(nvidia-smi unavailable: {e})"
    torch_ver = getattr(torch, "__version__", "unknown")
    cuda_ver  = getattr(torch.version, "cuda", "unknown")
    return textwrap.dedent(f"""\
        === ENVIRONMENT =====================================
        torch={torch_ver}  cuda={cuda_ver}
        nvidia-smi:
        {smi_out}
        ======================================================
    """)

def _source_text():
    try:
        with open(sys.argv[0], "r") as f:
            return f.read()
    except Exception as e:
        return f"(could not read source file: {e})"

def _format_lr(opt: Optional[torch.optim.Optimizer], label: str) -> str:
    if opt is None:
        return f"{label}=None"
    lrs = [pg.get("lr") for pg in opt.param_groups if "lr" in pg]
    if not lrs:
        return f"{label}=[]"
    uniq = sorted({float(x) for x in lrs})
    if len(uniq) == 1:
        return f"{label}={uniq[0]:.8f}"
    return f"{label}=[" + ", ".join(f"{x:.8f}" for x in lrs) + "]"
    
# ============================== begin logging ================================
if master_process:
    run_id = str(uuid.uuid4())
    logdir = f'logs/{run_id}/'
    os.makedirs(logdir, exist_ok=True)
    logfile = f'logs/{run_id}.txt'

    print("=== SCHEDULER PLAN ===", flush=True)
    def _plan_line(tag: str, opt, is_sf: bool):
        if opt is None:
            return f"  - {tag}=None"
        if is_sf:
            return f"  - {tag} (Schedule-Free): external linear warmup={args.warmup_steps} → constant"
        else:
            r = _ratio_for_opt(opt)
            return f"  - {tag}: external linear warmup={args.warmup_steps} → cosine to r={r:.4f}"
    print(_plan_line("opt1", optimizer1, opt1_is_sf), flush=True)
    print(_plan_line("opt2", optimizer2, opt2_is_sf), flush=True)

    # Preview the exact multipliers using the same lambdas the schedulers use
    preview = [
        0,
        max(0, args.warmup_steps // 2),
        max(0, args.warmup_steps - 1),
        args.warmup_steps,
        args.num_iterations // 2,
        max(0, args.num_iterations - 1),
    ]
    print("=== LR MULTIPLIER PREVIEW ===", flush=True)
    for s in preview:
        s = max(0, min(int(s), int(args.num_iterations) - 1))
        m1 = lam1(s) if optimizer1 is not None else float("nan")
        m2 = lam2(s) if optimizer2 is not None else float("nan")
        print(f"step={s} opt1_mult={m1:.6f} opt2_mult={m2:.6f}", flush=True)
    print("================================", flush=True)

    # Capture full source for reproducibility
    code = _source_text()

    # Single, consolidated header write to logfile
    with open(logfile, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(_run_config_text() + "\n")
        f.write("=" * 100 + "\n")
        f.write(_env_block_text() + "\n")
        f.write("=" * 100 + "\n")
        f.write(code + "\n")
        f.write("=" * 100 + "\n")

    # Mirror essentials to console
    print(_run_config_text(), flush=True)
# ============================ end begin logging ==============================
        
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        _optimizers_eval(optimizers)  # switch optimizer to evaluation mode
        
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        
        # switch optimizers back to train
        model.train()
        _optimizers_train(optimizers)
        
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(
    f"step:{step+1}/{args.num_iterations} "
    f"{_format_lr(optimizer1, 'opt1_lr')} {_format_lr(optimizer2, 'opt2_lr')} "
    f"train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms "
    f"step_avg:{approx_time/timed_steps:.2f}ms",
    flush=True)
        with open(logfile, "a") as f:
            f.write(
                f"step:{step+1}/{args.num_iterations} "
                f"{_format_lr(optimizer1, 'opt1_lr')} {_format_lr(optimizer2, 'opt2_lr')} "
                f"train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms "
                f"step_avg:{approx_time/timed_steps:.2f}ms\n"
            )

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()