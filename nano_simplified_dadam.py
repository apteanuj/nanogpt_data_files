import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import Muon, DistributedDataLoader, GPT, GPTConfig
from dadapt_adam import DAdaptAdam

#------------------------------------------------------------------------------
def _dadam_bias_correction(k: int, beta1: float, beta2: float, use_bias_correction: bool) -> float:
    if not use_bias_correction:
        return 1.0
    # matches your DAdaptAdam: ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
    return ((1.0 - (beta2 ** (k + 1))) ** 0.5) / (1.0 - (beta1 ** (k + 1)))

@torch.no_grad()
def gather_dadapt_adam_stats(opt) -> list[dict]:
    """
    Return one dict per param group with internal scalars and derived effective LR.
    Also reconstructs sk_l1 by summing |s| across params (weighted by layer_scale r).
    """
    stats = []
    for gi, g in enumerate(opt.param_groups):
        d      = g.get('d', None)
        k      = g.get('k', None)
        lr_s   = g.get('lr', None)               # scheduled lr multiplier
        betas  = g.get('betas', (0.9, 0.999))
        use_bc = g.get('use_bias_correction', False)
        r      = g.get('layer_scale', 1.0)
        Nw     = g.get('numerator_weighted', None)

        # Bias correction and effective scalars
        bc   = _dadam_bias_correction(k if k is not None else 0, betas[0], betas[1], use_bc)
        dlr  = None if (d is None or lr_s is None) else (d * lr_s * bc)   # scalar used inside the step
        eff  = None if dlr is None else (r * dlr)                         # layer-scaled effective lr

        # Reconstruct sk_l1 = sum(r * |s|) across params in this group (matches optimizer’s accumulation)
        sk_l1 = 0.0
        for p in g['params']:
            st = opt.state.get(p, None)
            if not st or 's' not in st:
                continue
            sk_l1 += float(r) * st['s'].abs().sum().item()

        stats.append(dict(
            group_index=gi,
            k=k,
            d=d,
            lr_sched=lr_s,
            betas=betas,
            use_bias_correction=use_bc,
            bias_correction=bc,
            dlr=dlr,
            eff_layer_lr=eff,
            layer_scale=r,
            numerator_weighted=Nw,
            sk_l1=sk_l1,
        ))
    return stats

def log_dadapt_stats_all(optimizers, step, logfile, prefix="DADAPT"):
    """Print + append optimizer stats for each optimizer and group."""
    for oi, opt in enumerate(optimizers):
        try:
            group_stats = gather_dadapt_adam_stats(opt)
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"[warn] failed to gather DAdapt stats for opt{oi}: {e}")
            continue
        for st in group_stats:
            line = (
                f"{prefix} step={step} opt={oi} grp={st['group_index']} "
                f"k={st['k']} d={st['d']:.4e} lr_sched={st['lr_sched']:.4e} "
                f"bc={st['bias_correction']:.4e} dlr={st['dlr']:.4e} "
                f"eff_layer_lr={st['eff_layer_lr']:.4e} r={st['layer_scale']:.3f} "
                f"sk_l1={st['sk_l1']:.3e} N={st['numerator_weighted']:.3e}"
            )
            print(line)
            with open(logfile, "a") as f:
                f.write(line + "\n")


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    # device_batch_size : int = 64 # batch size, in sequences, per device
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    embed_learning_rate : float = 0.002
    muon_learning_rate : float = 0.02
    warmup_iters : int = 0
    warmdown_iters : int = 1500 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
# optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.embed_learning_rate, betas=(0.9, 0.95),
#                                weight_decay=args.weight_decay, fused=True)
# optimizer2 = torch.optim.AdamW(raw_model.transformer.h.parameters(), lr=args.embed_learning_rate, betas=(0.9, 0.95),
#                                weight_decay=args.weight_decay, fused=True)

optimizer1 = DAdaptAdam(raw_model.lm_head.parameters(), lr=1.0, betas=(0.9, 0.99), 
                               weight_decay=args.weight_decay , growth_rate=float('inf'), d0=2e-3, use_bias_correction=True)
optimizer2 = DAdaptAdam(raw_model.transformer.h.parameters(), lr=1.0 , betas=(0.9, 0.99), 
                               weight_decay=args.weight_decay , growth_rate=float('inf'), d0=2e-3, use_bias_correction=True)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

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
            # >>> NEW: log D-AdaptAdam internals
            log_dadapt_stats_all(optimizers, step=step, logfile=logfile)
        # start the clock again
        torch.cuda.synchronize()
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
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()