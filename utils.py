import sys
# with open(sys.argv[0]) as f:
#     code = f.read() # read the code of this file ASAP, for logging
import glob

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


class Luon(torch.optim.Optimizer):
    """
    Luon - MomentUm Orthogonalized by Newton-Schulz

    Now uses Lion-style momentum:
      update_seed = beta1 * exp_avg + (1 - beta1) * grad
      exp_avg     <- beta2 * exp_avg + (1 - beta2) * grad

    Direction:
      - If param is 2D: NS5 polar( update_seed ) with same scaling as before
      - If param is 1D: sign( update_seed )  (Lion-style)
    """
    def __init__(self, params, lr=3e-4,
                 momentum=0.95, nesterov=True,               # kept for compatibility; unused
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1,
                 betas=(0.9, 0.95)):                          # NEW: Lion betas
        defaults = dict(lr=lr,
                        momentum=momentum, nesterov=nesterov,  # kept; not used
                        backend=backend, backend_steps=backend_steps,
                        betas=betas)                           # NEW
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            beta1, beta2 = group['betas']                     # NEW
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue

                    state = self.state[p]
                    if 'exp_avg' not in state:                # NEW: Lion EMA buffer
                        state['exp_avg'] = torch.zeros_like(g)
                    exp_avg = state['exp_avg']

                    # Lion-style blended seed
                    u = exp_avg * beta1 + g * (1.0 - beta1)   # NEW

                    # Direction: NS5 for matrices, sign for vectors
                    if p.ndim == 1:
                        g_dir = u.sign()
                    else:
                        g_dir = zeropower_backend(u, steps=group['backend_steps'])
                        g_dir *= max(1, g_dir.size(0)/g_dir.size(1))**0.5  # same scaling as before

                    # Write into flat buffer (dtype bf16 preserved)
                    updates_flat[curr_idx:curr_idx+p.numel()] = g_dir.flatten().to(updates_flat.dtype)

                    # Update Lion EMA after forming the step (matches Lion)
                    exp_avg.mul_(beta2).add_(g, alpha=1.0 - beta2)   # NEW

                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


class Gluon(torch.optim.Optimizer):
    """
    Generalized Lion update with Ortho Normalization
    Now with decoupled weight decay (AdamW-style).
    """
    def __init__(self, params, lr=3e-4,
                 momentum=0.95, nesterov=True,               # kept for compatibility; unused
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1,
                 betas=(0.9, 0.95),                          # Lion betas
                 weight_decay=0.0,                            # NEW
                 decouple_bias_norm=True):                    # NEW
        defaults = dict(
            lr=lr,
            momentum=momentum, nesterov=nesterov,            # kept; not used
            backend=backend, backend_steps=backend_steps,
            betas=betas,
            weight_decay=weight_decay,
            decouple_bias_norm=decouple_bias_norm,
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            beta1, beta2 = group['betas']                     # NEW
            wd = group.get('weight_decay', 0.0)
            dec_bias_norm = group.get('decouple_bias_norm', True)
            zeropower_backend = zeropower_backends[group['backend']]


            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue

                    state = self.state[p]
                    if 'exp_avg' not in state:                # NEW: Lion EMA buffer
                        state['exp_avg'] = torch.zeros_like(g)
                    exp_avg = state['exp_avg']

                    # Lion-style blended seed
                    u = exp_avg * beta1 + g * (1.0 - beta1)   # NEW

                    # Direction: NS5 for matrices, sign for vectors
                    if p.ndim == 1:
                        g_dir = u.sign()
                    else:
                        g_dir = zeropower_backend(u, steps=group['backend_steps'])
                        g_dir *= max(1, g_dir.size(0)/g_dir.size(1))**0.5  # same scaling as before
                        # g_dir *= float(max(g_dir.size(0), g_dir.size(1)))**0.5 # to ensure that the RMS matches that of the Lion Update following Kimi Paper https://arxiv.org/pdf/2502.16982

                    # Write into flat buffer (dtype bf16 preserved)
                    updates_flat[curr_idx:curr_idx+p.numel()] = g_dir.flatten().to(updates_flat.dtype)

                    # Update Lion EMA after forming the step (matches Lion)
                    exp_avg.mul_(beta2).add_(g, alpha=1.0 - beta2)   # NEW

                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize, apply decoupled weight decay, then apply direction step
            curr_idx = 0
            for p in group['params']:
                g_dir = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)

                # --- Decoupled weight decay (AdamW style) ---
                if wd != 0.0:
                    if not (dec_bias_norm and p.ndim == 1):   # typically skip biases/norms
                        # p <- (1 - lr*wd) * p
                        p.data.mul_(1.0 - lr * wd)

                # Step along direction (sign or polar)
                p.data.add_(g_dir, alpha=-lr)

                curr_idx += p.numel()
                
# DOES NOT WORK VEY WELL NEED  TO STUDY WHY ?
# class Luon(torch.optim.Optimizer):
#     """
#     Luon - Lion-style momentum + NS5 polar direction for 2D params, Lion sign for 1D.

#       update_seed = beta1 * exp_avg + (1 - beta1) * grad
#       exp_avg     <- beta2 * exp_avg + (1 - beta2) * grad

#     Direction:
#       - 2D params: g_dir = NS5_polar(update_seed); then scale by sqrt(max(h/w,1))
#       - 1D params: g_dir = sign(update_seed)
#     """
#     def __init__(self, params, lr=3e-4,
#                  momentum=0.95, nesterov=True,               # kept for compatibility; unused
#                  backend='newtonschulz5', backend_steps=5,
#                  rank=0, world_size=1,
#                  betas=(0.9, 0.95)):
#         defaults = dict(
#             lr=lr,
#             momentum=momentum, nesterov=nesterov,  # kept; not used
#             backend=backend, backend_steps=backend_steps,
#             betas=betas
#         )
#         super().__init__(params, defaults)
#         self.rank = rank
#         self.world_size = world_size
#         # per-group scratch buffers live in group['_updates_flat']

#     def step(self):
#         for group in self.param_groups:
#             lr = group['lr']
#             beta1, beta2 = group['betas']
#             zeropower_backend = zeropower_backends[group['backend']]

#             # flat buffer reused across steps for THIS group
#             total_params = sum(p.numel() for p in group['params'])
#             uf = group.get('_updates_flat', None)
#             if uf is None or uf.numel() != total_params:
#                 uf = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
#                 group['_updates_flat'] = uf
#             else:
#                 uf.zero_()
#             updates_flat = uf

#             curr_idx = 0
#             for i, p in enumerate(group['params']):
#                 # simple striped assignment across ranks
#                 if i % self.world_size == self.rank:
#                     g = p.grad
#                     if g is None:
#                         curr_idx += p.numel()
#                         continue

#                     # state init (keep EMA in same dtype as grad to avoid casts)
#                     state = self.state[p]
#                     if 'exp_avg' not in state:
#                         state['exp_avg'] = torch.zeros_like(g)
#                     exp_avg = state['exp_avg']

#                     # ---- Build Lion seed IN-PLACE on exp_avg to avoid temp 'u' ----
#                     # Now: exp_avg := beta1 * exp_avg
#                     exp_avg.mul_(beta1)
#                     # Let 'u' alias exp_avg memory; u = beta1*e_old + (1-beta1)*g
#                     u = exp_avg
#                     u.add_(g, alpha=(1.0 - beta1))
#                     # ----------------------------------------------------------------

#                     # ---- Direction ----
#                     if p.ndim == 1:
#                         g_dir = u.sign()
#                     else:
#                         # NS5 on the seed; same scaling as your Muon
#                         g_dir = zeropower_backend(u, steps=group['backend_steps'])
#                         g_dir *= max(1.0, g_dir.size(0) / g_dir.size(1)) ** 0.5
#                     # Write to bf16 flat buffer once
#                     updates_flat[curr_idx:curr_idx + p.numel()] = g_dir.flatten().to(updates_flat.dtype)

#                     # ---- EMA update using the scaled exp_avg we already have ----
#                     # exp_avg currently holds beta1 * e_old; convert to beta2*e_old + (1-beta2)*g
#                     if beta1 > 0:
#                         exp_avg.mul_(beta2 / beta1).add_(g, alpha=(1.0 - beta2))
#                     else:
#                         exp_avg.zero_().add_(g, alpha=(1.0 - beta2))
#                     # ----------------------------------------------------------------

#                 # advance slice (exactly once per param)
#                 curr_idx += p.numel()

#             # sum contributions from all ranks
#             dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

#             # deserialize and apply updates
#             curr_idx = 0
#             for p in group['params']:
#                 g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
#                 p.data.add_(g, alpha=-lr)
#                 curr_idx += p.numel()          
# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()
#------------------------------------------------------------------------------
# D Adaptation version of Muon 

import math
import torch
import torch.distributed as dist
from torch.optim import Optimizer

# _SOP_EPS = 1e-12      # avoid divide-by-zero for ||S||_op
# _DOT_EPS = 0.0        # keep 0; change if you want to bias num upward slightly
# _SAN_POSINF = 1e6     # clamp infs when sanitizing tensors
# _SAN_NEGINF = -1e6

# @torch.no_grad()
# def _isfinite_tensor(x: torch.Tensor) -> bool:
#     return torch.isfinite(x).all().item()

# @torch.no_grad()
# def _nan_to_num_(x: torch.Tensor):
#     # in-place sanitize
#     return torch.nan_to_num_(x, nan=0.0, posinf=_SAN_POSINF, neginf=_SAN_NEGINF)

# # ---- spectral norm via short power iteration (3 steps) with sanitization ----
# @torch.no_grad()
# def _spec_norm_power_2d(M2d: torch.Tensor, v: torch.Tensor | None = None, iters: int = 3, eps: float = 1e-12):
#     """
#     Approximate ||M||_2 for a 2D matrix M using power iteration.
#     Returns (sigma (float), v_new (Tensor)). Works in float32 for stability.
#     """
#     M = M2d.float()
#     _nan_to_num_(M)
#     m, n = M.shape
#     if v is None:
#         v = torch.randn(n, device=M.device, dtype=M.dtype)
#     v = v / (v.norm() + eps)
#     for _ in range(iters):
#         u = M @ v
#         un = u.norm() + eps
#         u = u / un
#         v = M.t() @ u
#         vn = v.norm() + eps
#         v = v / vn
#     sigma = (M @ v).norm()
#     return float(sigma.item()), v


# class DMuon(Optimizer):
#     """
#     DMuon — Muon (NS-5 orthogonalized) with D-Adaptation (SGD-style, operator-norm geometry)

#     - Simple momentum (no Nesterov). Default lr=1.0.
#     - Effective step per iter: alpha_k = lr * d_k.
#     - Accumulator S mirrors the actually applied (all-reduced) update.
#     - Numerator uses <H_k, S_{k-1}> with H_k = u / s (unit spectral).
#     - Denominator is ||S||_op (block-max spectral; power iteration).
#     - No gamma, no weight decay. Includes NaN/Inf guards.

#     For clean theory, set momentum=0. For stability, you can also keep rho finite (e.g., 1.02).
#     """
#     def __init__(self, params, lr=1.0, momentum=0.0,
#                  backend='newtonschulz5', backend_steps=5,
#                  d0=1e-6, rho=float('inf'), power_iters=3,
#                  rank=0, world_size=1, log_every=0):
#         if lr <= 0 or d0 <= 0:
#             raise ValueError("lr and d0 must be > 0")
#         defaults = dict(lr=lr, momentum=momentum,
#                         backend=backend, backend_steps=backend_steps,
#                         d=d0, num=0.0, k=0, rho=rho, power_iters=power_iters,
#                         log_every=log_every, initial_lr=lr)
#         super().__init__(params, defaults)
#         self.rank = rank
#         self.world_size = world_size

#     @torch.no_grad()
#     def step(self):
#         for group in self.param_groups:
#             lr   = group['lr']
#             mom  = group['momentum']
#             d    = group['d']
#             num  = group['num']
#             k    = group['k']
#             rho  = group['rho']
#             pit  = group['power_iters']
#             log_every = group['log_every']

#             alpha = lr * d  # effective LR
#             backend = zeropower_backends[group['backend']]

#             # 1) Build flattened, distributed updates + accumulate inner product with unit-spectral H
#             total_params = sum(p.numel() for p in group['params'])
#             updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
#             curr_idx = 0
#             delta_num_local = 0.0  # sum_i <H, S_{k-1}>, pre-alpha

#             for i, p in enumerate(group['params']):
#                 if i % self.world_size == self.rank:
#                     g = p.grad
#                     if g is None:
#                         curr_idx += p.numel()
#                         continue

#                     state = self.state[p]
#                     if 'momentum_buffer' not in state:
#                         state['momentum_buffer'] = torch.zeros_like(g)
#                     if 'S' not in state:
#                         state['S'] = torch.zeros_like(p.data)
#                     if 'pow_v' not in state:
#                         state['pow_v'] = None

#                     # simple momentum (Polyak)
#                     buf = state['momentum_buffer']
#                     _nan_to_num_(buf)  # just in case
#                     _nan_to_num_(g)
#                     buf.mul_(mom).add_(g)
#                     g_eff = buf if mom != 0.0 else g

#                     # orthogonalize & shape-scale (Muon)
#                     u = backend(g_eff, steps=group['backend_steps'])
#                     _nan_to_num_(u)
#                     # shape scale s = sqrt(max(1, m/n))
#                     m, n = u.size(0), u.size(1)
#                     s = math.sqrt(max(1.0, float(m) / float(max(1, n))))
#                     u = u * s  # apply the same scaling as Muon

#                     # fill shard
#                     updates_flat[curr_idx:curr_idx + p.numel()] = u.flatten().to(updates_flat.dtype)

#                     # --- Correct numerator: use H = u / s (unit spectral) vs prior S ---
#                     # guard: if s somehow zero (shouldn't be), skip this contribution
#                     if s > 0.0:
#                         S_prev = state['S'].float()
#                         _nan_to_num_(S_prev)
#                         h_unit = (u.float() / s)
#                         _nan_to_num_(h_unit)
#                         # dot(H, S_{k-1}): tr(H^T S)
#                         delta_num_local += float(torch.dot(h_unit.flatten(), S_prev.flatten()).item() + _DOT_EPS)

#                 curr_idx += p.numel()

#             # sync updates across devices (sum)
#             dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

#             # Ensure per-param state exists on this rank before dereference below
#             for p in group['params']:
#                 st = self.state[p]
#                 if 'S' not in st:
#                     st['S'] = torch.zeros_like(p.data)
#                 if 'pow_v' not in st:
#                     st['pow_v'] = None
#                 if 'momentum_buffer' not in st:
#                     proto = p.grad if p.grad is not None else p.data
#                     st['momentum_buffer'] = torch.zeros_like(proto)

#             # 2) Deserialize, apply updates with alpha, advance S; track ||S||_op and stats
#             curr_idx = 0
#             max_spec_S_local = 0.0
#             u_norm_sum = 0.0
#             u_count = 0
#             S_frob_sq_local = 0.0

#             for p in group['params']:
#                 u = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).to(p.data.dtype)
#                 curr_idx += p.numel()

#                 _nan_to_num_(u)
#                 # X_{k+1} = X_k - alpha * u   (skip if alpha is not finite)
#                 if math.isfinite(alpha):
#                     p.data.add_(u, alpha=-alpha)

#                 # S_k = S_{k-1} + alpha * u
#                 state = self.state[p]
#                 S = state['S']
#                 _nan_to_num_(S)
#                 if math.isfinite(alpha):
#                     S.add_(u, alpha=alpha)
#                 _nan_to_num_(S)

#                 # stats
#                 u_norm_sum += float(u.float().norm().item()); u_count += 1
#                 S_frob_sq_local += float(S.float().pow(2).sum().item())

#                 # estimate ||S||_2 via power iteration on 2D view
#                 if S.ndim == 2:
#                     S2 = S
#                 else:
#                     S2 = S.reshape(S.shape[0], -1)
#                 spec, v_new = _spec_norm_power_2d(S2, state['pow_v'], iters=pit)
#                 state['pow_v'] = v_new
#                 if math.isfinite(spec) and spec > max_spec_S_local:
#                     max_spec_S_local = spec

#             # 3) All-reduce scalars and stats
#             # num increment = alpha * sum_i <H, S_{k-1}>
#             delta_num = alpha * delta_num_local
#             # sanitize before reduce
#             if not math.isfinite(delta_num):
#                 delta_num = 0.0
#             delta_num_t = torch.tensor([delta_num], device=updates_flat.device, dtype=torch.float32)
#             dist.all_reduce(delta_num_t, op=dist.ReduceOp.SUM)
#             num = num + float(delta_num_t.item())

#             # MAX reduce for Sop
#             max_spec_S_t = torch.tensor([max_spec_S_local], device=updates_flat.device, dtype=torch.float32)
#             dist.all_reduce(max_spec_S_t, op=dist.ReduceOp.MAX)
#             Sop = float(max_spec_S_t.item())

#             # optional stats
#             u_norm_sum_t = torch.tensor([u_norm_sum], device=updates_flat.device, dtype=torch.float32)
#             u_count_t    = torch.tensor([u_count],    device=updates_flat.device, dtype=torch.float32)
#             dist.all_reduce(u_norm_sum_t, op=dist.ReduceOp.SUM)
#             dist.all_reduce(u_count_t,    op=dist.ReduceOp.SUM)
#             avg_u_norm = float((u_norm_sum_t / torch.clamp(u_count_t, min=1.0)).item())

#             S_frob_sq_t = torch.tensor([S_frob_sq_local], device=updates_flat.device, dtype=torch.float32)
#             dist.all_reduce(S_frob_sq_t, op=dist.ReduceOp.SUM)
#             Sfro = float(torch.sqrt(torch.clamp(S_frob_sq_t, min=0)).item())

#             # 4) D update: require Sop large enough and finite num
#             if Sop > _SOP_EPS and math.isfinite(num):
#                 d_hat = (2.0 * num) / Sop
#                 if math.isfinite(d_hat) and d_hat > 0.0:
#                     # paper: d = max(d, d_hat)
#                     # safer (optional): d = max(d, min(d_hat, rho * d))
#                     d = max(d, min(d_hat, rho * d)) if math.isfinite(rho) else max(d, d_hat)
#                 else:
#                     d_hat = float('nan')
#             else:
#                 d_hat = float('nan')

#             # 5) write back
#             group['d'] = d
#             group['num'] = num
#             group['k'] = k + 1

#             # 6) debug print
#             if log_every and ((k % log_every) == 0) and (self.rank == 0):
#                 lr_sched = lr
#                 lr_base  = group['initial_lr']
#                 alpha_now = lr_sched * d if math.isfinite(d) else float('inf')
#                 alpha_base = lr_base * d if math.isfinite(d) else float('inf')
#                 print(
#                     "[DMuon]"
#                     f" k={k} "
#                     f"alpha={alpha_now:.3e} alpha_base={alpha_base:.3e} "
#                     f"lr_sched={lr_sched:.3e} lr_base={lr_base:.3e} "
#                     f"d={d:.6e} d_hat={d_hat:.6e} "
#                     f"Sop={Sop:.6e} Sfro={Sfro:.6e} "
#                     f"num={num:.6e} avg||u||={avg_u_norm:.3e}"
#                 )

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss