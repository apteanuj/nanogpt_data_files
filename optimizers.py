import sys
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

## NOTE UPDATE MUON TO HAVE RIGHT RMS AND GRADIENT SCALING TO MATCH TORCH IMPLEMENTATION 
## ALSO INCLUDE NORMUON IN ADDITION AS WELL 
## INCLUDE OTHER OPTIMIZERS FROM https://github.com/epfml/llm-optimizer-benchmark/blob/main/src/optim/


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

# -----------------------------------------------------------------------------
# Muon optimizer with iterate averaging 


class MuonIterAvg(torch.optim.Optimizer):  # [ia] renamed class from `Muon` -> `MuonIterAvg`
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz, with iterate averaging of weights

    Minimal additions:
      - `ia_beta` hyperparameter (default 0.999).
      - Maintains per-param `state['ia']` and `state['step']`.
      - `eval()` / `train()` methods to swap to bias-corrected ia weights and back.
    """

    def __init__(self, params, lr = 3e-4, momentum = 0.95, nesterov = True,
                 backend = 'newtonschulz5', backend_steps = 5,
                 ia_beta = 0.999,  # [ia] new arg with default 0.999
                 rank = 0, world_size = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps,
                        ia_beta=ia_beta,            # [ia] stored in group
                        train_mode=True)              # [ia] track swap state
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def __setstate__(self, state):
        super().__setstate__(state)
        # [ia] ensure `step` is a tensor after load (no other changes)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is not None and "step" in st and not torch.is_tensor(st["step"]):
                    st["step"] = torch.tensor(float(st["step"]), device=p.device)

    @torch.no_grad()
    def eval(self):  # [ia] NEW: swap to bias-corrected ia weights for evaluation/checkpointing
        for group in self.param_groups:
            if group.get('train_mode', True):
                ia_beta = group['ia_beta']
                for p in group['params']:
                    st = self.state.get(p)
                    if st is None:
                        continue
                    # buffer live params for restoration
                    st['param_buffer'] = p.detach().clone()
                    # lazy-init ia/step
                    if 'ia' not in st:
                        st['ia'] = p.detach().clone()
                    if 'step' not in st:
                        st['step'] = torch.tensor(0.0, device=p.device)
                    # bias correction
                    step_t = float(st['step'].item()) if torch.is_tensor(st['step']) else float(st['step'])
                    denom = 1.0 - (ia_beta ** max(step_t, 1.0))
                    if denom <= 0.0:
                        denom = 1.0
                    p.copy_(st['ia'] / denom)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):  # [ia] NEW: restore live weights to continue training
        for group in self.param_groups:
            if not group.get('train_mode', True):
                for p in group['params']:
                    st = self.state.get(p)
                    if st is None:
                        continue
                    if 'param_buffer' in st:
                        p.copy_(st['param_buffer'])
                        del st['param_buffer']
                group['train_mode'] = True

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            ia_beta = group['ia_beta']        # [ia] read ia rate
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion (unchanged)
            params_list = list(group['params'])
            total_params = sum(p.numel() for p in params_list)
            if total_params == 0:
                continue
            # choose a device from first param (unchanged)
            first_dev = None
            for p in params_list:
                first_dev = p.device
                break
            updates_flat = torch.zeros(total_params, device=first_dev, dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(params_list):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % max(1, self.world_size) == self.rank:
                    g = p.grad
                    if g is not None:
                        state = self.state[p]
                        if 'momentum_buffer' not in state:
                            state['momentum_buffer'] = torch.zeros_like(g)
                        if 'step' not in state:                          # [ia] init step
                            state['step'] = torch.tensor(0.0, device=p.device)
                        if 'ia' not in state:                           # [ia] init ia
                            state['ia'] = p.detach().clone()

                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(g)
                        upd = g
                        if group['nesterov']:
                            upd = g.add(buf, alpha=momentum)
                        upd = zeropower_backend(upd, steps=group['backend_steps'])
                        upd *= max(1, upd.size(0)/upd.size(1))**0.5
                        updates_flat[curr_idx:curr_idx+p.numel()] = upd.flatten().to(updates_flat.dtype)
                curr_idx += p.numel()

            # sync updates across devices (unchanged)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates (unchanged except ia lines below)
            curr_idx = 0
            for p in params_list:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                # [ia] per-param: increment step and update ia of *live* weights
                st = self.state[p]
                if 'step' not in st:
                    st['step'] = torch.tensor(0.0, device=p.device)
                st['step'] += 1.0
                if 'ia' not in st:
                    st['ia'] = p.detach().clone()
                st['ia'].lerp_(p, 1.0 - ia_beta)  # ia = ia_beta*ia + (1-ia_beta)*p
                # [/ia]
                curr_idx += p.numel()


# -----------------------------------------------------------------------------
# Muon optimizer with Lion style averaging 


class Gluon(torch.optim.Optimizer):
    """
    Generalized Lion update with Ortho Normalization
    Now with decoupled weight decay (AdamW-style).
    """
    def __init__(self, params, lr=3e-4,
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