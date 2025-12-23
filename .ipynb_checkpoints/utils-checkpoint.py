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

class MuonEMA(torch.optim.Optimizer):  # [EMA] renamed class from `Muon` -> `MuonEMA`
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz, with EMA evaluation weights (minimal changes).

    Minimal additions:
      - `ema_beta` hyperparameter (default 0.999).
      - Maintains per-param `state['ema']` and `state['step']`.
      - `eval()` / `train()` methods to swap to bias-corrected EMA weights and back.
    """

    def __init__(self, params, lr = 3e-4, momentum = 0.95, nesterov = True,
                 backend = 'newtonschulz5', backend_steps = 5,
                 ema_beta = 0.999,  # [EMA] new arg with default 0.999
                 rank = 0, world_size = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps,
                        ema_beta=ema_beta,            # [EMA] stored in group
                        train_mode=True)              # [EMA] track swap state
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def __setstate__(self, state):
        super().__setstate__(state)
        # [EMA] ensure `step` is a tensor after load (no other changes)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is not None and "step" in st and not torch.is_tensor(st["step"]):
                    st["step"] = torch.tensor(float(st["step"]), device=p.device)

    @torch.no_grad()
    def eval(self):  # [EMA] NEW: swap to bias-corrected EMA weights for evaluation/checkpointing
        for group in self.param_groups:
            if group.get('train_mode', True):
                ema_beta = group['ema_beta']
                for p in group['params']:
                    st = self.state.get(p)
                    if st is None:
                        continue
                    # buffer live params for restoration
                    st['param_buffer'] = p.detach().clone()
                    # lazy-init ema/step
                    if 'ema' not in st:
                        st['ema'] = p.detach().clone()
                    if 'step' not in st:
                        st['step'] = torch.tensor(0.0, device=p.device)
                    # bias correction
                    step_t = float(st['step'].item()) if torch.is_tensor(st['step']) else float(st['step'])
                    denom = 1.0 - (ema_beta ** max(step_t, 1.0))
                    if denom <= 0.0:
                        denom = 1.0
                    p.copy_(st['ema'] / denom)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):  # [EMA] NEW: restore live weights to continue training
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
            ema_beta = group['ema_beta']        # [EMA] read EMA rate
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
                        if 'step' not in state:                          # [EMA] init step
                            state['step'] = torch.tensor(0.0, device=p.device)
                        if 'ema' not in state:                           # [EMA] init ema
                            state['ema'] = p.detach().clone()

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

            # deserialize and apply updates (unchanged except EMA lines below)
            curr_idx = 0
            for p in params_list:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                # [EMA] per-param: increment step and update EMA of *live* weights
                st = self.state[p]
                if 'step' not in st:
                    st['step'] = torch.tensor(0.0, device=p.device)
                st['step'] += 1.0
                if 'ema' not in st:
                    st['ema'] = p.detach().clone()
                st['ema'].lerp_(p, 1.0 - ema_beta)  # ema = ema_beta*ema + (1-ema_beta)*p
                # [/EMA]
                curr_idx += p.numel()

############################################
class MuonSF(torch.optim.Optimizer):
    r"""
    Schedule-Free Muon (no momentum) with AUTOMATIC y-blend via a model forward pre-hook.

    Eqs. (3)-(5):
        y_t       = (1 - β) z_t + β x_t
        z_{t+1}   = z_t - γ_t * NS5(∇f(y_t))
        x_{t+1}   = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},  where c_{t+1} = γ_t^2 / Σ_{i≤t} γ_i^2

    Zero loop changes:
      - We register a forward pre-hook on the model to swap z->y before *every* train-time forward.
      - step() auto-restores z and updates (z, x).

    Usage A (one-liner change at construction):
        opt = MuonSF_AutoY(model.parameters(), model=model, lr=..., beta=0.9)

    Usage B (attach later, still no loop changes):
        opt = MuonSF_AutoY(model.parameters(), lr=..., beta=0.9)
        opt.attach(model)

    Eval / checkpoint (unchanged pattern):
        opt.eval_avg(); validate(...); opt.train_live()
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
        beta: float = 0.9,
        rank: int = 0,
        world_size: int = 1
    ):
        defaults = dict(
            lr=lr,
            backend=backend,
            backend_steps=backend_steps,
            beta=beta,
            train_mode=True,
            _y_active=False,   # currently at y?
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._model_ref: Optional[weakref.ReferenceType] = None
        self._hook_handle = None

        # running sum S = Σ γ^2 per group (for c_t)
        for g in self.param_groups:
            g["_S_gamma2"] = 0.0

        if model is not None:
            self.attach(model)

    # ---------- Public: attach/detach model hook ----------
    def attach(self, model: torch.nn.Module):
        """Register a forward pre-hook to auto-apply y before train-time forwards."""
        self.detach()  # in case re-attaching
        self._model_ref = weakref.ref(model)

        def _pre_forward_hook(mod, inputs):
            # Only when training *and* grads enabled (skip eval/inference)
            if not self._any_group("train_mode") or not torch.is_grad_enabled():
                return
            # If we are already at y (e.g., multiple forwards per step for grad-accum),
            # do nothing; otherwise blend to y.
            for group in self.param_groups:
                if not group.get("_y_active", False):
                    self._set_y_group(group)
            return  # no modification to inputs

        self._hook_handle = model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=False)

    def detach(self):
        """Remove the registered forward pre-hook, if present."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._model_ref = None

    # ---------- Internal helpers ----------
    def _any_group(self, key: str) -> bool:
        for g in self.param_groups:
            if g.get(key, False):
                return True
        return False

    @torch.no_grad()
    def _set_y_group(self, group):
        """Blend params in this group to y = (1-β)z + βx; stash z in 'param_buffer'."""
        beta = float(group["beta"])
        for p in group["params"]:
            if p is None:
                continue
            st = self.state.get(p)
            if st is None:
                st = self.state[p] = {}
            if "x" not in st:
                st["x"] = p.detach().clone()
            if "param_buffer" in st:
                continue  # already at y for this param
            st["param_buffer"] = p.detach().clone()  # keep z
            p.copy_(st["param_buffer"] * (1.0 - beta) + st["x"] * beta)  # write y
        group["_y_active"] = True

    @torch.no_grad()
    def _auto_restore_from_y(self, group):
        """If currently at y, restore params to z once before applying updates."""
        if not group.get("_y_active", False):
            return
        for p in group["params"]:
            if p is None:
                continue
            st = self.state.get(p)
            if st and "param_buffer" in st:
                p.copy_(st["param_buffer"])
                del st["param_buffer"]
        group["_y_active"] = False

    # ---------- Eval/Checkpoint swaps ----------
    @torch.no_grad()
    def eval_avg(self):
        """Swap to averaged weights x for evaluation/checkpointing."""
        for group in self.param_groups:
            if group.get("train_mode", True):
                # ensure we're not at y for safety
                self._auto_restore_from_y(group)
                for p in group["params"]:
                    if p is None:
                        continue
                    st = self.state.get(p)
                    if st is None:
                        st = self.state[p] = {}
                    if "x" not in st:
                        st["x"] = p.detach().clone()
                    st["live_buffer_eval"] = p.detach().clone()
                    p.copy_(st["x"])
                group["train_mode"] = False

    @torch.no_grad()
    def train_live(self):
        """Restore training (z) weights after eval_avg()."""
        for group in self.param_groups:
            if not group.get("train_mode", True):
                for p in group["params"]:
                    if p is None:
                        continue
                    st = self.state.get(p)
                    if st and "live_buffer_eval" in st:
                        p.copy_(st["live_buffer_eval"])
                        del st["live_buffer_eval"]
                group["train_mode"] = True

    # ---------- Optimizer step ----------
    @torch.no_grad()
    def step(self, closure=None):
        """
        Auto-restore to z if currently at y, then:
          - build NS-5 processed updates using grads (which were taken at y)
          - z <- z - lr * upd
          - x <- (1 - c)x + c z, with c = lr^2 / Σ lr^2
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            backend = zeropower_backends[group["backend"]]
            steps = int(group["backend_steps"])

            # ensure back on z before applying updates
            self._auto_restore_from_y(group)

            params_list = list(group["params"])
            total_params = sum(p.numel() for p in params_list if p is not None)
            if total_params == 0:
                continue

            # device from first param
            first_dev = None
            for p in params_list:
                if p is not None:
                    first_dev = p.device
                    break

            updates_flat = torch.zeros(total_params, device=first_dev, dtype=torch.bfloat16)

            # build NS-5 processed updates (no momentum)
            curr = 0
            for i, p in enumerate(params_list):
                if p is None:
                    continue
                if i % max(1, self.world_size) == self.rank:
                    g = p.grad
                    if g is not None:
                        st = self.state.get(p)
                        if st is None:
                            st = self.state[p] = {}
                        if "x" not in st:
                            st["x"] = p.detach().clone()
                        upd = backend(g, steps=steps)
                        if upd.ndim >= 2:
                            upd = upd * (max(1, upd.size(0) / max(1, upd.size(1))) ** 0.5)
                        updates_flat[curr:curr + p.numel()] = upd.flatten().to(updates_flat.dtype)
                curr += p.numel()

            # distributed sum
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # gamma^2-weighted averaging coefficient c_t
            group["_S_gamma2"] += lr * lr
            c = (lr * lr) / group["_S_gamma2"]

            # apply z-step, then update x
            curr = 0
            for p in params_list:
                if p is None:
                    continue
                upd = updates_flat[curr:curr + p.numel()].view_as(p.data).type_as(p.data)
                curr += p.numel()

                # z_{t+1} = z_t - lr * upd
                p.data.add_(upd, alpha=-lr)

                # x_{t+1} = (1 - c)x + c*z
                st = self.state[p]
                if "x" not in st:
                    st["x"] = p.detach().clone()
                st["x"].lerp_(p, c)

        return loss

#######################################
import math

# assume zeropower_backends is defined elsewhere, as in your original code

import math
import torch
import torch.distributed as dist

# Assumes you already define: zeropower_backends = {"newtonschulz5": ... , ...}

class CautiousRowMuon(torch.optim.Optimizer):
    """
    Muon with cautious masking against the *instantaneous* gradient (pre-momentum).

    Per 2D parameter:
      1) g_inst = p.grad (detached)           # capture BEFORE momentum/Nesterov
      2) Apply momentum/Nesterov to get g     # as usual
      3) u = zeropower_backend(g); Muon scale
      4) Compare u vs g_inst along the chosen axis using cosine with a small margin:
         - 'row' (default): mask rows with cos_row <= margin
         - 'col':          mask columns with cos_col <= margin
         - 'head':         mask heads with cos_head <= margin  (requires num_heads, head_dim)
         Then divide by mask mean (paper's scaling) to preserve RMS.
      5) All-reduce and apply LR. Weight decay (if any) should be decoupled outside.

    New param-group options (all optional):
      - cautious_axis: 'row' | 'col' | 'head'   (default 'row')
      - cautious_margin: float cosine margin    (default 0.01)
      - num_heads, head_dim: required for 'head'
      - agree_log: bool to print fractions      (default True)
      - agree_log_every: int steps              (default 125)
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            backend=backend, backend_steps=backend_steps,
            cautious_axis='row',
            cautious_margin=0.01,
            num_heads=None, head_dim=None,
            agree_log=True,
            agree_log_every=125
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._step_idx = 0

    def step(self):
        self._step_idx += 1

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            axis = group.get('cautious_axis', 'row')
            margin = float(group.get('cautious_margin', 0.01))
            num_heads = group.get('num_heads', None)
            head_dim  = group.get('head_dim', None)

            do_agree_log = (
                bool(group.get('agree_log', True))
                and self.rank == 0
                and (self._step_idx % max(1, int(group.get('agree_log_every', 125))) == 0)
            )

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0

            for i, p in enumerate(group['params']):
                if i % self.world_size != self.rank:
                    curr_idx += p.numel()
                    continue

                g = p.grad
                if g is None:
                    curr_idx += p.numel()
                    continue

                # --- capture instantaneous gradient BEFORE momentum/Nesterov ---
                g_inst = g.detach()

                # momentum + optional Nesterov (unchanged)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)      # SGD momentum on current grad
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                # Muon orthogonalization backend (e.g., Newton–Schulz)
                u = zeropower_backend(g, steps=group['backend_steps'])
                # Muon’s size scaling to match RMS across aspect ratios
                u = u * (max(1.0, u.size(0) / u.size(1)) ** 0.5)

                # ---- cautious masking vs instantaneous gradient (pre-momentum) ----
                # match dtype/device for numerical stability (bf16)
                gr = g_inst.to(dtype=u.dtype, device=u.device)
                eps = torch.finfo(u.dtype).eps

                if axis == 'row':
                    # row agreements: cosine per row
                    r = (u * gr).sum(dim=1)                       # [m]
                    u_row_n = (u * u).sum(dim=1).sqrt() + eps     # [m]
                    g_row_n = (gr * gr).sum(dim=1).sqrt() + eps   # [m]
                    cos_row = r / (u_row_n * g_row_n)
                    m = (cos_row > margin).to(u.dtype).unsqueeze(1)   # [m,1]
                    mbar = m.mean()
                    # optional logging
                    if do_agree_log:
                        frac = float(mbar.item())
                        print(f"[CautiousMuon step {self._step_idx}] "
                              f"param_idx={i} shape={tuple(p.shape)} "
                              f"frac_rows_agree={frac:.3f} axis=row")
                    # divide-by-mean scaling (paper)
                    u = (m * u) / (mbar + eps)

                elif axis == 'col':
                    # column agreements: cosine per column
                    c = (u * gr).sum(dim=0)                       # [n]
                    u_col_n = (u * u).sum(dim=0).sqrt() + eps     # [n]
                    g_col_n = (gr * gr).sum(dim=0).sqrt() + eps   # [n]
                    cos_col = c / (u_col_n * g_col_n)
                    m = (cos_col > margin).to(u.dtype).unsqueeze(0)   # [1,n]
                    mbar = m.mean()
                    if do_agree_log:
                        frac = float(mbar.item())
                        print(f"[CautiousMuon step {self._step_idx}] "
                              f"param_idx={i} shape={tuple(p.shape)} "
                              f"frac_cols_agree={frac:.3f} axis=col")
                    u = (m * u) / (mbar + eps)

                elif axis == 'head':
                    # head agreements: group consecutive rows per head
                    assert num_heads is not None and head_dim is not None, \
                        "head masking requires num_heads and head_dim"
                    m_rows = u.size(0)
                    assert m_rows == num_heads * head_dim, \
                        f"rows ({m_rows}) must equal num_heads*head_dim ({num_heads*head_dim})"
                    # flatten rows within each head, compute cosine per head
                    u_rows = u.view(num_heads, head_dim, -1)      # [H, Dh, n]
                    g_rows = gr.view(num_heads, head_dim, -1)
                    # dot and norms per head
                    dot_h = (u_rows * g_rows).sum(dim=(1,2))      # [H]
                    u_h_n = (u_rows * u_rows).sum(dim=(1,2)).sqrt() + eps
                    g_h_n = (g_rows * g_rows).sum(dim=(1,2)).sqrt() + eps
                    cos_h = dot_h / (u_h_n * g_h_n)               # [H]
                    mh = (cos_h > margin).to(u.dtype).view(num_heads, 1).repeat(1, head_dim)  # [H,Dh]
                    m = mh.reshape(m_rows, 1)                     # broadcast to rows
                    mbar = m.mean()
                    if do_agree_log:
                        frac_heads = float((cos_h > margin).to(u.dtype).mean().item())
                        print(f"[CautiousMuon step {self._step_idx}] "
                              f"param_idx={i} shape={tuple(p.shape)} "
                              f"frac_heads_agree={frac_heads:.3f} axis=head")
                    u = (m * u) / (mbar + eps)

                else:
                    # unknown axis: no cautious mask
                    pass
                # -------------------------------------------------------------------

                # serialize to flat buffer
                updates_flat[curr_idx:curr_idx + p.numel()] = u.flatten()
                curr_idx += p.numel()

            # sync updates across devices (simple deserialization)
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                u = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(u, alpha=-lr)
                curr_idx += p.numel()


class CautiousMuon(torch.optim.Optimizer):
    """
    Muon + Cautious cone-projection (reflect variant) with optional diagnostics.

    Diagnostics (when enabled):
      - prints inner product <u, g_raw>, norms, cosθ, θ(deg), and action (reflect/keep)
      - only prints on rank==0
      - prints at most `max_logs_per_step` entries per step, every `log_every` steps

    Args (new):
      cautious_log (bool): enable/disable diagnostics (default: True)
      log_every (int): print every N steps (default: 125)
      max_logs_per_step (int): cap number of per-param logs per step (default: 8)
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1,
                 cautious_log=True, log_every=125, max_logs_per_step=8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self.cautious_log = cautious_log
        self.log_every = log_every
        self.max_logs_per_step = max_logs_per_step
        self._step_idx = 0

    def step(self):
        self._step_idx += 1
        do_log = (self.cautious_log and self.rank == 0 and
                  (self._step_idx % max(1, self.log_every) == 0))

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0

            # local counters for simple per-step stats
            n_params_local = 0
            n_reflect_local = 0
            logs_emitted = 0

            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue

                    # keep a copy of the raw gradient for the cone test (match dtype/device later)
                    g_raw = g.detach()

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)

                    # Muon orthogonalization backend (e.g., Newton–Schulz)
                    u = zeropower_backend(g, steps=group['backend_steps'])
                    # Muon’s size scaling to match RMS across aspect ratios
                    u = u * (max(1.0, u.size(0) / u.size(1)) ** 0.5)

                    # ----- Cautious cone-projection (reflect) -----
                    gr = g_raw.to(dtype=u.dtype, device=u.device)
                    dot = (u * gr).sum()
                    eps = torch.finfo(u.dtype).eps
                    g2  = (gr * gr).sum().clamp_min(eps)
                    act = "keep"
                    if dot < 0:
                        # reflect along g_raw: u' = u - 2 * proj_{g_raw}(u)
                        u = u - 2.0 * (dot / g2) * gr
                        act = "reflect"
                        n_reflect_local += 1
                    n_params_local += 1
                    # ---------------------------------------------

                    # Diagnostics (rank 0, throttled)
                    if do_log and logs_emitted < self.max_logs_per_step:
                        # norms and angle
                        u2 = (u * u).sum().clamp_min(eps)
                        gr2 = g2  # already computed
                        un = torch.sqrt(u2).item()
                        gn = torch.sqrt(gr2).item()
                        # cosθ based on *pre-reflection* dot to report disagreement
                        # (use the stored 'dot' before reflection)
                        # Clamp for numerical safety:
                        denom = (torch.sqrt((u2 + eps) * (gr2 + eps)))
                        cos_theta = (dot / denom).item()
                        cos_theta = max(-1.0, min(1.0, float(cos_theta)))
                        theta_deg = math.degrees(math.acos(cos_theta))
                        print(f"[CautiousMuon step {self._step_idx}] param_idx={i} shape={tuple(p.shape)} "
                              f"dot={float(dot):+.3e} ||u||={un:.3e} ||g||={gn:.3e} "
                              f"cosθ={cos_theta:+.4f} θ={theta_deg:6.2f}° action={act}")
                        logs_emitted += 1

                    # serialize to flat buffer
                    updates_flat[curr_idx:curr_idx+p.numel()] = u.flatten()
                curr_idx += p.numel()

            # sync updates across devices
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # (Optional) simple per-step summary on rank 0
            if do_log and self.rank == 0 and n_params_local > 0:
                # We could all-reduce these counts for global stats; for minimal changes,
                # we just print this rank's tallies.
                refl_rate = (n_reflect_local / max(1, n_params_local)) * 100.0
                print(f"[CautiousMuon step {self._step_idx}] reflect_rate_local={refl_rate:.1f}% "
                      f"(reflected {n_reflect_local}/{n_params_local} local blocks)")

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                u = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(u, alpha=-lr)
                curr_idx += p.numel()

# # assume zeropower_backends is defined elsewhere, as in your original code
# class CautiousRowMuon(torch.optim.Optimizer):
#     def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
#                  backend='newtonschulz5', backend_steps=5,
#                  rank=0, world_size=1):
#         defaults = dict(
#             lr=lr, momentum=momentum, nesterov=nesterov,
#             backend=backend, backend_steps=backend_steps,
#             cautious_axis='row',      # 'row' | 'col' | 'head'
#             num_heads=None, head_dim=None,
#             agree_log=True,           # [AGREE-LOG] new: enable logging by default
#             agree_log_every=125       # [AGREE-LOG] new: print every k steps (default 125)
#         )
#         super().__init__(params, defaults)
#         self.rank = rank
#         self.world_size = world_size
#         self._step_idx = 0           # [AGREE-LOG] new: step counter

#     def step(self):
#         self._step_idx += 1          # [AGREE-LOG]
#         for group in self.param_groups:

#             lr = group['lr']
#             momentum = group['momentum']
#             zeropower_backend = zeropower_backends[group['backend']]
#             axis = group.get('cautious_axis', 'row')
#             num_heads = group.get('num_heads', None)
#             head_dim  = group.get('head_dim', None)

#             # [AGREE-LOG] decide if we log this step (rank 0 only)
#             do_agree_log = (
#                 group.get('agree_log', True)
#                 and self.rank == 0
#                 and (self._step_idx % max(1, int(group.get('agree_log_every', 125))) == 0)
#             )

#             total_params = sum(p.numel() for p in group['params'])
#             updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
#             curr_idx = 0

#             for i, p in enumerate(group['params']):
#                 if i % self.world_size == self.rank:
#                     g = p.grad
#                     if g is None:
#                         curr_idx += p.numel(); continue

#                     state = self.state[p]
#                     if 'momentum_buffer' not in state:
#                         state['momentum_buffer'] = torch.zeros_like(g)
#                     buf = state['momentum_buffer']
#                     buf.mul_(momentum).add_(g)
#                     if group['nesterov']:
#                         g = g.add(buf, alpha=momentum)

#                     g_raw = g.detach()  # keep for masking

#                     # Muon direction
#                     u = zeropower_backend(g, steps=group['backend_steps'])
#                     u = u * (max(1.0, u.size(0)/u.size(1))**0.5)

#                     # ---- cautious masking ----
#                     gr = g_raw.to(dtype=u.dtype, device=u.device)
#                     eps = torch.finfo(u.dtype).eps

#                     if axis == 'row':
#                         r = (u * gr).sum(dim=1)                  # [m]
#                         m = (r > 0).to(u.dtype).unsqueeze(1)     # [m,1]
#                         mbar = m.mean()
#                         # [AGREE-LOG] print fraction of agreeing rows
#                         if do_agree_log:
#                             frac = float(mbar.item())
#                             print(f"[CautiousMuon step {self._step_idx}] "
#                                   f"param_idx={i} shape={tuple(p.shape)} "
#                                   f"frac_rows_agree={frac:.3f} axis=row")
#                         u = (m * u) / (mbar + eps)

#                     elif axis == 'col':
#                         c = (u * gr).sum(dim=0)                  # [n]
#                         m = (c > 0).to(u.dtype).unsqueeze(0)     # [1,n]
#                         mbar = m.mean()
#                         # [AGREE-LOG] (optional) also report columns if you ever set axis='col'
#                         if do_agree_log:
#                             frac = float(mbar.item())
#                             print(f"[CautiousMuon step {self._step_idx}] "
#                                   f"param_idx={i} shape={tuple(p.shape)} "
#                                   f"frac_cols_agree={frac:.3f} axis=col")
#                         u = (m * u) / (mbar + eps)

#                     elif axis == 'head':
#                         assert num_heads is not None and head_dim is not None, \
#                             "head masking requires num_heads and head_dim"
#                         m_rows = u.size(0)
#                         assert m_rows == num_heads * head_dim, \
#                             "rows must equal num_heads * head_dim"
#                         r_per_row = (u * gr).sum(dim=1).view(num_heads, head_dim)  # [H,Dh]
#                         r_head = r_per_row.sum(dim=1)                              # [H]
#                         mh = (r_head > 0).to(u.dtype).view(num_heads, 1).repeat(1, head_dim)
#                         m = mh.reshape(m_rows, 1)
#                         mbar = m.mean()
#                         # [AGREE-LOG] fraction of agreeing heads
#                         if do_agree_log:
#                             frac_heads = float((r_head > 0).to(u.dtype).mean().item())
#                             print(f"[CautiousMuon step {self._step_idx}] "
#                                   f"param_idx={i} shape={tuple(p.shape)} "
#                                   f"frac_heads_agree={frac_heads:.3f} axis=head")
#                         u = (m * u) / (mbar + eps)
#                     # -------------------------

#                     updates_flat[curr_idx:curr_idx+p.numel()] = u.flatten()
#                 curr_idx += p.numel()

#             dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

#             curr_idx = 0
#             for p in group['params']:
#                 u = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
#                 p.data.add_(u, alpha=-lr)
#                 curr_idx += p.numel()


class ScheduleFreeMuon(torch.optim.Optimizer):
    """
    Schedule-Free Muon (with momentum)
      y_t = (1 - beta) * z_t + beta * x_t
      g_t = grad(loss, params=y_t)
      m_{t} = momentum * m_{t-1} + (1 - momentum) * g_t        # EMA of gradients
      g_eff = g_t + momentum * m_t   (if nesterov)  else m_t
      g_muon = MuonOrth(g_eff) * scale
      z_{t+1} = z_t - lr * g_muon
      x_{t+1} = (t/(t+1)) * x_t + (1/(t+1)) * z_{t+1}
      params  = y_{t+1} = (1 - beta) * z_{t+1} + beta * x_{t+1}

    Assumes all params here are 2D tensors. Route non-2D to another optimizer.
    """

    def __init__(self, params, lr=0.02, beta=0.95, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._t = 0  # global schedule-free step index

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p is None:
                        continue
                    state = self.state[p]
                    if 'z' not in state:
                        state['z'] = p.data.clone()
                    if 'x' not in state:
                        state['x'] = p.data.clone()
                    if 'm' not in state:
                        state['m'] = torch.zeros_like(p.data)  # momentum buffer (EMA of grads)
            # Initially, y_0 == z_0 == x_0 == p.data

    @torch.no_grad()
    def step(self):
        self._t += 1
        t = self._t

        for group in self.param_groups:
            lr        = group['lr']
            beta      = group['beta']
            mu        = group['momentum']
            nesterov  = group['nesterov']
            backend   = zeropower_backends[group['backend']]
            ns_steps  = group['backend_steps']

            # Flattened buffer for distributed all-reduce of Muon updates
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0

            # Build local updates (striped by rank)
            for i, p in enumerate(group['params']):
                g = p.grad
                if g is None:
                    curr_idx += p.numel()
                    continue
                if i % self.world_size != self.rank:
                    curr_idx += p.numel()
                    continue
                if g.ndim != 2:
                    raise RuntimeError("ScheduleFreeMuon expects 2D params only.")

                state = self.state[p]
                m = state['m']

                # momentum EMA of gradients at y_t
                # note: classic EMA form: m <- mu*m + (1-mu)*g
                m.mul_(mu).add_(g, alpha=(1.0 - mu))

                # Nesterov? use g + mu*m ; otherwise use m
                g_eff = g.add(m, alpha=mu) if nesterov else m

                # Muon orthogonalization backend on the effective direction
                g_muon = backend(g_eff, steps=ns_steps)

                # Same Muon scaling as baseline
                msz, nsz = g_muon.size()
                scale = max(1.0, msz / nsz) ** 0.5
                g_muon = g_muon * scale

                updates_flat[curr_idx:curr_idx + p.numel()] = g_muon.flatten().to(updates_flat.dtype)
                curr_idx += p.numel()

            # All-reduce across ranks
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Apply to z, average into x, set params to y for next step
            curr_idx = 0
            for p in group['params']:
                state = self.state[p]
                z = state['z']
                x = state['x']

                g_muon = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                curr_idx += p.numel()

                # z_{t+1}
                z.add_(g_muon, alpha=-lr)
                # x_{t+1}
                x.mul_(t / (t + 1.0)).add_(z, alpha=1.0 / (t + 1.0))
                # y_{t+1}
                p.data.copy_(z).mul_(1.0 - beta).add_(x, alpha=beta)

        return None

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


# muon_shampoo_polar.py

import torch
import torch.distributed as dist

# ------------------------------------------------------------
# Small, reusable helpers (GEMM-only)
# ------------------------------------------------------------

def power_spectral_norm_small(G: torch.Tensor, iters: int = 4) -> float:
    """
    Estimate λ_max(G) for a small square matrix G via power iteration.
    G is fp32 (recommended). Returns a Python float.
    """
    n = G.shape[0]
    v = torch.randn(n, device=G.device, dtype=G.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = G @ v
        v = v / (v.norm() + 1e-12)
    return float(v @ (G @ v))


def inv_sqrt_newton_schulz(G: torch.Tensor, steps: int = 5, power_iters: int = 4) -> torch.Tensor:
    """
    Compute G^{-1/2} (approx) using fixed-count Newton–Schulz on small SPD G (n×n).
    B_{k+1} = 0.5 * B_k * (3I - G * B_k^2), initialized with B0 = (λ_max(G))^{-1/2} I.
    All GEMMs; recommended in fp32.
    """
    n = G.shape[0]
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    lam_max = power_spectral_norm_small(G, iters=power_iters)
    B = I * (lam_max ** -0.5 if lam_max > 0 else 1.0)
    for _ in range(max(steps, 0)):
        B2 = B @ B
        B = 0.5 * (B @ (3.0 * I - G @ B2))
    return B


def build_right_gram(S: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """
    Right Gram: G = S^T S + eps I (n×n), returned in fp32.
    eps = eps_scale * trace(G)/n to stabilize tiny modes.
    """
    G = (S.transpose(0, 1) @ S).to(torch.float32)
    n = G.shape[0]
    eps = eps_scale * (torch.trace(G) / n + 1e-12)
    G = G + eps * torch.eye(n, device=G.device, dtype=G.dtype)
    return G


def build_left_gram(S: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """
    Left Gram: H = S S^T + eps I (m×m), returned in fp32.
    eps = eps_scale * trace(H)/m.
    """
    H = (S @ S.transpose(0, 1)).to(torch.float32)
    m = H.shape[0]
    eps = eps_scale * (torch.trace(H) / m + 1e-12)
    H = H + eps * torch.eye(m, device=H.device, dtype=H.dtype)
    return H


def allreduce_small_matrix(G: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    All-reduce a tiny (m×m or n×n) Gram across data-parallel ranks and average.
    No-op if world_size == 1.
    """
    if world_size > 1:
        dist.all_reduce(G, op=dist.ReduceOp.SUM)
        G = G / world_size
    return G


def refresh_inverse_sqrt_cache(
    state: dict,
    S_bf16: torch.Tensor,
    use_right: bool,
    inv_steps: int,
    inv_power_iters: int,
    inv_eps: float,
    world_size: int,
    step_idx: int,
) -> None:
    """
    Build (and cache) the small inverse square root:
      if use_right:  M ≈ (S^T S + eps I)^(-1/2)
      else:          M ≈ (S S^T + eps I)^(-1/2)
    Saves in state['inv_sqrt_cache'], state['use_right'], state['last_refresh_step'].
    """
    if use_right:
        G = build_right_gram(S_bf16, eps_scale=inv_eps)
    else:
        G = build_left_gram(S_bf16, eps_scale=inv_eps)

    G = allreduce_small_matrix(G, world_size=world_size)
    M = inv_sqrt_newton_schulz(G, steps=inv_steps, power_iters=inv_power_iters)  # fp32 small matrix

    state['inv_sqrt_cache'] = M.detach()
    state['use_right'] = use_right
    state['last_refresh_step'] = step_idx


def apply_cached_inverse_sqrt(S_bf16: torch.Tensor, state: dict) -> torch.Tensor:
    """
    Apply cached inverse-sqrt to get the polar direction:
      if use_right:  U = S * M
      else:          U = M * S
    M is fp32; cast to bf16 for the big GEMM.
    """
    assert 'inv_sqrt_cache' in state and 'use_right' in state, "Cache missing; call refresh first."
    M = state['inv_sqrt_cache'].to(dtype=S_bf16.dtype, device=S_bf16.device)
    if state['use_right']:
        return S_bf16 @ M
    else:
        return M @ S_bf16


def residual_to_cache(S_bf16: torch.Tensor, state: dict) -> torch.Tensor:
    """
    GEMM-only drift metric relative to cached direction Q = apply_cached_inverse_sqrt(S):
      r = ||S - Q(Q^T S)||_F / ||S||_F  (tall)
        or ||S - (S Q^T)Q||_F / ||S||_F (fat)
    """
    Q = apply_cached_inverse_sqrt(S_bf16, state)
    m, n = S_bf16.shape
    if m >= n:
        H = Q.transpose(0, 1) @ S_bf16     # n×n
        S_proj = Q @ H                     # m×n
    else:
        H = S_bf16 @ Q.transpose(0, 1)     # m×m
        S_proj = H @ Q                     # m×n
    num = torch.linalg.norm(S_bf16 - S_proj)
    den = torch.linalg.norm(S_bf16) + 1e-12
    return num / den


# ------------------------------------------------------------
# Optimizer (wires the helpers)
# ------------------------------------------------------------

class MuonShampooPolar(torch.optim.Optimizer):
    """
    Muon with Shampoo-style split polar:
      U ≈ S (S^T S + eps I)^(-1/2)   [tall]   or   U ≈ (S S^T + eps I)^(-1/2) S  [fat]

    Refresh the small inverse-sqrt every `refresh_period` steps; otherwise reuse it and do
    one skinny GEMM per param. No QR/SVD; only GEMMs.

    Args:
      refresh_period: int, rebuild small inverse-sqrt every k steps (default 10)
      inv_steps:      int, Newton–Schulz steps on the small Gram (default 5)
      inv_power_iters:int, power-iteration steps to scale the small Gram (default 4)
      inv_eps:        float, ridge factor (scaled by trace/n) (default 1e-6)
      drift_tol:      Optional[float], if set, early-refresh when residual > drift_tol
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 refresh_period=10, inv_steps=5, inv_power_iters=4, inv_eps=1e-6,
                 drift_tol: float | None = None,
                 rank=0, world_size=1):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            refresh_period=refresh_period, inv_steps=inv_steps,
            inv_power_iters=inv_power_iters, inv_eps=inv_eps,
            drift_tol=drift_tol
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._step_idx = 0

    def step(self):
        self._step_idx += 1

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            refresh_period = max(1, group['refresh_period'])
            inv_steps = group['inv_steps']
            inv_power_iters = group['inv_power_iters']
            inv_eps = group['inv_eps']
            drift_tol = group.get('drift_tol', None)

            do_refresh = (self._step_idx % refresh_period == 0)

            # Flatten updates to a single buffer (same pattern as your Muon)
            total_params = sum(p.numel() for p in group['params'])
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            updates_flat = torch.zeros(total_params, device=device, dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Simple round-robin sharding across ranks
                if i % getattr(self, "world_size", 1) != getattr(self, "rank", 0):
                    curr_idx += p.numel()
                    continue

                g = p.grad
                if g is None:
                    curr_idx += p.numel()
                    continue

                # Momentum buffer (same as Muon)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                S = g.add(buf, alpha=momentum) if nesterov else buf

                # Work in bf16 for big GEMMs
                S_bf16 = S.detach().to(dtype=torch.bfloat16, device=device)
                m, n = S_bf16.shape
                use_right = (m >= n)

                need_build = do_refresh or ('inv_sqrt_cache' not in state) or ('use_right' not in state)

                # Optional early refresh if drift is large
                if (not need_build) and (drift_tol is not None):
                    r = residual_to_cache(S_bf16, state)
                    if r.item() > drift_tol:
                        need_build = True

                if need_build:
                    refresh_inverse_sqrt_cache(
                        state=state,
                        S_bf16=S_bf16,
                        use_right=use_right,
                        inv_steps=inv_steps,
                        inv_power_iters=inv_power_iters,
                        inv_eps=inv_eps,
                        world_size=self.world_size,
                        step_idx=self._step_idx,
                    )

                # Apply cached small inverse-sqrt (cheap path)
                U = apply_cached_inverse_sqrt(S_bf16, state)

                # Optional: same Muon shape scaling
                U = U * (max(1.0, m / n) ** 0.5)

                updates_flat[curr_idx:curr_idx + p.numel()] = U.flatten()
                curr_idx += p.numel()

            # All-reduce the updates (unchanged)
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Deserialize and apply
            curr_idx = 0
            for p in group['params']:
                upd = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data)
                p.data.add_(upd.to(dtype=p.data.dtype, device=p.data.device), alpha=-lr)
                curr_idx += p.numel()
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

class MuonEMA(torch.optim.Optimizer):  # [EMA] renamed class from `Muon` -> `MuonEMA`
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz, with EMA evaluation weights (minimal changes).

    Minimal additions:
      - `ema_beta` hyperparameter (default 0.999).
      - Maintains per-param `state['ema']` and `state['step']`.
      - `eval()` / `train()` methods to swap to bias-corrected EMA weights and back.
    """

    def __init__(self, params, lr = 3e-4, momentum = 0.95, nesterov = True,
                 backend = 'newtonschulz5', backend_steps = 5,
                 ema_beta = 0.999,  # [EMA] new arg with default 0.999
                 rank = 0, world_size = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps,
                        ema_beta=ema_beta,            # [EMA] stored in group
                        train_mode=True)              # [EMA] track swap state
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def __setstate__(self, state):
        super().__setstate__(state)
        # [EMA] ensure `step` is a tensor after load (no other changes)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is not None and "step" in st and not torch.is_tensor(st["step"]):
                    st["step"] = torch.tensor(float(st["step"]), device=p.device)

    @torch.no_grad()
    def eval(self):  # [EMA] NEW: swap to bias-corrected EMA weights for evaluation/checkpointing
        for group in self.param_groups:
            if group.get('train_mode', True):
                ema_beta = group['ema_beta']
                for p in group['params']:
                    st = self.state.get(p)
                    if st is None:
                        continue
                    # buffer live params for restoration
                    st['param_buffer'] = p.detach().clone()
                    # lazy-init ema/step
                    if 'ema' not in st:
                        st['ema'] = p.detach().clone()
                    if 'step' not in st:
                        st['step'] = torch.tensor(0.0, device=p.device)
                    # bias correction
                    step_t = float(st['step'].item()) if torch.is_tensor(st['step']) else float(st['step'])
                    denom = 1.0 - (ema_beta ** max(step_t, 1.0))
                    if denom <= 0.0:
                        denom = 1.0
                    p.copy_(st['ema'] / denom)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):  # [EMA] NEW: restore live weights to continue training
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
            ema_beta = group['ema_beta']        # [EMA] read EMA rate
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
                        if 'step' not in state:                          # [EMA] init step
                            state['step'] = torch.tensor(0.0, device=p.device)
                        if 'ema' not in state:                           # [EMA] init ema
                            state['ema'] = p.detach().clone()

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

            # deserialize and apply updates (unchanged except EMA lines below)
            curr_idx = 0
            for p in params_list:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                # [EMA] per-param: increment step and update EMA of *live* weights
                st = self.state[p]
                if 'step' not in st:
                    st['step'] = torch.tensor(0.0, device=p.device)
                st['step'] += 1.0
                if 'ema' not in st:
                    st['ema'] = p.detach().clone()
                st['ema'].lerp_(p, 1.0 - ema_beta)  # ema = ema_beta*ema + (1-ema_beta)*p
                # [/EMA]
                curr_idx += p.numel()


import math

# assume zeropower_backends is defined elsewhere, as in your original code

import math
import torch
import torch.distributed as dist

# Assumes you already define: zeropower_backends = {"newtonschulz5": ... , ...}

class CautiousRowMuon(torch.optim.Optimizer):
    """
    Muon with cautious masking against the *instantaneous* gradient (pre-momentum).

    Per 2D parameter:
      1) g_inst = p.grad (detached)           # capture BEFORE momentum/Nesterov
      2) Apply momentum/Nesterov to get g     # as usual
      3) u = zeropower_backend(g); Muon scale
      4) Compare u vs g_inst along the chosen axis using cosine with a small margin:
         - 'row' (default): mask rows with cos_row <= margin
         - 'col':          mask columns with cos_col <= margin
         - 'head':         mask heads with cos_head <= margin  (requires num_heads, head_dim)
         Then divide by mask mean (paper's scaling) to preserve RMS.
      5) All-reduce and apply LR. Weight decay (if any) should be decoupled outside.

    New param-group options (all optional):
      - cautious_axis: 'row' | 'col' | 'head'   (default 'row')
      - cautious_margin: float cosine margin    (default 0.01)
      - num_heads, head_dim: required for 'head'
      - agree_log: bool to print fractions      (default True)
      - agree_log_every: int steps              (default 125)
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            backend=backend, backend_steps=backend_steps,
            cautious_axis='row',
            cautious_margin=0.01,
            num_heads=None, head_dim=None,
            agree_log=True,
            agree_log_every=125
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._step_idx = 0

    def step(self):
        self._step_idx += 1

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            axis = group.get('cautious_axis', 'row')
            margin = float(group.get('cautious_margin', 0.01))
            num_heads = group.get('num_heads', None)
            head_dim  = group.get('head_dim', None)

            do_agree_log = (
                bool(group.get('agree_log', True))
                and self.rank == 0
                and (self._step_idx % max(1, int(group.get('agree_log_every', 125))) == 0)
            )

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0

            for i, p in enumerate(group['params']):
                if i % self.world_size != self.rank:
                    curr_idx += p.numel()
                    continue

                g = p.grad
                if g is None:
                    curr_idx += p.numel()
                    continue

                # --- capture instantaneous gradient BEFORE momentum/Nesterov ---
                g_inst = g.detach()

                # momentum + optional Nesterov (unchanged)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)      # SGD momentum on current grad
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                # Muon orthogonalization backend (e.g., Newton–Schulz)
                u = zeropower_backend(g, steps=group['backend_steps'])
                # Muon’s size scaling to match RMS across aspect ratios
                u = u * (max(1.0, u.size(0) / u.size(1)) ** 0.5)

                # ---- cautious masking vs instantaneous gradient (pre-momentum) ----
                # match dtype/device for numerical stability (bf16)
                gr = g_inst.to(dtype=u.dtype, device=u.device)
                eps = torch.finfo(u.dtype).eps

                if axis == 'row':
                    # row agreements: cosine per row
                    r = (u * gr).sum(dim=1)                       # [m]
                    u_row_n = (u * u).sum(dim=1).sqrt() + eps     # [m]
                    g_row_n = (gr * gr).sum(dim=1).sqrt() + eps   # [m]
                    cos_row = r / (u_row_n * g_row_n)
                    m = (cos_row > margin).to(u.dtype).unsqueeze(1)   # [m,1]
                    mbar = m.mean()
                    # optional logging
                    if do_agree_log:
                        frac = float(mbar.item())
                        print(f"[CautiousMuon step {self._step_idx}] "
                              f"param_idx={i} shape={tuple(p.shape)} "
                              f"frac_rows_agree={frac:.3f} axis=row")
                    # divide-by-mean scaling (paper)
                    u = (m * u) / (mbar + eps)

                elif axis == 'col':
                    # column agreements: cosine per column
                    c = (u * gr).sum(dim=0)                       # [n]
                    u_col_n = (u * u).sum(dim=0).sqrt() + eps     # [n]
                    g_col_n = (gr * gr).sum(dim=0).sqrt() + eps   # [n]
                    cos_col = c / (u_col_n * g_col_n)
                    m = (cos_col > margin).to(u.dtype).unsqueeze(0)   # [1,n]
                    mbar = m.mean()
                    if do_agree_log:
                        frac = float(mbar.item())
                        print(f"[CautiousMuon step {self._step_idx}] "
                              f"param_idx={i} shape={tuple(p.shape)} "
                              f"frac_cols_agree={frac:.3f} axis=col")
                    u = (m * u) / (mbar + eps)

                elif axis == 'head':
                    # head agreements: group consecutive rows per head
                    assert num_heads is not None and head_dim is not None, \
                        "head masking requires num_heads and head_dim"
                    m_rows = u.size(0)
                    assert m_rows == num_heads * head_dim, \
                        f"rows ({m_rows}) must equal num_heads*head_dim ({num_heads*head_dim})"
                    # flatten rows within each head, compute cosine per head
                    u_rows = u.view(num_heads, head_dim, -1)      # [H, Dh, n]
                    g_rows = gr.view(num_heads, head_dim, -1)
                    # dot and norms per head
                    dot_h = (u_rows * g_rows).sum(dim=(1,2))      # [H]
                    u_h_n = (u_rows * u_rows).sum(dim=(1,2)).sqrt() + eps
                    g_h_n = (g_rows * g_rows).sum(dim=(1,2)).sqrt() + eps
                    cos_h = dot_h / (u_h_n * g_h_n)               # [H]
                    mh = (cos_h > margin).to(u.dtype).view(num_heads, 1).repeat(1, head_dim)  # [H,Dh]
                    m = mh.reshape(m_rows, 1)                     # broadcast to rows
                    mbar = m.mean()
                    if do_agree_log:
                        frac_heads = float((cos_h > margin).to(u.dtype).mean().item())
                        print(f"[CautiousMuon step {self._step_idx}] "
                              f"param_idx={i} shape={tuple(p.shape)} "
                              f"frac_heads_agree={frac_heads:.3f} axis=head")
                    u = (m * u) / (mbar + eps)

                else:
                    # unknown axis: no cautious mask
                    pass
                # -------------------------------------------------------------------

                # serialize to flat buffer
                updates_flat[curr_idx:curr_idx + p.numel()] = u.flatten()
                curr_idx += p.numel()

            # sync updates across devices (simple deserialization)
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                u = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(u, alpha=-lr)
                curr_idx += p.numel()


class CautiousMuon(torch.optim.Optimizer):
    """
    Muon + Cautious cone-projection (reflect variant) with optional diagnostics.

    Diagnostics (when enabled):
      - prints inner product <u, g_raw>, norms, cosθ, θ(deg), and action (reflect/keep)
      - only prints on rank==0
      - prints at most `max_logs_per_step` entries per step, every `log_every` steps

    Args (new):
      cautious_log (bool): enable/disable diagnostics (default: True)
      log_every (int): print every N steps (default: 125)
      max_logs_per_step (int): cap number of per-param logs per step (default: 8)
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1,
                 cautious_log=True, log_every=125, max_logs_per_step=8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self.cautious_log = cautious_log
        self.log_every = log_every
        self.max_logs_per_step = max_logs_per_step
        self._step_idx = 0

    def step(self):
        self._step_idx += 1
        do_log = (self.cautious_log and self.rank == 0 and
                  (self._step_idx % max(1, self.log_every) == 0))

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0

            # local counters for simple per-step stats
            n_params_local = 0
            n_reflect_local = 0
            logs_emitted = 0

            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        curr_idx += p.numel()
                        continue

                    # keep a copy of the raw gradient for the cone test (match dtype/device later)
                    g_raw = g.detach()

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)

                    # Muon orthogonalization backend (e.g., Newton–Schulz)
                    u = zeropower_backend(g, steps=group['backend_steps'])
                    # Muon’s size scaling to match RMS across aspect ratios
                    u = u * (max(1.0, u.size(0) / u.size(1)) ** 0.5)

                    # ----- Cautious cone-projection (reflect) -----
                    gr = g_raw.to(dtype=u.dtype, device=u.device)
                    dot = (u * gr).sum()
                    eps = torch.finfo(u.dtype).eps
                    g2  = (gr * gr).sum().clamp_min(eps)
                    act = "keep"
                    if dot < 0:
                        # reflect along g_raw: u' = u - 2 * proj_{g_raw}(u)
                        u = u - 2.0 * (dot / g2) * gr
                        act = "reflect"
                        n_reflect_local += 1
                    n_params_local += 1
                    # ---------------------------------------------

                    # Diagnostics (rank 0, throttled)
                    if do_log and logs_emitted < self.max_logs_per_step:
                        # norms and angle
                        u2 = (u * u).sum().clamp_min(eps)
                        gr2 = g2  # already computed
                        un = torch.sqrt(u2).item()
                        gn = torch.sqrt(gr2).item()
                        # cosθ based on *pre-reflection* dot to report disagreement
                        # (use the stored 'dot' before reflection)
                        # Clamp for numerical safety:
                        denom = (torch.sqrt((u2 + eps) * (gr2 + eps)))
                        cos_theta = (dot / denom).item()
                        cos_theta = max(-1.0, min(1.0, float(cos_theta)))
                        theta_deg = math.degrees(math.acos(cos_theta))
                        print(f"[CautiousMuon step {self._step_idx}] param_idx={i} shape={tuple(p.shape)} "
                              f"dot={float(dot):+.3e} ||u||={un:.3e} ||g||={gn:.3e} "
                              f"cosθ={cos_theta:+.4f} θ={theta_deg:6.2f}° action={act}")
                        logs_emitted += 1

                    # serialize to flat buffer
                    updates_flat[curr_idx:curr_idx+p.numel()] = u.flatten()
                curr_idx += p.numel()

            # sync updates across devices
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # (Optional) simple per-step summary on rank 0
            if do_log and self.rank == 0 and n_params_local > 0:
                # We could all-reduce these counts for global stats; for minimal changes,
                # we just print this rank's tallies.
                refl_rate = (n_reflect_local / max(1, n_params_local)) * 100.0
                print(f"[CautiousMuon step {self._step_idx}] reflect_rate_local={refl_rate:.1f}% "
                      f"(reflected {n_reflect_local}/{n_params_local} local blocks)")

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                u = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(u, alpha=-lr)
                curr_idx += p.numel()

# # assume zeropower_backends is defined elsewhere, as in your original code
# class CautiousRowMuon(torch.optim.Optimizer):
#     def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
#                  backend='newtonschulz5', backend_steps=5,
#                  rank=0, world_size=1):
#         defaults = dict(
#             lr=lr, momentum=momentum, nesterov=nesterov,
#             backend=backend, backend_steps=backend_steps,
#             cautious_axis='row',      # 'row' | 'col' | 'head'
#             num_heads=None, head_dim=None,
#             agree_log=True,           # [AGREE-LOG] new: enable logging by default
#             agree_log_every=125       # [AGREE-LOG] new: print every k steps (default 125)
#         )
#         super().__init__(params, defaults)
#         self.rank = rank
#         self.world_size = world_size
#         self._step_idx = 0           # [AGREE-LOG] new: step counter

#     def step(self):
#         self._step_idx += 1          # [AGREE-LOG]
#         for group in self.param_groups:

#             lr = group['lr']
#             momentum = group['momentum']
#             zeropower_backend = zeropower_backends[group['backend']]
#             axis = group.get('cautious_axis', 'row')
#             num_heads = group.get('num_heads', None)
#             head_dim  = group.get('head_dim', None)

#             # [AGREE-LOG] decide if we log this step (rank 0 only)
#             do_agree_log = (
#                 group.get('agree_log', True)
#                 and self.rank == 0
#                 and (self._step_idx % max(1, int(group.get('agree_log_every', 125))) == 0)
#             )

#             total_params = sum(p.numel() for p in group['params'])
#             updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
#             curr_idx = 0

#             for i, p in enumerate(group['params']):
#                 if i % self.world_size == self.rank:
#                     g = p.grad
#                     if g is None:
#                         curr_idx += p.numel(); continue

#                     state = self.state[p]
#                     if 'momentum_buffer' not in state:
#                         state['momentum_buffer'] = torch.zeros_like(g)
#                     buf = state['momentum_buffer']
#                     buf.mul_(momentum).add_(g)
#                     if group['nesterov']:
#                         g = g.add(buf, alpha=momentum)

#                     g_raw = g.detach()  # keep for masking

#                     # Muon direction
#                     u = zeropower_backend(g, steps=group['backend_steps'])
#                     u = u * (max(1.0, u.size(0)/u.size(1))**0.5)

#                     # ---- cautious masking ----
#                     gr = g_raw.to(dtype=u.dtype, device=u.device)
#                     eps = torch.finfo(u.dtype).eps

#                     if axis == 'row':
#                         r = (u * gr).sum(dim=1)                  # [m]
#                         m = (r > 0).to(u.dtype).unsqueeze(1)     # [m,1]
#                         mbar = m.mean()
#                         # [AGREE-LOG] print fraction of agreeing rows
#                         if do_agree_log:
#                             frac = float(mbar.item())
#                             print(f"[CautiousMuon step {self._step_idx}] "
#                                   f"param_idx={i} shape={tuple(p.shape)} "
#                                   f"frac_rows_agree={frac:.3f} axis=row")
#                         u = (m * u) / (mbar + eps)

#                     elif axis == 'col':
#                         c = (u * gr).sum(dim=0)                  # [n]
#                         m = (c > 0).to(u.dtype).unsqueeze(0)     # [1,n]
#                         mbar = m.mean()
#                         # [AGREE-LOG] (optional) also report columns if you ever set axis='col'
#                         if do_agree_log:
#                             frac = float(mbar.item())
#                             print(f"[CautiousMuon step {self._step_idx}] "
#                                   f"param_idx={i} shape={tuple(p.shape)} "
#                                   f"frac_cols_agree={frac:.3f} axis=col")
#                         u = (m * u) / (mbar + eps)

#                     elif axis == 'head':
#                         assert num_heads is not None and head_dim is not None, \
#                             "head masking requires num_heads and head_dim"
#                         m_rows = u.size(0)
#                         assert m_rows == num_heads * head_dim, \
#                             "rows must equal num_heads * head_dim"
#                         r_per_row = (u * gr).sum(dim=1).view(num_heads, head_dim)  # [H,Dh]
#                         r_head = r_per_row.sum(dim=1)                              # [H]
#                         mh = (r_head > 0).to(u.dtype).view(num_heads, 1).repeat(1, head_dim)
#                         m = mh.reshape(m_rows, 1)
#                         mbar = m.mean()
#                         # [AGREE-LOG] fraction of agreeing heads
#                         if do_agree_log:
#                             frac_heads = float((r_head > 0).to(u.dtype).mean().item())
#                             print(f"[CautiousMuon step {self._step_idx}] "
#                                   f"param_idx={i} shape={tuple(p.shape)} "
#                                   f"frac_heads_agree={frac_heads:.3f} axis=head")
#                         u = (m * u) / (mbar + eps)
#                     # -------------------------

#                     updates_flat[curr_idx:curr_idx+p.numel()] = u.flatten()
#                 curr_idx += p.numel()

#             dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

#             curr_idx = 0
#             for p in group['params']:
#                 u = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
#                 p.data.add_(u, alpha=-lr)
#                 curr_idx += p.numel()



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


# muon_shampoo_polar.py

import torch
import torch.distributed as dist

# ------------------------------------------------------------
# Small, reusable helpers (GEMM-only)
# ------------------------------------------------------------

def power_spectral_norm_small(G: torch.Tensor, iters: int = 4) -> float:
    """
    Estimate λ_max(G) for a small square matrix G via power iteration.
    G is fp32 (recommended). Returns a Python float.
    """
    n = G.shape[0]
    v = torch.randn(n, device=G.device, dtype=G.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = G @ v
        v = v / (v.norm() + 1e-12)
    return float(v @ (G @ v))


def inv_sqrt_newton_schulz(G: torch.Tensor, steps: int = 5, power_iters: int = 4) -> torch.Tensor:
    """
    Compute G^{-1/2} (approx) using fixed-count Newton–Schulz on small SPD G (n×n).
    B_{k+1} = 0.5 * B_k * (3I - G * B_k^2), initialized with B0 = (λ_max(G))^{-1/2} I.
    All GEMMs; recommended in fp32.
    """
    n = G.shape[0]
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    lam_max = power_spectral_norm_small(G, iters=power_iters)
    B = I * (lam_max ** -0.5 if lam_max > 0 else 1.0)
    for _ in range(max(steps, 0)):
        B2 = B @ B
        B = 0.5 * (B @ (3.0 * I - G @ B2))
    return B


def build_right_gram(S: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """
    Right Gram: G = S^T S + eps I (n×n), returned in fp32.
    eps = eps_scale * trace(G)/n to stabilize tiny modes.
    """
    G = (S.transpose(0, 1) @ S).to(torch.float32)
    n = G.shape[0]
    eps = eps_scale * (torch.trace(G) / n + 1e-12)
    G = G + eps * torch.eye(n, device=G.device, dtype=G.dtype)
    return G


def build_left_gram(S: torch.Tensor, eps_scale: float) -> torch.Tensor:
    """
    Left Gram: H = S S^T + eps I (m×m), returned in fp32.
    eps = eps_scale * trace(H)/m.
    """
    H = (S @ S.transpose(0, 1)).to(torch.float32)
    m = H.shape[0]
    eps = eps_scale * (torch.trace(H) / m + 1e-12)
    H = H + eps * torch.eye(m, device=H.device, dtype=H.dtype)
    return H


def allreduce_small_matrix(G: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    All-reduce a tiny (m×m or n×n) Gram across data-parallel ranks and average.
    No-op if world_size == 1.
    """
    if world_size > 1:
        dist.all_reduce(G, op=dist.ReduceOp.SUM)
        G = G / world_size
    return G


def refresh_inverse_sqrt_cache(
    state: dict,
    S_bf16: torch.Tensor,
    use_right: bool,
    inv_steps: int,
    inv_power_iters: int,
    inv_eps: float,
    world_size: int,
    step_idx: int,
) -> None:
    """
    Build (and cache) the small inverse square root:
      if use_right:  M ≈ (S^T S + eps I)^(-1/2)
      else:          M ≈ (S S^T + eps I)^(-1/2)
    Saves in state['inv_sqrt_cache'], state['use_right'], state['last_refresh_step'].
    """
    if use_right:
        G = build_right_gram(S_bf16, eps_scale=inv_eps)
    else:
        G = build_left_gram(S_bf16, eps_scale=inv_eps)

    G = allreduce_small_matrix(G, world_size=world_size)
    M = inv_sqrt_newton_schulz(G, steps=inv_steps, power_iters=inv_power_iters)  # fp32 small matrix

    state['inv_sqrt_cache'] = M.detach()
    state['use_right'] = use_right
    state['last_refresh_step'] = step_idx


def apply_cached_inverse_sqrt(S_bf16: torch.Tensor, state: dict) -> torch.Tensor:
    """
    Apply cached inverse-sqrt to get the polar direction:
      if use_right:  U = S * M
      else:          U = M * S
    M is fp32; cast to bf16 for the big GEMM.
    """
    assert 'inv_sqrt_cache' in state and 'use_right' in state, "Cache missing; call refresh first."
    M = state['inv_sqrt_cache'].to(dtype=S_bf16.dtype, device=S_bf16.device)
    if state['use_right']:
        return S_bf16 @ M
    else:
        return M @ S_bf16


def residual_to_cache(S_bf16: torch.Tensor, state: dict) -> torch.Tensor:
    """
    GEMM-only drift metric relative to cached direction Q = apply_cached_inverse_sqrt(S):
      r = ||S - Q(Q^T S)||_F / ||S||_F  (tall)
        or ||S - (S Q^T)Q||_F / ||S||_F (fat)
    """
    Q = apply_cached_inverse_sqrt(S_bf16, state)
    m, n = S_bf16.shape
    if m >= n:
        H = Q.transpose(0, 1) @ S_bf16     # n×n
        S_proj = Q @ H                     # m×n
    else:
        H = S_bf16 @ Q.transpose(0, 1)     # m×m
        S_proj = H @ Q                     # m×n
    num = torch.linalg.norm(S_bf16 - S_proj)
    den = torch.linalg.norm(S_bf16) + 1e-12
    return num / den


# ------------------------------------------------------------
# Optimizer (wires the helpers)
# ------------------------------------------------------------

class MuonShampooPolar(torch.optim.Optimizer):
    """
    Muon with Shampoo-style split polar:
      U ≈ S (S^T S + eps I)^(-1/2)   [tall]   or   U ≈ (S S^T + eps I)^(-1/2) S  [fat]

    Refresh the small inverse-sqrt every `refresh_period` steps; otherwise reuse it and do
    one skinny GEMM per param. No QR/SVD; only GEMMs.

    Args:
      refresh_period: int, rebuild small inverse-sqrt every k steps (default 10)
      inv_steps:      int, Newton–Schulz steps on the small Gram (default 5)
      inv_power_iters:int, power-iteration steps to scale the small Gram (default 4)
      inv_eps:        float, ridge factor (scaled by trace/n) (default 1e-6)
      drift_tol:      Optional[float], if set, early-refresh when residual > drift_tol
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 refresh_period=10, inv_steps=5, inv_power_iters=4, inv_eps=1e-6,
                 drift_tol: float | None = None,
                 rank=0, world_size=1):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            refresh_period=refresh_period, inv_steps=inv_steps,
            inv_power_iters=inv_power_iters, inv_eps=inv_eps,
            drift_tol=drift_tol
        )
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._step_idx = 0

    def step(self):
        self._step_idx += 1

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            refresh_period = max(1, group['refresh_period'])
            inv_steps = group['inv_steps']
            inv_power_iters = group['inv_power_iters']
            inv_eps = group['inv_eps']
            drift_tol = group.get('drift_tol', None)

            do_refresh = (self._step_idx % refresh_period == 0)

            # Flatten updates to a single buffer (same pattern as your Muon)
            total_params = sum(p.numel() for p in group['params'])
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            updates_flat = torch.zeros(total_params, device=device, dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                # Simple round-robin sharding across ranks
                if i % getattr(self, "world_size", 1) != getattr(self, "rank", 0):
                    curr_idx += p.numel()
                    continue

                g = p.grad
                if g is None:
                    curr_idx += p.numel()
                    continue

                # Momentum buffer (same as Muon)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                S = g.add(buf, alpha=momentum) if nesterov else buf

                # Work in bf16 for big GEMMs
                S_bf16 = S.detach().to(dtype=torch.bfloat16, device=device)
                m, n = S_bf16.shape
                use_right = (m >= n)

                need_build = do_refresh or ('inv_sqrt_cache' not in state) or ('use_right' not in state)

                # Optional early refresh if drift is large
                if (not need_build) and (drift_tol is not None):
                    r = residual_to_cache(S_bf16, state)
                    if r.item() > drift_tol:
                        need_build = True

                if need_build:
                    refresh_inverse_sqrt_cache(
                        state=state,
                        S_bf16=S_bf16,
                        use_right=use_right,
                        inv_steps=inv_steps,
                        inv_power_iters=inv_power_iters,
                        inv_eps=inv_eps,
                        world_size=self.world_size,
                        step_idx=self._step_idx,
                    )

                # Apply cached small inverse-sqrt (cheap path)
                U = apply_cached_inverse_sqrt(S_bf16, state)

                # Optional: same Muon shape scaling
                U = U * (max(1.0, m / n) ** 0.5)

                updates_flat[curr_idx:curr_idx + p.numel()] = U.flatten()
                curr_idx += p.numel()

            # All-reduce the updates (unchanged)
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # Deserialize and apply
            curr_idx = 0
            for p in group['params']:
                upd = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data)
                p.data.add_(upd.to(dtype=p.data.dtype, device=p.data.device), alpha=-lr)
                curr_idx += p.numel()
