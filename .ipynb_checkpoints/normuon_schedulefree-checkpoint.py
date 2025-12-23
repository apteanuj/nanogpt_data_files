# nor_muon_schedulefree.py
# Schedule-Free NorMuon (based on the same scaffolding as SGDScheduleFree / AdamWScheduleFree)
#
# NOTE:
#   - Call optimizer.train() before training (before any .step()).
#   - Call optimizer.eval() before validation / checkpointing (so weights are mapped to the averaged iterate).
#
# This implementation follows NorMuon Algorithm 1 for the base direction:
#   M_t = β1 M_{t-1} + (1-β1) G_t
#   O_t = NS5(M_t)
#   v_t = β2 v_{t-1} + (1-β2) mean_cols(O_t ⊙ O_t)   (row-wise mean of squares)
#   Obar = O_t ⊘ (sqrt(ExpandRows(v_t)) + eps)
#   eta_hat = eta_scale * eta * sqrt(mn) / ||Obar||_F        (eta_scale default 0.2)
#   W <- W - eta*wd*W - eta_hat*Obar
#
# Then wraps that base step in the schedule-free (iterate-averaging) machinery.

from typing import Union, Optional, Iterable, Dict, Callable, Any, Tuple
from typing_extensions import TypeAlias

import math
import torch
import torch.optim

try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


# -----------------------------------------------------------------------------
# Zeropower / orthogonalization backends (same as your Muon code)

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton–Schulz quintic iteration for orthogonalization. Runs in bf16.
    Returns an "approximately orthogonal" matrix.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    for _ in range(int(steps)):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if transposed:
        X = X.T
    return X

zeropower_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5
)


# -----------------------------------------------------------------------------
# Schedule-Free NorMuon

class NorMuonScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free NorMuon

    Schedule-free wrapper (iterate averaging) + NorMuon direction.

    Args:
        params: iterable of parameters (expects 2D matrices)
        lr: base learning rate "eta" (used for warmup and for weight decay scale)
        betas: (beta1, beta2) where
               beta1 = first-moment EMA for M_t (and schedule-free mixing parameter)
               beta2 = second-moment EMA for v_t (row-wise)
        eps: epsilon for normalization denom (perturbation parameter)
        weight_decay: lambda (applied as W <- W - eta*lambda*W) each step
        warmup_steps: linear warmup steps (internal, schedule-free style)
        r: polynomial weighting power for schedule-free averaging
        weight_lr_power: power of lr_max for schedule-free weighting
        backend: 'svd' or 'newtonschulz5'
        backend_steps: iterations for the backend (if iterative)
        eta_scale: scale factor in eta_hat formula (default 0.2 per NorMuon pseudocode)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 0.02,
        betas: Tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
        eta_scale: float = 0.2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 < betas[0] < 1.0):
            raise ValueError(f"beta1 must be in (0,1), got {betas[0]}")
        if not (0.0 < betas[1] < 1.0):
            raise ValueError(f"beta2 must be in (0,1), got {betas[1]}")
        if backend not in zeropower_backends:
            raise ValueError(f"Unknown backend: {backend}")
        if backend_steps <= 0:
            raise ValueError(f"backend_steps must be positive, got {backend_steps}")
        if eta_scale <= 0.0:
            raise ValueError(f"eta_scale must be positive, got {eta_scale}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,

            # schedule-free machinery
            r=r,
            k=0,
            warmup_steps=int(warmup_steps),
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,         # for logging: we'll set to mean(eta_hat) each step
            base_scheduled_lr=0.0,    # for debugging: eta after warmup (used for weights/decay)
            weight_lr_power=weight_lr_power,

            # NorMuon backend config
            backend=backend,
            backend_steps=int(backend_steps),
            eta_scale=eta_scale,
        )
        super().__init__(params, defaults)

    # ----------------- mode switches (same pattern as SGDScheduleFree / AdamWScheduleFree) -----------------

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to x (averaged)
                        p.lerp_(end=state["z"].to(p.device), weight=1.0 - 1.0 / beta1)
                        # p.lerp_(end=state["z"].to(p.device), weight=1.0 - beta1) # JLK, setting p to y
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to y (interpolation point)
                        p.lerp_(end=state["z"].to(p.device), weight=1.0 - beta1)
                group["train_mode"] = True

    # ----------------- main step -----------------

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step() was called. "
                "Call optimizer.train() before training and optimizer.eval() "
                "before evaluation/checkpointing."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            base_lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            k = int(group["k"])
            warmup_steps = int(group["warmup_steps"])
            r = float(group["r"])
            wlp = float(group["weight_lr_power"])
            eta_scale = float(group["eta_scale"])

            # --- schedule-free warmup for eta (the "eta" in NorMuon pseudocode) ---
            if warmup_steps > 0 and k < warmup_steps:
                sched = (k + 1) / float(warmup_steps)
            else:
                sched = 1.0
            eta = float(base_lr) * float(sched)   # eta after warmup
            group["base_scheduled_lr"] = eta

            # --- schedule-free averaging weights (we use eta, not eta_hat, to keep wrapper scalar) ---
            lr_max = group["lr_max"] = max(float(group["lr_max"]), eta)
            weight = ((k + 1) ** r) * (lr_max ** wlp)
            weight_sum = group["weight_sum"] = float(group["weight_sum"]) + float(weight)
            ckp1 = float(weight) / float(weight_sum) if weight_sum > 0.0 else 0.0

            zeropower_backend = zeropower_backends[group["backend"]]
            backend_steps = int(group["backend_steps"])

            eta_hat_sum = 0.0
            eta_hat_cnt = 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"NorMuonScheduleFree expects 2D params only; got shape={tuple(p.shape)}"
                    )

                grad = p.grad
                state = self.state[p]

                # init schedule-free fast iterate
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)

                # init NorMuon moments
                if "M" not in state:
                    state["M"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if "v" not in state:
                    # row-wise vector, keep in fp32 for stability
                    m = p.shape[0]
                    state["v"] = torch.zeros((m,), device=p.device, dtype=torch.float32)

                z = state["z"]
                M = state["M"]
                v = state["v"]

                # --- NorMuon direction ---
                # M_t = beta1 * M + (1-beta1) * G
                M.mul_(beta1).add_(grad, alpha=(1.0 - beta1))

                # O_t = NS5(M_t)
                O = zeropower_backend(M, steps=backend_steps)
                O = O.to(dtype=p.dtype)

                # v_t = beta2*v + (1-beta2)*mean_cols(O^2)  (row-wise mean squares)
                row_ms = (O * O).mean(dim=1).to(dtype=torch.float32)   # shape (m,)
                v.mul_(beta2).add_(row_ms, alpha=(1.0 - beta2))

                # Obar = O / (sqrt(v) + eps) with row-wise broadcasting
                denom = (v.sqrt() + float(eps)).to(dtype=O.dtype).unsqueeze(1)  # (m,1)
                Obar = O / denom

                # eta_hat = eta_scale * eta * sqrt(mn) / ||Obar||_F
                m, n = p.shape
                fro = torch.linalg.norm(Obar.float(), ord="fro")
                fro_val = float(fro.item())
                eta_hat = eta_scale * eta * math.sqrt(float(m * n)) / max(1e-12, fro_val)

                eta_hat_sum += eta_hat
                eta_hat_cnt += 1

                # --- Apply NorMuon weight decay term: W <- W - eta*wd*W
                # We apply it to both y (p) and z for consistency with schedule-free mapping.
                if wd != 0.0 and eta != 0.0:
                    decay = 1.0 - eta * wd
                    p.mul_(decay)
                    z.mul_(decay)

                # --- Schedule-free wrapper update using direction=Obar and step size=eta_hat ---
                # y update:
                #   y <- (1-ckp1)*y + ckp1*z + eta_hat * (beta1*(1-ckp1) - 1) * Obar
                # z update:
                #   z <- z - eta_hat * Obar
                p.lerp_(end=z, weight=ckp1)
                p.add_(Obar, alpha=eta_hat * (beta1 * (1.0 - ckp1) - 1.0))
                z.sub_(Obar, alpha=eta_hat)

            # log a representative LR (mean eta_hat across params in this group)
            group["scheduled_lr"] = (eta_hat_sum / max(1, eta_hat_cnt)) if eta_hat_cnt > 0 else 0.0

            group["k"] = k + 1

        return loss
