from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss

@register_loss("mse")
class MSELoss(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


@register_loss("mae")
@register_loss("l1")
class MAELoss(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


@register_loss("huber")
class HuberLoss(nn.Module):

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta, reduction=self.reduction)


@register_loss("logcosh")
class LogCoshLoss(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred - target
        loss = F.softplus(2.0 * x) - x - torch.log(
            torch.tensor(2.0, device=x.device, dtype=x.dtype)
        )

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


@register_loss("quantile")
class QuantileLoss(nn.Module):

    def __init__(self, q: float = 0.5, reduction: str = "mean"):
        super().__init__()
        if not (0.0 < q < 1.0):
            raise ValueError(f"q must be in (0,1), got {q}")
        self.q = float(q)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = target - pred
        loss = self.q * torch.clamp(e, min=0.0) + (1.0 - self.q) * torch.clamp(-e, min=0.0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


@register_loss("poisson")
class PoissonLoss(nn.Module):

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.log_input = bool(log_input)
        self.full = bool(full)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.poisson_nll_loss(
            pred,
            target,
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
        )
