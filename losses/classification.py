# losses/classification.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_loss


@register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.from_logits = bool(from_logits)
        self.reduction = reduction
        # 用 buffer 保存 pos_weight（可跟随 device）
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits_or_prob: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            return F.binary_cross_entropy_with_logits(
                logits_or_prob,
                target01,
                pos_weight=self.pos_weight,
                reduction=self.reduction,
            )
        return F.binary_cross_entropy(logits_or_prob, target01, reduction=self.reduction)


@register_loss("cce")
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        loss = -(target_prob * logp).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


@register_loss("sparse_cce")
class SparseCategoricalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if class_weight is not None and not isinstance(class_weight, torch.Tensor):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target_idx.long(),
            weight=self.class_weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


@register_loss("wce")
@register_loss("weighted_ce")
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        if class_weight is not None and not isinstance(class_weight, torch.Tensor):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target_idx.long(),
            weight=self.class_weight,
            reduction=self.reduction,
        )


@register_loss("label_smoothing_ce")
class CrossEntropyWithLabelSmoothing(nn.Module):

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0,1), got {smoothing}")
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        target_idx = target_idx.long()
        num_classes = logits.size(-1)

        with torch.no_grad():
            target = torch.zeros_like(logits).fill_(self.smoothing / (num_classes - 1))
            target.scatter_(1, target_idx.unsqueeze(1), 1.0 - self.smoothing)

        logp = F.log_softmax(logits, dim=-1)
        loss = -(target * logp).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


@register_loss("nll")
class NegativeLogLikelihoodLoss(nn.Module):

    def __init__(self, reduction: str = "mean", class_weight: torch.Tensor | None = None):
        super().__init__()
        if class_weight is not None and not isinstance(class_weight, torch.Tensor):
            class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.nll = nn.NLLLoss(weight=class_weight, reduction=reduction)

    def forward(self, log_prob: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        return self.nll(log_prob, target_idx.long())


@register_loss("polyloss")
class PolyLoss(nn.Module):

    def __init__(self, eps: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        target_idx = target_idx.long()
        logp = F.log_softmax(logits, dim=-1)

        ce = F.nll_loss(logp, target_idx, reduction="none")
        pt = torch.exp(-ce)  # p_t
        loss = ce + self.eps * (1.0 - pt)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")


@register_loss("hinge")
class HingeLoss(nn.Module):

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction

    def forward(self, score: torch.Tensor, target_pm1: torch.Tensor) -> torch.Tensor:
        loss = torch.clamp(self.margin - target_pm1 * score, min=0.0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unknown reduction: {self.reduction}")
