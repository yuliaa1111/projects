from __future__ import annotations
import torch.nn as nn
from .registry import get_loss_cls

def build_loss(name: str, **kwargs) -> nn.Module:
    cls = get_loss_cls(name)
    return cls(**kwargs)
