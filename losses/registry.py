from __future__ import annotations
from typing import Callable, Dict
import torch.nn as nn

_LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_loss(name: str):
    key = name.lower()
    def deco(cls):
        _LOSS_REGISTRY[key] = cls
        return cls
    return deco

def get_loss_cls(name: str):
    key = name.lower()
    if key not in _LOSS_REGISTRY:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(_LOSS_REGISTRY.keys())}")
    return _LOSS_REGISTRY[key]
