# ret_pred/models/registry.py
from __future__ import annotations
from typing import Callable, Dict, Type
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Decorator to register a torch model class.

    Usage:
        @register_model("linear")
        class LinearModel(nn.Module): ...
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Model name must be a non-empty string")

    def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered: {MODEL_REGISTRY[name]}")
        if not issubclass(cls, nn.Module):
            raise TypeError("Registered model must be a subclass of torch.nn.Module")
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper
