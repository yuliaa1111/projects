# ret_pred/nn_models/builder.py
from __future__ import annotations

import inspect
from typing import Any, Dict

import torch.nn as nn

from .registry import MODEL_REGISTRY  # 你现有 registry


def _filter_kwargs_for_ctor(ctor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only params accepted by ctor (or ctor.__init__ if class).
    This makes switching model.name smooth in notebooks/config.
    """
    if kwargs is None:
        return {}

    target = ctor.__init__ if inspect.isclass(ctor) else ctor
    sig = inspect.signature(target)

    # if **kwargs exists, no need to filter
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return dict(kwargs)

    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    filtered = {}
    dropped = []
    for k, v in kwargs.items():
        if k in allowed:
            filtered[k] = v
        else:
            dropped.append(k)

    if dropped:
        print(f"[nn_models][builder] Dropped unsupported params for {getattr(ctor,'__name__',ctor)}: {dropped}")

    return filtered


def build_model(cfg_or_name, **override_params) -> nn.Module:
    """
    Accept either:
      - cfg dict: {"name": "...", "params": {...}}
      - name str: "lstm" / "linear" ...
    """
    if isinstance(cfg_or_name, str):
        name = cfg_or_name
        params = {}
    elif isinstance(cfg_or_name, dict):
        name = cfg_or_name.get("name")
        params = dict(cfg_or_name.get("params") or {})
    else:
        raise TypeError(f"build_model expects str or dict, got {type(cfg_or_name)}")

    if not name:
        raise ValueError("model.name is required")

    # override params from function call
    params.update(override_params or {})

    available = sorted(MODEL_REGISTRY.keys())
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    cls = MODEL_REGISTRY[name]

    # ✅ critical: filter params by model ctor signature
    params2 = _filter_kwargs_for_ctor(cls, params)

    model = cls(**params2)

    # Optional: move to device if provided
    device = params.get("device", None)
    if device is not None:
        model = model.to(device)

    return model
