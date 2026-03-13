from __future__ import annotations

from typing import Any, Dict, Union
import inspect

from .registry import TREE_MODEL_REGISTRY


def _call_drop_unknown_kwargs(ctor, params: Dict[str, Any], verbose: bool = True):
    params = params or {}
    sig = inspect.signature(ctor)

    # ctor 支持 **kwargs => 直接传
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return ctor(**params)

    # ctor 不支持 **kwargs => 过滤未知参数（避免报错）
    allowed = set(sig.parameters.keys())
    filtered: Dict[str, Any] = {}
    dropped = []

    for k, v in params.items():
        if k in allowed:
            filtered[k] = v
        else:
            dropped.append(k)

    if dropped and verbose:
        name = getattr(ctor, "__name__", str(ctor))
        print(f"[tree_models][builder] Dropped unsupported params for {name}: {dropped}")

    return ctor(**filtered)


def build_tree_model(cfg_or_name: Union[str, Dict[str, Any]], **override_params):
    # -------------------------
    # 1) parse name + params
    # -------------------------
    if isinstance(cfg_or_name, str):
        name = cfg_or_name
        params = {}
    elif isinstance(cfg_or_name, dict):
        name = cfg_or_name.get("name")

        # 优先 model_config
        params = dict(cfg_or_name.get("model_config", {}) or {})

        # 向后兼容：如果没写 model_config，就退回 params
        if not params:
            params = dict(cfg_or_name.get("params", {}) or {})
    else:
        raise TypeError(f"cfg_or_name must be str or dict, got {type(cfg_or_name)}")

    # override 永远最后覆盖
    params.update(override_params)

    if not name:
        raise ValueError("Tree model name is missing. Expect cfg['name'] or pass name str.")

    if name not in TREE_MODEL_REGISTRY:
        raise KeyError(f"Unknown tree model '{name}', available: {list(TREE_MODEL_REGISTRY.keys())}")

    ctor = TREE_MODEL_REGISTRY[name]
    return _call_drop_unknown_kwargs(ctor, params, verbose=True)
