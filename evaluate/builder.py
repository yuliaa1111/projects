from __future__ import annotations
from typing import Any, Dict, Union

from .evaluator import Evaluator


def build_evaluator(cfg_or_name: Union[str, Dict[str, Any]], **override_params) -> Evaluator:

    if isinstance(cfg_or_name, str):
        name = cfg_or_name
        params = dict(override_params)
    else:
        name = cfg_or_name.get("name", "default")
        params = dict(cfg_or_name.get("params", {}))
        params.update(override_params)

    if name not in ("default", "evaluator"):
        raise KeyError(f"Unknown evaluator '{name}', available: ['default','evaluator']")
    return Evaluator(**params)
