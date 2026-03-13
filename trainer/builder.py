from __future__ import annotations

from typing import Any, Dict, Union

from .registry import get_trainer_cls, TRAINER_REGISTRY


def build_trainer(cfg_or_name: Union[str, Dict[str, Any]], **override_params):

    if isinstance(cfg_or_name, str):
        name = cfg_or_name
        params = dict(override_params)
    else:
        name = cfg_or_name["name"]
        params = dict(cfg_or_name.get("params", {}))
        params.update(override_params)

    cls = get_trainer_cls(name)
    return cls(**params)
