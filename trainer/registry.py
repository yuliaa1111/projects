from __future__ import annotations

from typing import Dict, Type, Callable, Any


TRAINER_REGISTRY: Dict[str, Type[Any]] = {}


def register_trainer(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a trainer class."""
    def _wrap(cls: Type[Any]) -> Type[Any]:
        if name in TRAINER_REGISTRY:
            raise KeyError(f"Trainer '{name}' already registered")
        TRAINER_REGISTRY[name] = cls
        return cls
    return _wrap


def get_trainer_cls(name: str) -> Type[Any]:
    if name not in TRAINER_REGISTRY:
        raise KeyError(f"Unknown trainer '{name}', available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[name]
