from typing import Dict, Type

TREE_MODEL_REGISTRY: Dict[str, Type] = {}

def register_tree_model(name: str):
    def wrapper(cls):
        if name in TREE_MODEL_REGISTRY:
            raise KeyError(f"Tree model '{name}' already registered")
        TREE_MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

__all__ = ["TREE_MODEL_REGISTRY", "register_tree_model"]
