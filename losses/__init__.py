"""
Loss package entrypoint.

Avoid importing torch-heavy modules at package import time.
This keeps tree-only workflows (e.g. lgbm/xgb/catboost) free from torch dependency
when they only need objective mapping.
"""

from __future__ import annotations


def build_loss(name: str, **kwargs):
    # Lazy import to avoid requiring torch unless NN loss is actually used.
    from . import regression  # noqa: F401
    from . import classification  # noqa: F401
    from .builder import build_loss as _build_loss
    return _build_loss(name, **kwargs)


__all__ = ["build_loss"]
