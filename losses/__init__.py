from .builder import build_loss

from . import regression  # noqa: F401
from . import classification  # noqa: F401

__all__ = ["build_loss"]
