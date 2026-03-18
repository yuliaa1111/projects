# ret_pred/models/linear.py
from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn

from .registry import register_model

@register_model("linear")
class LinearModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        bias: bool = True,
        seq_reduce: Literal["last", "mean"] = "last",
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.seq_reduce = seq_reduce
        self.out_dim = int(out_dim)
        self.fc = nn.Linear(self.input_size, self.out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"x must be torch.Tensor, got {type(x)}")

        if x.dim() == 3:
            # (B, T, F)
            if self.seq_reduce == "last":
                x = x[:, -1, :]
            elif self.seq_reduce == "mean":
                x = x.mean(dim=1)
            else:
                raise ValueError(f"Unknown seq_reduce={self.seq_reduce}")
        elif x.dim() == 2:
            # (B, F)
            pass
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {x.dim()} with shape {tuple(x.shape)}")

        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected last dim={self.input_size}, got {x.size(-1)}")

        y = self.fc(x)  # (B, out_dim)
        if y.size(-1) == 1:
            return y.squeeze(-1)  # (B,)
        return y
