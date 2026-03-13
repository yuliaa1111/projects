# ret_pred/models/lstm.py
from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn

from .registry import register_model

@register_model("lstm")
class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pooling: Literal["last", "mean"] = "last",
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.pooling = pooling
        self.out_dim = int(out_dim)

        # nn.LSTM only applies dropout when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        d = 2 if self.bidirectional else 1
        self.head = nn.Linear(self.hidden_size * d, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"x must be torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"Expected x dim=3 (B,T,F), got dim={x.dim()} shape={tuple(x.shape)}")
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected last dim={self.input_size}, got {x.size(-1)}")

        # out: (B, T, H*d)
        out, _ = self.lstm(x)

        if self.pooling == "last":
            feat = out[:, -1, :]      # (B, H*d)
        elif self.pooling == "mean":
            feat = out.mean(dim=1)    # (B, H*d)
        else:
            raise ValueError(f"Unknown pooling={self.pooling}")

        y = self.head(feat)  # (B, out_dim)
        if y.size(-1) == 1:
            return y.squeeze(-1)  # (B,)
        return y
