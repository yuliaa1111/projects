# ret_pred/nn_models/transformer.py
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .registry import register_model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


@register_model("transformer")
class TransformerModel(nn.Module):
    """
    TransformerEncoder-based sequence model.

    Input:  x (N, L, F)
    Output: y (N,)
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                 # (N, L, d_model)
        h = self.pe(h)
        h = self.encoder(h)              # (N, L, d_model)
        last = h[:, -1, :]               # (N, d_model)
        y = self.head(last).squeeze(-1)  # (N,)
        return y
