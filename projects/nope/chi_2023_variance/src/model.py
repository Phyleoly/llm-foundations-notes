"""A minimal NoPE attention block for operator checks, not a paper reproduction."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seq_length: int = 512
    hidden_dim: int = 768
    eps: float = 1e-5
    num_heads: int = 12


class NoPEModel(nn.Module):
    """LayerNorm followed by attention, with batch-first inputs [B, T, D]."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            config.hidden_dim,
            eps=config.eps,
            device=config.device,
            dtype=config.dtype,
        )
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=0.0,
            batch_first=True,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, inputs: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [B, T, D]")
        normalized = self.layer_norm(inputs)
        length = inputs.size(1)
        mask = (
            torch.ones(length, length, device=inputs.device, dtype=torch.bool).triu(1)
            if causal
            else None
        )
        output, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        return output
