from __future__ import annotations
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Deterministic sinusoidal time encoding. No learnable parameters."""

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.time_dim = time_dim

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        d = self.time_dim
        pos = torch.arange(d, dtype=torch.float32, device=delta_t.device)
        freqs = delta_t.float() / (10000.0 ** (2 * (pos // 2) / d))
        return torch.where(pos % 2 == 0, torch.sin(freqs), torch.cos(freqs))
