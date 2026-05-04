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


class TemporalMessageEncoder(nn.Module):
    """Projects (src_mem, dst_mem, sign, time_enc, weight) → memory_dim."""

    def __init__(self, memory_dim: int, time_dim: int) -> None:
        super().__init__()
        # 2*memory_dim (memories) + time_dim + 1 (sign) + 1 (weight)
        self.proj = nn.Linear(2 * memory_dim + time_dim + 2, memory_dim)
        self.norm = nn.LayerNorm(memory_dim)

    def forward(
        self,
        src_mem: torch.Tensor,   # (memory_dim,)
        dst_mem: torch.Tensor,   # (memory_dim,)
        sign: float,
        time_enc: torch.Tensor,  # (time_dim,)
        weight: float,
    ) -> torch.Tensor:           # (memory_dim,)
        dev = src_mem.device
        x = torch.cat([
            src_mem,
            dst_mem,
            torch.tensor([sign], dtype=torch.float32, device=dev),
            time_enc.to(dev),
            torch.tensor([abs(weight)], dtype=torch.float32, device=dev),
        ])
        return self.norm(self.proj(x))
