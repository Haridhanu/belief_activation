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


class NodeMemory:
    """Per-node memory buffer. Plain Python — not an nn.Module.
    Memory is always detached so gradients never accumulate across steps."""

    def __init__(self, memory_dim: int) -> None:
        self.memory_dim = memory_dim
        self._store: dict[str, torch.Tensor] = {}

    def get(self, node_id: str) -> torch.Tensor:
        return self._store.get(node_id, torch.zeros(self.memory_dim))

    def set(self, node_id: str, memory: torch.Tensor) -> None:
        self._store[node_id] = memory.detach()

    def get_batch(self, node_ids: list[str]) -> torch.Tensor:
        return torch.stack([self.get(nid) for nid in node_ids])

    def reset(self) -> None:
        self._store.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self._store.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._store = {k: v.clone() for k, v in state.items()}


class MemoryUpdater(nn.Module):
    """GRUCell that updates one node's memory from an aggregated message."""

    def __init__(self, memory_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim)

    def forward(
        self, message: torch.Tensor, current_memory: torch.Tensor
    ) -> torch.Tensor:
        # GRUCell expects (batch, dim) — unsqueeze/squeeze around the call
        return self.gru(
            message.unsqueeze(0), current_memory.unsqueeze(0)
        ).squeeze(0)


class TemporalNeighborhoodAggregator(nn.Module):
    """Attention over neighbor memories — used for soft link prediction."""

    def __init__(self, memory_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=n_heads, batch_first=True
        )

    def forward(
        self,
        query_mem: torch.Tensor,      # (memory_dim,)
        neighbor_mems: torch.Tensor,  # (n_neighbors, memory_dim)
    ) -> torch.Tensor:                # (memory_dim,)
        q = query_mem.unsqueeze(0).unsqueeze(0)   # (1, 1, memory_dim)
        kv = neighbor_mems.unsqueeze(0)            # (1, n_neighbors, memory_dim)
        out, _ = self.attn(q, kv, kv)
        return out.squeeze(0).squeeze(0)           # (memory_dim,)
