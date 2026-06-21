"""Temporal Graph Network for belief activation — PyTorch Geometric backend.

The module keeps an authoritative **data index** for the belief graph: a
``node_id → row`` mapping plus an ``edge_index`` (2, E) of committed signed
edges. Neighbourhood aggregation runs through a PyG ``TransformerConv`` over
that ``edge_index`` — a single vectorised message-passing pass enriches every
node at once, instead of one batch-size-1 attention call per pair.

Two prediction paths:

* ``predict_link(src, dst, nbr_mems_src=, nbr_mems_dst=)`` — single pair, kept
  for the per-pair callers and for autograd in training. When neighbour
  memories are supplied they are enriched via a small star-graph conv.
* ``predict_links(pairs)`` — **batched**: one whole-graph conv pass over
  ``edge_index`` enriches all nodes, then one ``link_head`` forward scores every
  pair. This is what the hot ``_impute_after_judge`` path uses.

Memory accumulation (message encoder + GRU updater) is event-driven and
unchanged in spirit; only the neighbourhood aggregator and the pair-scoring
fan-out moved to PyG + batched tensor ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


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
        src_mem: torch.Tensor,  # (memory_dim,)
        dst_mem: torch.Tensor,  # (memory_dim,)
        sign: float,
        time_enc: torch.Tensor,  # (time_dim,)
        weight: float,
    ) -> torch.Tensor:  # (memory_dim,)
        dev = src_mem.device
        x = torch.cat(
            [
                src_mem,
                dst_mem,
                torch.tensor([sign], dtype=torch.float32, device=dev),
                time_enc.to(dev),
                torch.tensor([abs(weight)], dtype=torch.float32, device=dev),
            ]
        )
        return self.norm(self.proj(x))


class NodeMemory:
    """Per-node memory buffer. Plain Python — not an nn.Module.
    Memory is always detached on ``set`` so gradients never accumulate across
    steps (use ``set_no_detach`` inside a training step)."""

    def __init__(self, memory_dim: int) -> None:
        self.memory_dim = memory_dim
        self._store: dict[str, torch.Tensor] = {}

    def get(self, node_id: str, device: torch.device | None = None) -> torch.Tensor:
        mem = self._store.get(node_id)
        if mem is None:
            return torch.zeros(self.memory_dim, device=device)
        if device is not None:
            return mem.to(device)
        return mem

    def set(self, node_id: str, memory: torch.Tensor) -> None:
        self._store[node_id] = memory.detach()

    def set_no_detach(self, node_id: str, memory: torch.Tensor) -> None:
        """Store memory tensor *without* detaching from the autograd graph.

        Use only inside a training step where the caller will explicitly
        call :py:meth:`TGNModule.detach_all_memory` after backward to
        cut the graph at the step boundary.
        """
        self._store[node_id] = memory

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
        return self.gru(message.unsqueeze(0), current_memory.unsqueeze(0)).squeeze(0)


class TemporalNeighborhoodAggregator(nn.Module):
    """Graph-attention over neighbour memories via PyG ``TransformerConv``.

    Replaces the previous batch-size-1 ``nn.MultiheadAttention`` aggregator.
    ``TransformerConv`` operates on an ``edge_index`` so the *same* layer serves
    both the single-node star-graph path (``forward``) and the vectorised
    whole-graph path (``enrich_graph``) — one shared set of weights.

    The conv keeps a learnable self-transform (``root_weight=True``), so a node
    with a single neighbour does not collapse to that neighbour's projection —
    the residual the old implementation added by hand is now built into the
    layer.
    """

    def __init__(self, memory_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        if memory_dim % n_heads != 0:
            raise ValueError(
                f"memory_dim ({memory_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.conv = TransformerConv(
            in_channels=memory_dim,
            out_channels=memory_dim // n_heads,
            heads=n_heads,
            concat=True,
            root_weight=True,
        )

    def enrich_graph(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Vectorised: enrich every node from its neighbours in one pass.

        ``x`` is ``(N, memory_dim)`` node memory; ``edge_index`` is ``(2, E)``
        with messages flowing ``src → dst`` (neighbour → node). Returns
        ``(N, memory_dim)``.
        """
        return self.conv(x, edge_index)

    def forward(
        self,
        query_mem: torch.Tensor,  # (memory_dim,)
        neighbor_mems: torch.Tensor,  # (n_neighbors, memory_dim)
    ) -> torch.Tensor:  # (memory_dim,)
        """Single-node enrichment via a star graph (neighbours → center)."""
        n = neighbor_mems.shape[0]
        x = torch.cat([query_mem.unsqueeze(0), neighbor_mems], dim=0)  # (n+1, d)
        src = torch.arange(1, n + 1, device=x.device, dtype=torch.long)
        dst = torch.zeros(n, device=x.device, dtype=torch.long)
        edge_index = torch.stack([src, dst])  # (2, n): neighbour -> center
        out = self.conv(x, edge_index)
        return out[0]  # center row


class TGNModule(nn.Module):
    """Temporal Graph Network with a PyG-backed data index.

    Maintains per-node memory plus an authoritative ``edge_index`` of committed
    signed edges. Standalone except for ``torch`` + ``torch_geometric``.

    Lifecycle per question/session::

        tgn = TGNModule(emb_dim=768)
        tgn.train_step(events)           # accumulate memory + commit edges
        scores = tgn.predict_links(pairs)  # batched scoring
        tgn.reset()                      # before each new question
    """

    def __init__(
        self,
        emb_dim: int,
        memory_dim: int = 128,
        time_dim: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if memory_dim % n_heads != 0:
            raise ValueError(
                f"memory_dim ({memory_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.emb_dim = emb_dim
        self.memory_dim = memory_dim

        self.time_encoder = TimeEncoder(time_dim)
        self.msg_encoder = TemporalMessageEncoder(memory_dim, time_dim)
        self.memory = NodeMemory(memory_dim)
        self.updater = MemoryUpdater(memory_dim)
        self.aggregator = TemporalNeighborhoodAggregator(memory_dim, n_heads)

        # Projects memory → belief embedding space for _z blending
        self.mem_to_emb = nn.Linear(memory_dim, emb_dim)
        # Link predictor head: concat(mem_i, mem_j) → signed score in [-1, 1].
        # Trainable end-to-end via ``link_loss``; replaces the legacy
        # untrained sigmoid head.
        self.link_head = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
            nn.Tanh(),
        )

        self._ref_time: float = 0.0
        # Data index: undirected committed edges (deduped). edge_index is built
        # on demand from these against the current node ordering.
        self._committed_edges: set[tuple[str, str]] = set()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # --- data index --------------------------------------------------------

    @staticmethod
    def _edge_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _commit_edge(self, src_id: str, dst_id: str) -> None:
        """Record an edge in the data index (undirected, deduped)."""
        if src_id != dst_id:
            self._committed_edges.add(self._edge_key(src_id, dst_id))

    def _build_index(self) -> tuple[list[str], dict[str, int]]:
        """Deterministic node ordering from current memory store."""
        ids = sorted(self.memory._store.keys())
        return ids, {nid: i for i, nid in enumerate(ids)}

    def _edge_index_for(self, index: dict[str, int]) -> torch.Tensor:
        """Build a (2, 2E) bidirectional edge_index over the given node order.

        Only edges whose both endpoints are in ``index`` (i.e. have memory) are
        included. Returns an empty ``(2, 0)`` long tensor when there are none.
        """
        device = self.device
        pairs: list[list[int]] = []
        for a, b in self._committed_edges:
            ia = index.get(a)
            ib = index.get(b)
            if ia is None or ib is None:
                continue
            pairs.append([ia, ib])
            pairs.append([ib, ia])
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

    def _enriched_all(self) -> tuple[dict[str, int], torch.Tensor]:
        """One whole-graph conv pass. Returns (node_index, enriched (N, d))."""
        ids, index = self._build_index()
        if not ids:
            return index, torch.empty((0, self.memory_dim), device=self.device)
        x = self.memory.get_batch(ids, device=self.device)
        edge_index = self._edge_index_for(index)
        enriched = self.aggregator.enrich_graph(x, edge_index)
        return index, enriched

    # --- memory accumulation ----------------------------------------------

    def update(
        self,
        src_id: str,
        dst_id: str,
        sign: float,
        timestamp: float,
        edge_weight: float,
    ) -> None:
        """Process one edge event: update both endpoints' memory and commit
        the edge to the data index. Detaches memory (no cross-call autograd)."""
        device = self.device
        delta_t = torch.tensor(
            timestamp - self._ref_time, dtype=torch.float32, device=device
        )
        time_enc = self.time_encoder(delta_t)

        src_mem = self.memory.get(src_id, device=device)
        dst_mem = self.memory.get(dst_id, device=device)

        msg_for_dst = self.msg_encoder(src_mem, dst_mem, sign, time_enc, edge_weight)
        msg_for_src = self.msg_encoder(dst_mem, src_mem, sign, time_enc, edge_weight)

        self.memory.set(dst_id, self.updater(msg_for_dst, dst_mem))
        self.memory.set(src_id, self.updater(msg_for_src, src_mem))
        self._commit_edge(src_id, dst_id)
        self._ref_time = timestamp

    def get_memory(self, node_ids: list[str]) -> torch.Tensor:
        """Return (N, memory_dim) memory matrix; zeros for unseen nodes."""
        return self.memory.get_batch(node_ids, device=self.device)

    def project_to_emb(self, memories: torch.Tensor) -> torch.Tensor:
        """Project (N, memory_dim) → (N, emb_dim). Detached — no grad."""
        with torch.no_grad():
            return self.mem_to_emb(memories.to(self.device))

    def predict_link(self, src_id: str, dst_id: str) -> float:
        """Trained signed score for edge (src, dst) in ``[-1, 1]``.

        Forward pass through ``link_head``. Detached — for inference. Use
        :py:meth:`predict_link_grad` when you need gradient (e.g. inside
        :py:meth:`link_loss`).
        """
        combined = torch.cat([self.memory.get(src_id), self.memory.get(dst_id)])
        with torch.no_grad():
            return float(self.link_head(combined).item())

    def predict_link_grad(self, src_id: str, dst_id: str) -> torch.Tensor:
        """Same as :py:meth:`predict_link` but with autograd enabled.

        Returns a 0-d tensor in ``[-1, 1]``. Memory tensors are detached on
        store, so gradient stops at the boundary — only ``link_head``'s
        parameters receive grad through this call.
        """
        combined = torch.cat([self.memory.get(src_id), self.memory.get(dst_id)])
        return self.link_head(combined).squeeze(-1)

    def link_loss(
        self, pairs: list[tuple[str, str, float]]
    ) -> torch.Tensor:
        """MSE between predicted signed link strength and judge ``y``.

        ``pairs`` is a list of ``(src_id, dst_id, y_target)``. Returns a
        scalar Tensor suitable for ``.backward()``. Empty list → 0-tensor.
        """
        if not pairs:
            return torch.zeros((), requires_grad=True)
        preds = torch.stack(
            [self.predict_link_grad(s, d) for s, d, _ in pairs]
        )
        targets = torch.tensor(
            [float(y) for _, _, y in pairs], dtype=torch.float32
        )
        return ((preds - targets) ** 2).mean()

    def train_step(
        self,
        events: list[tuple[str, str, float, float, float, float]],
    ) -> torch.Tensor:
        """One training pass over a batch of judged events.

        ``events`` is a list of ``(src, dst, sign, timestamp, edge_weight, y_truth)``
        tuples processed in the order given. For each event:

        1. The pair's prediction is computed from *pre-event* memory via
           ``link_head``. Gradient flows through ``link_head`` and — if
           prior events in this batch updated memory under autograd —
           also through the message encoder and GRU updater of those
           prior events.
        2. The squared error ``(pred - y_truth)²`` is added to the loss.
        3. Memory for both endpoints is updated under autograd. Storage
           uses :py:meth:`NodeMemory.set_no_detach` so the autograd graph
           stays alive for subsequent events in this batch.

        Returns the mean loss as a 0-d tensor. The caller is responsible
        for ``loss.backward()``, ``optimizer.step()``, and
        :py:meth:`detach_all_memory` afterwards. The detach-after-storage
        discipline confines BPTT to a single batch.

        This replaces the old ``link_loss + update`` pattern as the
        canonical training entry point — and is what actually trains the
        message encoder + GRU updater (which were previously only
        forward-applied, never gradient-updated).
        """
        if not events:
            return torch.zeros((), requires_grad=True)

        losses: list[torch.Tensor] = []
        for src, dst, sign, timestamp, edge_weight, y_truth in events:
            m_src = self.memory.get(src)
            m_dst = self.memory.get(dst)

            combined = torch.cat([m_src, m_dst])
            pred = self.link_head(combined).squeeze(-1)
            target = combined.new_tensor(float(y_truth))
            losses.append((pred - target) ** 2)

            delta_t = m_src.new_tensor(timestamp - self._ref_time)
            time_enc = self.time_encoder(delta_t)
            msg_for_dst = self.msg_encoder(
                m_src, m_dst, sign, time_enc, edge_weight
            )
            msg_for_src = self.msg_encoder(
                m_dst, m_src, sign, time_enc, edge_weight
            )
            new_m_dst = self.updater(msg_for_dst, m_dst)
            new_m_src = self.updater(msg_for_src, m_src)

            self.memory.set_no_detach(dst, new_m_dst)
            self.memory.set_no_detach(src, new_m_src)
            self._ref_time = timestamp

        return torch.stack(losses).mean()

    def detach_all_memory(self) -> None:
        """Detach every stored memory tensor in place.

        Called after ``loss.backward()`` and ``optimizer.step()`` to cut
        the autograd graph at the step boundary so BPTT does not persist
        across steps.
        """
        for node_id in list(self.memory._store.keys()):
            v = self.memory._store[node_id]
            if v.requires_grad:
                self.memory._store[node_id] = v.detach()

    def reset(self) -> None:
        """Zero all memory and the data index. Call before a new question."""
        self.memory.reset()
        self._committed_edges.clear()
        self._ref_time = 0.0

    # --- persistence -------------------------------------------------------

    def state_dict(self, **kwargs) -> dict:
        sd = super().state_dict(**kwargs)
        sd["_node_memory"] = self.memory.state_dict()
        sd["_ref_time"] = torch.tensor(self._ref_time)
        sd["_committed_edges"] = sorted(self._committed_edges)
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        state_dict = dict(state_dict)
        node_mem = state_dict.pop("_node_memory", {})
        ref_time_t = state_dict.pop("_ref_time", torch.tensor(0.0))
        committed = state_dict.pop("_committed_edges", [])
        result = super().load_state_dict(state_dict, strict=strict)
        self.memory.load_state_dict(node_mem)
        device = next(self.parameters()).device
        for k in self.memory._store:
            self.memory._store[k] = self.memory._store[k].to(device)
        self._committed_edges = {self._edge_key(a, b) for a, b in committed}
        self._ref_time = float(ref_time_t.item())
        return result
