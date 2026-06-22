"""Balance-theory signed multi-hop GNN substrate.

A static signed-graph link predictor + node encoder. ``layers`` stacked PyG
``SignedConv`` layers (= hops) propagate balanced/unbalanced channels by the
structural-balance rules (friend-of-friend → +, enemy-of-enemy → +). The node
embeddings double as candidate representations for seed selection; a small link
head turns a node pair into a signed score in ``(-1, 1)``.

Unlike the temporal TGN, there is no per-node memory and no time encoding — the
graph is treated as a static signed set of edges, recomputed on demand.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SignedConv


class SignedGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, layers: int = 3) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.in_dim = in_dim
        self.hidden = hidden
        self.convs = nn.ModuleList(
            [SignedConv(in_dim, hidden, first_aggr=True)]
            + [SignedConv(hidden, hidden, first_aggr=False) for _ in range(layers - 1)]
        )
        # SignedConv concatenates balanced+unbalanced → 2*hidden per node, so a
        # pair's concat is 4*hidden.
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self._cache: tuple | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def emb(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.relu(self.convs[0](x, pos_edge_index, neg_edge_index))
        for conv in self.convs[1:]:
            z = torch.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def score(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raw signed logit per pair; apply tanh/sigmoid at the call site."""
        return self.head(
            torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        ).squeeze(-1)

    @staticmethod
    def _edge_tensors(
        edges: list[tuple[int, int, float]], device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = [[i, j] for i, j, y in edges if y > 0] + [
            [j, i] for i, j, y in edges if y > 0
        ]
        neg = [[i, j] for i, j, y in edges if y < 0] + [
            [j, i] for i, j, y in edges if y < 0
        ]

        def ten(e):
            if not e:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            return torch.tensor(e, dtype=torch.long, device=device).t().contiguous()

        return ten(pos), ten(neg)

    def fit(
        self,
        x,
        edges: list[tuple[int, int, float]],
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> None:
        """Train the signed link predictor on observed ``(i, j, y)`` edges
        (``y`` in ``{+,-}``). Caches the edge index for subsequent predict()."""
        device = self.device
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        pos, neg = self._edge_tensors(edges, device=device)
        self._cache = (x, pos, neg)
        if not edges:
            return
        ei = torch.tensor([[i, j] for i, j, _ in edges], dtype=torch.long, device=device).t()
        y = torch.tensor([1.0 if v > 0 else 0.0 for *_, v in edges], device=device)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()
        for _ in range(epochs):
            opt.zero_grad()
            z = self.emb(x, pos, neg)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                self.score(z, ei), y
            )
            loss.backward()
            opt.step()
        self.eval()

    @torch.no_grad()
    def predict(self, x, pairs: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Signed score in ``[-1, 1]`` per pair, using the edges cached at fit()."""
        if not pairs:
            return {}
        device = self.device
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if self._cache is not None:
            pos, neg = self._cache[1], self._cache[2]
        else:
            pos, neg = self._edge_tensors([], device=device)
        z = self.emb(x, pos, neg)
        ei = torch.tensor(pairs, dtype=torch.long, device=device).t()
        s = torch.tanh(self.score(z, ei)).tolist()
        if not isinstance(s, list):
            s = [s]
        return {p: float(v) for p, v in zip(pairs, s)}
