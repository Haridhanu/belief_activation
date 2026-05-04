from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from multi_agent.tgn import TGNModule


@dataclass
class Graph:
    emb_dim: int = 768
    attention_step: float = 0.2
    prior_variance: float = 1.0
    obs_variance: float = 0.05
    confidence_floor: float = 0.25

    _raw: dict[str, np.ndarray] = field(default_factory=dict)
    _z: dict[str, np.ndarray] = field(default_factory=dict)
    _adj: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _edges: dict[tuple[str, str], float] = field(default_factory=dict)
    _z_tensor: np.ndarray | None = None
    _z_index: dict[str, int] = field(default_factory=dict)

    _tgn: TGNModule | None = field(default=None, repr=False)
    _edge_count: int = field(default=0, init=False, repr=False)
    tgn_blend: float = field(default=0.3, repr=False)

    def __len__(self) -> int:
        return len(self._raw)

    def get_nodes(self) -> list[str]:
        return list(self._raw.keys())

    def get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        if node_id not in self._raw:
            return []
        return [
            (nid, self._edges.get(self._edge_key(node_id, nid), 0.0))
            for nid in self._adj[node_id]
        ]

    def get_representations_fast(self, node_ids: list[str]) -> np.ndarray:
        """Batched read from a cached ``(N, emb_dim)`` tensor."""
        if self._z_tensor is None:
            all_ids = list(self._z.keys())
            self._z_index = {nid: i for i, nid in enumerate(all_ids)}
            self._z_tensor = np.stack([self._z[nid] for nid in all_ids])
        rows = np.array([self._z_index[nid] for nid in node_ids])
        return self._z_tensor[rows]

    def extend(
        self,
        new_ids: list[str],
        new_embs: np.ndarray,
        edges: list[tuple[str, str, float]],
    ) -> None:
        touched: set[str] = set()
        for nid, emb in zip(new_ids, new_embs):
            if nid in self._raw:
                continue
            e = np.asarray(emb)
            self._raw[nid] = e.copy()
            self._z[nid] = e[: self.emb_dim].copy()
            touched.add(nid)
        for a, b, w in edges:
            if a in self._raw and b in self._raw:
                key = self._edge_key(a, b)
                if key in self._edges:
                    continue
                self._adj[a].add(b)
                self._adj[b].add(a)
                self._edges[key] = float(w)
                # Notify TGN of the new edge event
                if self._tgn is not None:
                    _sign = 1.0 if w > 0 else (-1.0 if w < 0 else 0.0)
                    self._tgn.update(
                        a, b,
                        sign=_sign,
                        timestamp=float(self._edge_count),
                        edge_weight=float(w),
                    )
                    self._edge_count += 1
                touched.add(a)
                touched.add(b)
        if touched:
            self._update_representations(touched)
            self._z_tensor = None

    def _update_representations(self, touched: set[str]) -> None:

        updates: dict[str, np.ndarray] = {}
        for v in touched:
            z_v = self._z.get(v)
            if z_v is None:
                continue
            neighbors = self.get_neighbors(v)
            if not neighbors:
                continue
            nbr_z = np.stack([self._z[u] for u, _ in neighbors if u in self._z])
            nbr_w = np.array(
                [w for u, w in neighbors if u in self._z], dtype=np.float32
            )
            if nbr_z.shape[0] == 0:
                continue
            v_norm = np.linalg.norm(z_v) or 1.0
            nbr_norm = np.linalg.norm(nbr_z, axis=1)
            nbr_norm[nbr_norm == 0.0] = 1.0
            cos = (nbr_z @ z_v) / (nbr_norm * v_norm)
            logits = cos * np.abs(nbr_w)
            logits = logits - logits.max()
            attn = np.exp(logits)
            attn = attn / attn.sum()
            signed_z = nbr_z * np.sign(nbr_w)[:, None]
            agg = attn @ signed_z
            new_z = z_v + self.attention_step * agg.astype(z_v.dtype)
            norm = np.linalg.norm(new_z)
            if norm > 0 and np.isfinite(norm):
                new_z = new_z / norm
            updates[v] = new_z

        # TGN blend: mix projected memory into _z for touched nodes
        if self._tgn is not None and touched:
            touched_with_z = [v for v in touched if v in self._z]
            if touched_with_z:
                tgn_mems = self._tgn.get_memory(touched_with_z)       # (N, memory_dim)
                projected = self._tgn.project_to_emb(tgn_mems)        # (N, emb_dim), detached
                proj_np = projected.cpu().numpy().astype(np.float32)
                for i, v in enumerate(touched_with_z):
                    base = updates.get(v, self._z[v])
                    tgn_contrib = proj_np[i]
                    tc_norm = np.linalg.norm(tgn_contrib)
                    if tc_norm > 0:
                        tgn_contrib = tgn_contrib / tc_norm
                    blended = (1.0 - self.tgn_blend) * base + self.tgn_blend * tgn_contrib
                    b_norm = np.linalg.norm(blended)
                    if b_norm > 0 and np.isfinite(b_norm):
                        blended = blended / b_norm
                    updates[v] = blended

        for v, new_z in updates.items():
            self._z[v] = new_z

    def _prior(self, q: str, c: str) -> tuple[float, float, float]:
        observed = self._edges.get(self._edge_key(q, c))
        if observed is not None:
            return observed, self.obs_variance, 1.0 / self.obs_variance
        numerator = 0.0
        sq_weight_sum = 0.0
        for k, w_qk in self.get_neighbors(q):
            if k == c or k == q:
                continue
            y_kc = self._edges.get(self._edge_key(k, c))
            if y_kc is None:
                continue
            w = float(w_qk)
            numerator += w * y_kc
            sq_weight_sum += w * w
        data_precision = sq_weight_sum / self.obs_variance
        total_precision = data_precision + 1.0 / self.prior_variance
        mu = (numerator / self.obs_variance) / total_precision
        var = 1.0 / total_precision
        return mu, var, data_precision

    def impute(self, q: str, c: str) -> float | None:
        if q == c:
            return None
        mu, _, data_precision = self._prior(q, c)
        if data_precision < self.confidence_floor:
            return None
        return float(max(-1.0, min(1.0, mu)))

    def field(self, q: str, c: str) -> float:
        """Bayesian mean prediction for edge ``(q, c)``, always defined.

        Same as ``impute`` but without the confidence floor — returns the
        prior mean clamped to ``[-1, 1]`` even when support is weak.
        """
        mu, _, _ = self._prior(q, c)
        return float(max(-1.0, min(1.0, mu)))

    def info_gain(self, q: str, c: str, y: float) -> float:
        if self._edge_key(q, c) in self._edges:
            return 0.0
        mu0, var0, _ = self._prior(q, c)
        var_post = var0 * self.obs_variance / (var0 + self.obs_variance)
        mu_post = (self.obs_variance * mu0 + var0 * float(y)) / (
            var0 + self.obs_variance
        )
        kl = 0.5 * (
            math.log(var0 / var_post) + (var_post + (mu_post - mu0) ** 2) / var0 - 1.0
        )
        return abs(float(y)) * max(0.0, kl)

    @staticmethod
    def _edge_key(a: str, b: str) -> tuple[str, str]:
        return (min(a, b), max(a, b))
