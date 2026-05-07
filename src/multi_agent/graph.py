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
    """Belief graph with optional TGN substrate.

    Two operating modes, controlled by the presence of an attached
    :py:class:`TGNModule` (``self._tgn``):

    1. **Bayesian baseline** (``_tgn is None``, default).
       ``_z[id]`` holds a signed-attention-updated representation that
       gets refreshed in :py:meth:`_update_representations` after every
       new edge. ``impute`` / ``field`` use a Bayesian 2-hop posterior
       over neighbour evidence.

    2. **TGN substrate** (``_tgn`` attached).
       ``candidate_reps`` come from the TGN's per-node memory (projected
       to ``emb_dim`` via the trained ``mem_to_emb`` linear). The
       signed-attention ``_z`` update is **not** run — the TGN owns
       representation. ``impute`` / ``field`` delegate to
       ``tgn.predict_link``. Memory propagation happens during PSRO's
       train_step hook (see ``psro.PSROLoop.step``), not from
       ``Graph.extend``.
    """

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

    _tgn: "TGNModule | None" = field(default=None, repr=False)
    _edge_count: int = field(default=0, init=False, repr=False)
    _edge_timestamps: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    # Cold-start mode for TGN substrate: "pure" (always project memory)
    # or "raw_fallback" (use raw embedding for nodes with no memory yet).
    tgn_cold_start: str = field(default="pure", repr=False)
    # Confidence threshold for TGN-substrate impute() — below this, impute
    # returns None and the pair escalates to the judge.
    tgn_predict_threshold: float = field(default=0.2, repr=False)

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

    # --- representations ---------------------------------------------------

    def _has_tgn_memory(self, node_id: str) -> bool:
        if self._tgn is None:
            return False
        mem = self._tgn.memory.get(node_id)
        # NodeMemory.get returns zeros for unseen nodes; treat zeros as
        # "memory has not yet been touched" (cold start).
        import torch
        return bool(torch.any(mem != 0.0).item())

    def _candidate_rep(self, node_id: str) -> np.ndarray:
        """Resolve the per-node representation that agents read.

        - No TGN: the signed-attention-updated ``_z[node_id]``.
        - TGN attached, memory cold (all-zeros) and ``tgn_cold_start="raw_fallback"``:
          fall back to ``_raw[node_id]``.
        - TGN attached otherwise: ``mem_to_emb(memory[node_id])`` projected.
        """
        if self._tgn is None:
            return self._z[node_id]

        if self.tgn_cold_start == "raw_fallback" and not self._has_tgn_memory(node_id):
            return self._raw[node_id].astype(np.float32)

        mem = self._tgn.get_memory([node_id])
        proj = self._tgn.project_to_emb(mem)
        return proj[0].detach().cpu().numpy().astype(np.float32)

    def get_representations_fast(self, node_ids: list[str]) -> np.ndarray:
        """Batched read of per-node candidate representations.

        Cached as a single ``(N, emb_dim)`` tensor when no TGN is
        attached. With TGN attached, recomputed per call from current
        memory (this is cheap — one Linear forward per node).
        """
        if self._tgn is None:
            if self._z_tensor is None:
                all_ids = list(self._z.keys())
                self._z_index = {nid: i for i, nid in enumerate(all_ids)}
                self._z_tensor = np.stack([self._z[nid] for nid in all_ids])
            rows = np.array([self._z_index[nid] for nid in node_ids])
            return self._z_tensor[rows]

        return np.stack([self._candidate_rep(nid) for nid in node_ids])

    # --- streaming update --------------------------------------------------

    def extend(
        self,
        new_ids: list[str],
        new_embs: np.ndarray,
        edges: list[tuple[str, str, float]],
    ) -> None:
        """Add nodes + edges. With TGN attached, this only commits edge
        bookkeeping — TGN memory is updated by ``PSROLoop.step`` via
        ``tgn.train_step`` so the encoder + GRU can learn under autograd.
        """
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
                self._edge_count += 1
                self._edge_timestamps[key] = self._edge_count
                touched.add(a)
                touched.add(b)
        if touched:
            self._update_representations(touched)
            self._z_tensor = None

    def _update_representations(self, touched: set[str]) -> None:
        """Refresh ``_z`` for touched nodes via signed attention.

        With TGN attached this is a **no-op** — the TGN's memory IS the
        representation, and is updated under autograd inside
        ``PSROLoop.step``. We still keep ``_z`` populated with the most
        recent projected memory so legacy readers (e.g. inference paths
        that snapshot ``_z``) see fresh values.
        """
        if self._tgn is not None:
            for v in touched:
                if v in self._raw:
                    self._z[v] = self._candidate_rep(v)
            return

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

        for v, new_z in updates.items():
            self._z[v] = new_z

    # --- imputation --------------------------------------------------------

    def _prior(self, q: str, c: str) -> tuple[float, float, float]:
        """Bayesian 2-hop posterior. Used only when no TGN is attached."""
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
        """Predict the signed weight of an unobserved edge ``(q, c)``,
        or return ``None`` when confidence is below the floor.

        Delegates to ``tgn.predict_link`` when a TGN is attached;
        otherwise uses the Bayesian 2-hop posterior.
        """
        if q == c:
            return None
        observed = self._edges.get(self._edge_key(q, c))
        if observed is not None:
            return float(max(-1.0, min(1.0, observed)))

        if self._tgn is not None:
            score = float(self._tgn.predict_link(q, c))
            if abs(score) < self.tgn_predict_threshold:
                return None
            return float(max(-1.0, min(1.0, score)))

        mu, _, data_precision = self._prior(q, c)
        if data_precision < self.confidence_floor:
            return None
        return float(max(-1.0, min(1.0, mu)))

    def field(self, q: str, c: str) -> float:
        """Always-defined signed prediction in ``[-1, 1]``.

        Delegates to ``tgn.predict_link`` when a TGN is attached;
        otherwise the Bayesian prior mean (clamped).
        """
        if q == c:
            return 0.0
        observed = self._edges.get(self._edge_key(q, c))
        if observed is not None:
            return float(max(-1.0, min(1.0, observed)))
        if self._tgn is not None:
            return float(max(-1.0, min(1.0, self._tgn.predict_link(q, c))))
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
