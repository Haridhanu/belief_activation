from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

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
    _nbr_mems_cache: dict[str, "torch.Tensor | None"] = field(
        default_factory=dict, init=False, repr=False
    )
    _divergence_log: list[float] = field(default_factory=list, init=False, repr=False)
    _edge_count: int = field(default=0, init=False, repr=False)
    _edge_timestamps: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    # Cold-start mode for TGN substrate. "raw_fallback" (default) keeps
    # cold nodes distinguishable by returning the raw embedding until
    # memory is written for that node. "pure" always projects memory
    # through ``mem_to_emb`` — under cold memory, every node returns the
    # bias of ``mem_to_emb`` and ranking degenerates. See
    # ``MultiAgentConfig.tgn_cold_start`` for the long form.
    tgn_cold_start: str = field(default="raw_fallback", repr=False)
    # Confidence threshold for TGN-substrate impute() — below this, impute
    # returns None and the pair escalates to the judge.
    tgn_predict_threshold: float = field(default=0.2, repr=False)

    # Signed-GNN hybrid substrate. _sgnn is a SignedGNN (attached by Trainer when
    # graph_substrate="signed_hybrid"); a pair uses it only when BOTH endpoints'
    # degree >= hybrid_density_threshold, else falls back to the Bayesian posterior.
    # _sgnn_index maps node_id -> row in _sgnn_features (set by Trainer after fit).
    _sgnn: "object | None" = field(default=None, repr=False)
    hybrid_density_threshold: int = field(default=6, repr=False)
    _sgnn_index: "dict[str, int] | None" = field(default=None, repr=False)
    _sgnn_features: "np.ndarray | None" = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self._raw)

    def get_nodes(self) -> list[str]:
        return list(self._raw.keys())

    @property
    def tgn(self) -> "TGNModule | None":
        return self._tgn

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

    def _tgn_raw_fallback_score(self, q: str, c: str) -> float | None:
        """Raw-embedding cosine for cold TGN pairs in raw_fallback mode.

        Cold TGN memories are all zeros, so ``predict_link(q, c)`` collapses
        every cold pair to the same random-init score. In raw_fallback mode,
        ``field`` uses raw embedding geometry until both endpoints have real
        memory. ``impute`` does not use this score to create edges because raw
        cosine cannot distinguish same-topic contradiction from coherence.
        """
        q_raw = self._raw.get(q)
        c_raw = self._raw.get(c)
        if q_raw is None or c_raw is None:
            return None
        q_norm = float(np.linalg.norm(q_raw))
        c_norm = float(np.linalg.norm(c_raw))
        if q_norm == 0.0 or c_norm == 0.0:
            return 0.0
        score = float(np.dot(q_raw, c_raw) / (q_norm * c_norm))
        return float(max(-1.0, min(1.0, score)))

    def _uses_tgn_raw_fallback(self, q: str, c: str) -> bool:
        """Whether a pair should avoid TGN zero-memory prediction."""
        if self._tgn is None or self.tgn_cold_start != "raw_fallback":
            return False
        return not (self._has_tgn_memory(q) and self._has_tgn_memory(c))

    # --- signed-GNN hybrid substrate ---------------------------------------

    def _degree(self, node_id: str) -> int:
        return len(self._adj.get(node_id, ()))

    def _hybrid_use_signed(self, q: str, c: str) -> bool:
        """True iff a Signed-GNN is attached AND both endpoints are dense."""
        if self._sgnn is None:
            return False
        return (
            self._degree(q) >= self.hybrid_density_threshold
            and self._degree(c) >= self.hybrid_density_threshold
        )

    def _sgnn_predict(self, q: str, c: str) -> float | None:
        """Signed-GNN score for (q, c), or None if it can't be computed yet."""
        idx = self._sgnn_index
        if idx is None or self._sgnn_features is None or q not in idx or c not in idx:
            return None
        out = self._sgnn.predict(self._sgnn_features, [(idx[q], idx[c])])
        return out.get((idx[q], idx[c]))

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
            self.clear_nbr_mems_cache()
            self._update_representations(touched)
            self._z_tensor = None

    def clear_nbr_mems_cache(self) -> None:
        self._nbr_mems_cache.clear()

    def mean_representation_divergence(self) -> float:
        if not self._divergence_log:
            return 0.0
        return float(np.mean(self._divergence_log))

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
                    new_rep = self._candidate_rep(v)
                    self._z[v] = new_rep

                    raw = self._raw[v][: self.emb_dim]
                    denom = (np.linalg.norm(new_rep) * np.linalg.norm(raw)) + 1e-9
                    cos_sim = float(np.dot(new_rep, raw) / denom)
                    cos_dist = 1.0 - cos_sim
                    if np.isfinite(cos_dist):
                        self._divergence_log.append(float(cos_dist))
                        if len(self._divergence_log) > 1000:
                            del self._divergence_log[:-1000]
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

    def _nbr_mems(self, node_id: str) -> "torch.Tensor | None":
        """Return stacked TGN memories for ``node_id``'s current neighbours.

        Used to pass live structural context to :py:meth:`TGNModule.predict_link`
        so predictions reflect the node's **current** graph neighbourhood,
        not just its stored pairwise history.

        Returns ``None`` when:
        - No TGN is attached (Bayesian baseline mode).
        - The node has no neighbours yet (cold node).
        - The node is not in the graph.

        The returned tensor has shape ``(n_neighbours, memory_dim)`` and is
        on the TGN's device, ready to be passed directly to
        :py:meth:`TGNModule.predict_link` as ``nbr_mems_src`` /
        ``nbr_mems_dst``.
        """
        if self._tgn is None:
            return None
        if node_id in self._nbr_mems_cache:
            return self._nbr_mems_cache[node_id]
        neighbors = [nid for nid, _ in self.get_neighbors(node_id) if nid in self._raw]
        if not neighbors:
            self._nbr_mems_cache[node_id] = None
            return None
        nbr_mems = self._tgn.memory.get_batch(neighbors, device=self._tgn.device)
        self._nbr_mems_cache[node_id] = nbr_mems
        return nbr_mems

    def _prior(self, q: str, c: str) -> tuple[float, float, float]:
        """Bayesian 2-hop posterior over the unobserved edge ``(q, c)``.

        Returns ``(mu, var, data_precision)``. Used by :py:meth:`impute`
        and :py:meth:`field` only when no TGN is attached, and by
        :py:meth:`info_gain` (which is itself TGN-incompatible)."""
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

        # Signed-GNN hybrid: use it only for dense pairs; sparse pairs fall
        # through to the Bayesian posterior below.
        if self._hybrid_use_signed(q, c):
            score = self._sgnn_predict(q, c)
            if score is not None:
                return float(max(-1.0, min(1.0, score)))

        if self._tgn is not None:
            if self._uses_tgn_raw_fallback(q, c):
                return None
            score = float(
                self._tgn.predict_link(
                    q,
                    c,
                    nbr_mems_src=self._nbr_mems(q),
                    nbr_mems_dst=self._nbr_mems(c),
                )
            )
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
        # Signed-GNN hybrid: dense pairs only; sparse fall through to Bayesian.
        if self._hybrid_use_signed(q, c):
            score = self._sgnn_predict(q, c)
            if score is not None:
                return float(max(-1.0, min(1.0, score)))
        if self._tgn is not None:
            if self._uses_tgn_raw_fallback(q, c):
                raw_score = self._tgn_raw_fallback_score(q, c)
                return 0.0 if raw_score is None else raw_score
            return float(
                max(
                    -1.0,
                    min(
                        1.0,
                        self._tgn.predict_link(
                            q,
                            c,
                            nbr_mems_src=self._nbr_mems(q),
                            nbr_mems_dst=self._nbr_mems(c),
                        ),
                    ),
                )
            )
        mu, _, _ = self._prior(q, c)
        return float(max(-1.0, min(1.0, mu)))

    def impute_batch(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], float | None]:
        """Batched :py:meth:`impute`. With a TGN attached, all warm pairs are
        scored in a single ``tgn.predict_links`` call (one conv pass + one
        link-head forward) instead of one forward per pair — the hot path in
        ``PSROLoop._impute_after_judge``. Gating (observed edge, cold raw_fallback,
        confidence threshold) matches :py:meth:`impute` exactly.
        """
        out: dict[tuple[str, str], float | None] = {}
        if self._tgn is None:
            for q, c in pairs:
                out[(q, c)] = self.impute(q, c)
            return out

        tgn_pairs: list[tuple[str, str]] = []
        for q, c in pairs:
            if q == c:
                out[(q, c)] = None
                continue
            observed = self._edges.get(self._edge_key(q, c))
            if observed is not None:
                out[(q, c)] = float(max(-1.0, min(1.0, observed)))
                continue
            if self._uses_tgn_raw_fallback(q, c):
                out[(q, c)] = None
                continue
            tgn_pairs.append((q, c))

        scores = self._tgn.predict_links(tgn_pairs)
        for (q, c), score in zip(tgn_pairs, scores):
            if abs(score) < self.tgn_predict_threshold:
                out[(q, c)] = None
            else:
                out[(q, c)] = float(max(-1.0, min(1.0, score)))
        return out

    def info_gain(self, q: str, c: str, y: float) -> float:
        """KL between Bayesian prior and posterior over edge ``(q, c)``.

        Only defined for the Bayesian baseline. The TGN substrate has no
        notion of posterior variance, so a faithful Gaussian-KL value
        cannot be computed — callers should use ``field`` and compute
        surprisal themselves if they need a TGN-mode signal.
        """
        if self._tgn is not None:
            raise NotImplementedError(
                "Graph.info_gain is not defined for the TGN substrate "
                "(no posterior variance). Use field()/predict_link and "
                "compute surprisal externally."
            )
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
