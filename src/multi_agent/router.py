"""Read-only consumer interface for the trained AttentionAgent population.

`Router.from_redis` is the entry point used by path_integral and vig. It loads
a snapshot, verifies HMACs, hydrates an AgentPopulation, and exposes
`rank(query_emb, k)` returning σ-weighted top-k node ids.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import torch

from multi_agent.agent import AgentPopulation
from multi_agent.config import MultiAgentConfig
from multi_agent.snapshot import (
    RouterSchemaMismatch,
    RouterSnapshot,
    SnapshotStore,
)

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _available_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


class RouterMissing(Exception):
    """Raised when no snapshot is available in Redis."""


class RouterVerifyFailed(Exception):
    """Raised when HMAC verification or post-load consistency checks fail."""


# Re-exported for callers; keeps the import surface in one place.
__all__ = [
    "Router",
    "RouterMissing",
    "RouterVerifyFailed",
    "RouterSchemaMismatch",
]


class Router:
    def __init__(
        self,
        *,
        population: AgentPopulation,
        bid_to_z: dict[str, np.ndarray],
        bid_to_text: dict[str, str],
        sigma: dict[str, float],
        roles: dict[str, str],
        emb_dim: int,
        fusion: str = "sigma",
    ) -> None:
        self.population = population
        self.bid_to_text = bid_to_text
        self.sigma = sigma
        self.roles = roles
        self.emb_dim = emb_dim
        self.fusion = fusion

        # Pre-stack candidate matrix for fast scoring.
        self._bid_order: list[str] = list(bid_to_z.keys())
        if self._bid_order:
            # Fix C: validate every vector matches emb_dim before stacking.
            bad = [
                bid for bid, v in bid_to_z.items() if np.asarray(v).shape != (emb_dim,)
            ]
            if bad:
                raise RouterVerifyFailed(
                    f"bid_to_z shape mismatch: emb_dim={emb_dim} but {bad[:3]} have wrong shape"
                )
            self._cand_matrix = np.stack(
                [bid_to_z[bid] for bid in self._bid_order]
            ).astype(np.float32, copy=False)
        else:
            self._cand_matrix = np.empty((0, emb_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rank(self, query_emb: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return σ-weighted top-k (bid, score) pairs over the in-memory pool."""
        if self._cand_matrix.shape[0] == 0:
            return []
        if self._cand_matrix.shape[0] == 1:
            return [(self._bid_order[0], 0.0)]

        fused = self._fused_scores(query_emb, self._cand_matrix)
        if fused is None:
            # All agents degenerate.
            return []
        order = np.argsort(-fused)[:k]
        return [(self._bid_order[i], float(fused[i])) for i in order]

    def score(self, query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
        """σ-weighted fused scores for caller-supplied candidates.

        Used by vig where the candidate set isn't the snapshot's full pool.
        Returns finite values; degenerate pools yield zeros.
        """
        if candidate_embs.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        if candidate_embs.shape[0] == 1:
            return np.zeros((1,), dtype=np.float32)
        fused = self._fused_scores(query_emb, candidate_embs)
        if fused is None:
            return np.zeros((candidate_embs.shape[0],), dtype=np.float32)
        return fused

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_snapshot(
        cls,
        snapshot: RouterSnapshot,
        weights: bytes,
        fusion: str = "sigma",
    ) -> "Router":
        cfg_d = snapshot.multi_agent_config
        if cfg_d.get("emb_dim") != snapshot.emb_dim:
            raise RouterVerifyFailed(
                f"emb_dim mismatch: config={cfg_d.get('emb_dim')!r} snap={snapshot.emb_dim!r}"
            )
        device = _available_device(str(cfg_d.get("device", "cpu")))
        config = MultiAgentConfig(
            emb_dim=int(cfg_d["emb_dim"]),
            device=device,
            num_agents=int(cfg_d["num_agents"]),
            k=int(cfg_d["k"]),
            temperature=float(cfg_d.get("temperature", 0.3)),
            agent_roles=dict(cfg_d.get("agent_roles", {})),
        )
        population = AgentPopulation(config)
        loaded = torch.load(
            io.BytesIO(weights), weights_only=True, map_location=torch.device(device)
        )
        try:
            population.load_state_dict(loaded["state_dict"])
        except KeyError as exc:
            raise RouterVerifyFailed(
                f"weights blob missing 'state_dict': {exc}"
            ) from exc
        except RuntimeError as exc:
            # Shape mismatch → typically an emb_dim discrepancy between the
            # snapshot's declared emb_dim and what the weights were trained with.
            raise RouterVerifyFailed(
                f"emb_dim mismatch or shape error loading weights: {exc}"
            ) from exc
        return cls(
            population=population,
            bid_to_z=snapshot.graph_z,
            bid_to_text=snapshot.bid_to_text,
            sigma=snapshot.sigma,
            roles=snapshot.roles,
            emb_dim=snapshot.emb_dim,
            fusion=fusion,
        )

    @classmethod
    def from_redis(
        cls,
        store: SnapshotStore,
        session_id: str,
        fusion: str = "sigma",
    ) -> "Router":
        loaded = store.load(session_id)
        if loaded is None:
            raise RouterMissing(session_id)
        snapshot, weights = loaded
        return cls.from_snapshot(snapshot, weights, fusion=fusion)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fused_scores(
        self, query_emb: np.ndarray, cand_matrix: np.ndarray
    ) -> np.ndarray | None:
        """Compute σ-weighted z-scored fusion. Returns None if all degenerate."""
        # Fix A: raise early if the requested single agent is not in the population.
        if self.fusion.startswith("single:"):
            wanted = self.fusion[len("single:") :]
            if wanted not in {a.agent_id for a in self.population.agents}:
                raise RouterVerifyFailed(
                    f"single fusion: agent {wanted!r} not in population"
                )

        device = self.population.device
        q = torch.from_numpy(np.asarray(query_emb, dtype=np.float32)).to(device)
        c = torch.from_numpy(np.asarray(cand_matrix, dtype=np.float32)).to(device)

        # Fix B: two-pass accumulation so uniform divisor counts only
        # non-degenerate contributors, not the full population size.
        contribs: list[np.ndarray] = []
        agents = self.population.agents
        for agent in agents:
            sig_a = self._sigma_for(agent.agent_id)
            if sig_a == 0.0 and self.fusion == "sigma":
                continue
            with torch.no_grad():
                raw = agent.score_candidates(q, c)
            scores = (
                (raw[0] if isinstance(raw, tuple) else raw)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            std = float(scores.std())
            if std < _EPS:
                continue  # degenerate agent contributes zero
            z = (scores - scores.mean()) / std
            if self.fusion == "sigma":
                contribs.append(sig_a * z)
            elif self.fusion == "uniform":
                contribs.append(z)  # divide later, after we know how many contributed
            elif self.fusion.startswith("single:"):
                wanted = self.fusion[len("single:") :]
                if agent.agent_id == wanted:
                    return z  # short-circuit; ordering only depends on this one
            else:
                raise ValueError(f"unknown fusion mode: {self.fusion!r}")
        if not contribs:
            return None
        if self.fusion == "uniform":
            return np.sum(contribs, axis=0) / len(contribs)
        return np.sum(contribs, axis=0)

    def _sigma_for(self, agent_id: str) -> float:
        return float(self.sigma.get(agent_id, 0.0))
