"""Blended imputer for the belief graph.

Replaces the pure Bayesian path in :py:meth:`Graph.impute` / :py:meth:`Graph.field`
with a learned three-way blend of:

1. TGN link prediction (sigmoid on memory inner product, rescaled to ``[-1, 1]``)
2. Recency-weighted Bayesian posterior mean (existing :py:meth:`Graph._prior`)
3. Cosine similarity of signed-attention representations ``_z``

A small gate MLP turns the three raw scores plus current edge density into a
softmax over the three sources. The gate is trained from judge-revealed truth
via MSE — :py:meth:`BlendedImputer.train_on_judged` is called by
:py:class:`PSROLoop` before the judged edges are committed to the graph, so the
training signal is computed against the *pre-update* state.

Gradients only update the gate weights — TGN memory is detached upstream and
the Bayes / cosine components are arithmetic, not learned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule


class ImputationScorer(nn.Module):
    """Gate MLP: ``(tgn, bayes, cosine, density) → softmax(3)``.

    Output is a 3-vector summing to 1 — the weights for blending the three
    component scores in the order ``[tgn, bayes, cosine]``.

    The final layer is initialised so that the *initial* softmax output is
    Bayes-dominant — roughly ``[0.10, 0.65, 0.25]`` — for two reasons:

    1. The Bayesian posterior is the only one of the three with a built-in
       structural justification at cold start. TGN memory cosine is near-zero
       on unseen nodes (no signal) and approaches ``+1`` everywhere once
       memories saturate (false signal); cosine on ``_z`` is mildly
       informative but noisy on small graphs.
    2. With limited supervision (a few dozen judged pairs per run) a
       uniform-init gate cannot recover from a misaligned component fast
       enough — it must be biased toward the safest source and *learn* to
       weight others up.
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            self.net[-1].bias.copy_(torch.tensor([-1.5, 2.0, -1.0]))

    def forward(
        self, scores: torch.Tensor, density: float | torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(density, torch.Tensor):
            density_t = scores.new_tensor([float(density)])
        else:
            density_t = density.reshape(1).to(scores)
        x = torch.cat([scores, density_t])
        return torch.softmax(self.net(x), dim=-1)


class BlendedImputer:
    """Three-way blended imputer over a :py:class:`Graph`.

    Args:
        graph: the graph this imputer is attached to.
        tgn: optional TGN module (used for the TGN component).
        scorer: optional pre-built scorer; one is created if not given.
        confidence_floor: minimum ``max_weight * agreement`` for ``impute`` to
            return a value rather than ``None``.
        lr: Adam lr for the gate optimizer.
    """

    def __init__(
        self,
        graph: "Graph",
        tgn: "TGNModule | None" = None,
        scorer: ImputationScorer | None = None,
        confidence_floor: float = 0.1,
        lr: float = 1e-3,
    ) -> None:
        self.graph = graph
        self.tgn = tgn
        self.scorer = scorer if scorer is not None else ImputationScorer()
        self.confidence_floor = float(confidence_floor)
        self.optimizer = torch.optim.Adam(self.scorer.parameters(), lr=lr)
        # Per-call audit. Each entry is (w_tgn, w_bayes, w_cosine).
        self.weight_history: list[tuple[float, float, float]] = []

    def _component_scores(self, q: str, c: str) -> tuple[float, float, float]:
        """Return raw ``(tgn, bayes, cosine)`` scores for pair ``(q, c)``.

        - ``tgn`` is the cosine of the two nodes' TGN memory vectors. Two
          nodes that received similar edge histories accumulate similar
          memory and so cosine-correlate; the sign is meaningful.
          (We don't use ``predict_link`` because its head is never trained
          anywhere in the codebase, so its output is random init noise — a
          softmax gate cannot invert a misaligned component.)
        - ``bayes`` is the recency-weighted Bayesian posterior mean.
        - ``cosine`` is the cosine of the signed-attention representations.

        All three live in roughly ``[-1, 1]``.
        """
        # Use the *structural* Bayes (no recency weighting) as the gate's
        # Bayes input. Recency weighting belongs to the gate's job, not
        # the input pre-processing.
        prev_decay = self.graph.time_decay
        self.graph.time_decay = 0.0
        try:
            bayes_mu, _, _ = self.graph._prior(q, c)
        finally:
            self.graph.time_decay = prev_decay
        bayes = float(max(-1.0, min(1.0, bayes_mu)))

        if self.tgn is not None:
            mem_q = self.tgn.memory.get(q)
            mem_c = self.tgn.memory.get(c)
            qn = float(mem_q.norm().item())
            cn = float(mem_c.norm().item())
            if qn > 0.0 and cn > 0.0:
                tgn = float(torch.dot(mem_q, mem_c).item() / (qn * cn))
            else:
                tgn = 0.0
        else:
            tgn = 0.0

        z_q = self.graph._z.get(q)
        z_c = self.graph._z.get(c)
        if z_q is not None and z_c is not None:
            denom = float(np.linalg.norm(z_q) * np.linalg.norm(z_c)) or 1.0
            cosine = float(np.dot(z_q, z_c) / denom)
        else:
            cosine = 0.0

        return float(tgn), bayes, cosine

    def _density(self) -> float:
        n = len(self.graph._raw)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2.0
        return min(1.0, len(self.graph._edges) / max_edges)

    def _blend(
        self, q: str, c: str
    ) -> tuple[float, torch.Tensor, tuple[float, float, float]]:
        components = self._component_scores(q, c)
        scores = torch.tensor(list(components), dtype=torch.float32)
        density = self._density()
        weights = self.scorer(scores, density)
        blended = float((weights.detach() * scores).sum().item())
        return blended, weights, components

    def impute(self, q: str, c: str) -> float | None:
        """Blended impute. Returns ``None`` when confidence is below floor."""
        if q == c or q not in self.graph._raw or c not in self.graph._raw:
            return None
        observed = self.graph._edges.get(self.graph._edge_key(q, c))
        if observed is not None:
            return float(max(-1.0, min(1.0, observed)))

        blended, weights, (tgn_s, bayes_s, _) = self._blend(q, c)
        w_np = weights.detach().cpu().numpy()
        max_weight = float(w_np.max())
        agreement = 1.0 - abs(tgn_s - bayes_s) / 2.0  # [0, 1]
        confidence = max_weight * agreement

        if confidence < self.confidence_floor:
            return None

        self.weight_history.append((float(w_np[0]), float(w_np[1]), float(w_np[2])))
        return float(max(-1.0, min(1.0, blended)))

    def field(self, q: str, c: str) -> float:
        """Always-defined blended prediction; clamped to ``[-1, 1]``."""
        if q == c:
            return 0.0
        observed = self.graph._edges.get(self.graph._edge_key(q, c))
        if observed is not None:
            return float(max(-1.0, min(1.0, observed)))
        blended, _, _ = self._blend(q, c)
        return float(max(-1.0, min(1.0, blended)))

    def train_on_judged(
        self, judged: list[tuple[tuple[str, str], float]]
    ) -> float:
        """One Adam step against MSE between blended prediction and judge ``y``.

        ``judged`` is a list of ``((qid, cid), y)`` pairs the judge actually
        scored this step. Returns the mean loss as a float (``0.0`` if empty).
        """
        if not judged:
            return 0.0

        self.optimizer.zero_grad()
        losses: list[torch.Tensor] = []
        for (q, c), y in judged:
            if q == c or q not in self.graph._raw or c not in self.graph._raw:
                continue
            components = self._component_scores(q, c)
            scores = torch.tensor(list(components), dtype=torch.float32)
            weights = self.scorer(scores, self._density())
            blended = (weights * scores).sum()
            target = scores.new_tensor(float(y))
            losses.append((blended - target) ** 2)

        if not losses:
            return 0.0

        loss = torch.stack(losses).mean()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
