"""TGN-only belief-activation trainer.

Sibling of :py:class:`multi_agent.runner.Trainer`. Same external interface
(``step(batch) → StepResult``) but a radically simpler internal:

  - No multi-agent population, no PSRO loop, no Bayesian ``_prior``,
    no BlendedImputer.
  - One :py:class:`TGNModule` with a trained ``link_head``.
  - Active-learning loop: cosine top-k pre-filter → TGN scores → judge the
    most-uncertain → MSE on judge truth → memory propagation → commit.

Selected via ``MultiAgentConfig.engine = "tgn_only"``. The default engine
remains ``"psro"`` and is bit-identical to the pre-TGN baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.graph import Graph
from multi_agent.judge import Judge
from multi_agent.runner import StepResult, StepStats
from multi_agent.tgn import TGNModule
from multi_agent.utils import run_sync, score_pairs


@dataclass
class TGNStepDebug:
    """Per-step diagnostic — kept on the trainer for inspection."""

    candidates: int = 0
    judged: int = 0
    committed_judged: int = 0
    committed_predicted: int = 0
    skipped: int = 0
    link_loss: float = 0.0
    step_ms: float = 0.0


class TGNTrainer:
    """Stateful TGN-only trainer. Init once, call ``step(batch)`` per belief batch."""

    def __init__(self, config: MultiAgentConfig, judge: Judge) -> None:
        self.config = config
        self.judge = judge
        self.graph = Graph(emb_dim=config.emb_dim)

        self.tgn = TGNModule(
            emb_dim=config.emb_dim,
            memory_dim=config.tgn_memory_dim,
            time_dim=config.tgn_time_dim,
            n_heads=config.tgn_n_attn_heads,
        )
        self.optimizer = torch.optim.Adam(
            self.tgn.parameters(), lr=config.tgn_only_lr
        )

        self.score_cache: dict[tuple[str, str], float] = {}
        self.node_texts: dict[str, str] = {}
        self.history: list[StepStats] = []
        self.debug_history: list[TGNStepDebug] = []
        self._step = 0
        # Calibration log: (|prediction|, sign_correct) for every judged
        # pair. Used by `_calibrated_commit_threshold` to gate predicted
        # edges by empirical accuracy rather than a static cutoff.
        self._calibration_log: list[tuple[float, bool]] = []

    # ----- Public API ------------------------------------------------------

    def step(self, batch: Batch) -> StepResult:
        t_start = time.perf_counter()
        self._step += 1
        self.node_texts.update(dict(zip(batch.ids, batch.texts)))

        # Add new nodes to the graph (no edges yet).
        existing_nodes = list(self.graph.get_nodes())
        new_ids = [nid for nid in batch.ids if nid not in self.graph._raw]
        new_embs_idx = [i for i, nid in enumerate(batch.ids) if nid in new_ids]
        new_embs = batch.embs[new_embs_idx] if new_embs_idx else np.empty((0, batch.embs.shape[1]), dtype=batch.embs.dtype)
        if new_ids:
            self.graph.extend(new_ids, new_embs, edges=[])

        all_node_ids = list(self.graph.get_nodes())
        if len(all_node_ids) < 2 or not new_ids:
            return self._empty_result(t_start)

        candidates = self._candidate_pairs(new_ids, all_node_ids)
        if not candidates:
            return self._empty_result(t_start)

        # Drop pairs already cached.
        scorable = [p for p in candidates if p not in self.score_cache]
        cached_pairs = {p: self.score_cache[p] for p in candidates if p in self.score_cache}

        # Score candidates with current TGN. Use uncertainty for budget allocation.
        preds = self._predict_pairs(scorable)
        budget = max(0, int(self.config.judge_budget_per_batch))
        to_judge = self._most_uncertain(preds, budget)

        # Run the judge on the budgeted slice; symmetrise like the PSRO loop.
        judged_results = self._run_judge(to_judge)
        for pair, y in judged_results.items():
            self.score_cache[pair] = y

        # Record calibration before training so the curve reflects the
        # model's own pre-training predictions vs the judge's truth.
        self._record_calibration(preds, judged_results)

        # Combined train + memory-propagation in a single autograd graph.
        # Each event predicts from pre-event memory (no leakage) AND
        # updates memory under autograd, so the next event's prediction
        # backprops through the prior event's encoder + GRU. This is
        # what actually trains the message encoder + GRU updater.
        loss_value = self._train_and_propagate(judged_results)

        # Commit edges: judged ones at judge_y, predicted ones above the
        # calibration-aware confidence gate.
        committed = self._commit_edges(scorable, preds, judged_results, cached_pairs)
        if committed:
            self.graph.extend([], np.empty((0, batch.embs.shape[1]), dtype=batch.embs.dtype), committed)

        # Bookkeeping for downstream consumers.
        n_coh = sum(1 for _, _, w in committed if w > 0)
        n_dis = sum(1 for _, _, w in committed if w < 0)
        n_predicted = sum(1 for _ in committed) - len(judged_results)
        n_skipped = len(scorable) - len(judged_results) - max(0, n_predicted)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        debug = TGNStepDebug(
            candidates=len(candidates),
            judged=len(judged_results),
            committed_judged=len(judged_results),
            committed_predicted=max(0, n_predicted),
            skipped=max(0, n_skipped),
            link_loss=loss_value,
            step_ms=elapsed_ms,
        )
        self.debug_history.append(debug)

        stats = StepStats(
            step=self._step,
            winner_id="tgn",
            reward=0.0,
            loss=loss_value,
            loss_spread=0.0,
            per_agent_loss={"tgn": loss_value},
            n_nodes=len(self.graph),
            n_coh=n_coh,
            n_dis=n_dis,
            judged=len(judged_results),
            scorable=len(candidates),
            cached=len(cached_pairs),
            imputed=max(0, n_predicted),
            skipped=max(0, n_skipped),
        )
        self.history.append(stats)

        return StepResult(
            step=self._step,
            winner_id="tgn",
            mean_rewards={"tgn": 0.0},
            edges=committed,
            stats=stats,
        )

    # ----- Internals -------------------------------------------------------

    def _candidate_pairs(
        self, new_ids: list[str], all_node_ids: list[str]
    ) -> list[tuple[str, str]]:
        """Cosine top-k per new node against the rest of the graph.

        Returns an unordered list of unordered pairs ``(min, max)`` with
        ``min < max``. Pre-filter is on the unit-normed raw embeddings.
        """
        if not new_ids or not all_node_ids:
            return []
        k = max(1, int(self.config.tgn_only_candidate_k))

        def unit(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v)) or 1.0
            return v / n

        ids_index: dict[str, int] = {nid: i for i, nid in enumerate(all_node_ids)}
        embs = np.stack([unit(self.graph._raw[nid]) for nid in all_node_ids])

        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[str, str]] = []
        for nid in new_ids:
            i = ids_index[nid]
            sims = embs @ embs[i]
            sims[i] = -np.inf
            top = np.argsort(-sims)[: min(k, len(all_node_ids) - 1)]
            for j in top:
                a, b = nid, all_node_ids[int(j)]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        return pairs

    def _predict_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], float]:
        if not pairs:
            return {}
        return {p: self.tgn.predict_link(p[0], p[1]) for p in pairs}

    def _most_uncertain(
        self, preds: dict[tuple[str, str], float], budget: int
    ) -> list[tuple[str, str]]:
        if budget <= 0 or not preds:
            return []
        ranked = sorted(preds.keys(), key=lambda p: 1.0 - abs(preds[p]), reverse=True)
        return ranked[:budget]

    def _run_judge(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], float]:
        if not pairs:
            return {}
        text_pairs: list[tuple[str, str]] = []
        for q, c in pairs:
            text_pairs.append((self.node_texts.get(q, q), self.node_texts.get(c, c)))
            text_pairs.append((self.node_texts.get(c, c), self.node_texts.get(q, q)))
        raw = run_sync(score_pairs(self.judge, text_pairs, self.config.llm_concurrency))
        out: dict[tuple[str, str], float] = {}
        for i, pair in enumerate(pairs):
            y = float(max(raw[2 * i], raw[2 * i + 1], key=abs))
            out[pair] = y
        return out

    def _train_and_propagate(
        self, judged: dict[tuple[str, str], float]
    ) -> float:
        """One combined train + memory-update step over judged events.

        Calls :py:meth:`TGNModule.train_step`, then ``backward()`` /
        ``optimizer.step()``, then :py:meth:`TGNModule.detach_all_memory`
        to cut the autograd graph at the step boundary.
        """
        if not judged:
            return 0.0
        # Build event tuples with monotonically increasing timestamps so
        # the time encoder sees real progression within a step. We use
        # the current edge counter as the base offset.
        base_t = float(self.graph._edge_count) + 1.0
        events: list[tuple[str, str, float, float, float, float]] = []
        for i, ((u, v), y) in enumerate(judged.items()):
            sign = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
            events.append((u, v, sign, base_t + i, abs(float(y)), float(y)))

        self.optimizer.zero_grad()
        loss = self.tgn.train_step(events)
        if not loss.requires_grad:
            self.tgn.detach_all_memory()
            return float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tgn.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.tgn.detach_all_memory()
        return float(loss.item())

    def _record_calibration(
        self,
        predictions: dict[tuple[str, str], float],
        judged: dict[tuple[str, str], float],
    ) -> None:
        """Append ``(|pred|, sign_correct)`` entries for every judged pair
        whose ground-truth ``y`` is non-trivially signed."""
        for pair, y_truth in judged.items():
            if abs(y_truth) < 1e-6:
                continue
            pred = predictions.get(pair)
            if pred is None:
                continue
            sign_correct = bool(np.sign(pred) == np.sign(y_truth) and abs(pred) > 1e-6)
            self._calibration_log.append((abs(float(pred)), sign_correct))

    def _calibrated_commit_threshold(self) -> float:
        """Return the magnitude threshold above which empirical sign
        accuracy meets ``tgn_only_calibration_target``. Falls back to the
        static config threshold if the warmup hasn't accumulated enough
        samples or no magnitude bin is accurate enough."""
        cfg_threshold = float(self.config.tgn_only_commit_threshold)
        target = float(self.config.tgn_only_calibration_target)
        warmup = int(self.config.tgn_only_calibration_warmup)
        if len(self._calibration_log) < warmup:
            return cfg_threshold
        # Find the smallest |pred| such that the empirical sign-accuracy
        # of pairs at-or-above that magnitude meets the target.
        sorted_log = sorted(self._calibration_log)
        n = len(sorted_log)
        for i in range(n):
            above = sorted_log[i:]
            acc = sum(1 for _, ok in above if ok) / len(above)
            if acc >= target:
                return float(sorted_log[i][0])
        return cfg_threshold

    def _commit_edges(
        self,
        scorable: list[tuple[str, str]],
        preds: dict[tuple[str, str], float],
        judged: dict[tuple[str, str], float],
        cached: dict[tuple[str, str], float],
    ) -> list[tuple[str, str, float]]:
        threshold = self._calibrated_commit_threshold()
        out: list[tuple[str, str, float]] = []
        for (u, v), y in cached.items():
            out.append((u, v, float(y)))
        for (u, v), y in judged.items():
            out.append((u, v, float(y)))
        committed_keys = {(u, v) for u, v, _ in out}
        for pair in scorable:
            if pair in committed_keys:
                continue
            score = preds.get(pair)
            if score is None:
                continue
            if abs(score) >= threshold:
                out.append((pair[0], pair[1], float(score)))
        return out

    def _empty_result(self, t_start: float) -> StepResult:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.debug_history.append(TGNStepDebug(step_ms=elapsed_ms))
        stats = StepStats(
            step=self._step,
            winner_id="tgn",
            reward=0.0,
            loss=0.0,
            loss_spread=0.0,
            per_agent_loss={"tgn": 0.0},
            n_nodes=len(self.graph),
            n_coh=0,
            n_dis=0,
            judged=0,
            scorable=0,
            cached=0,
            imputed=0,
            skipped=0,
        )
        self.history.append(stats)
        return StepResult(
            step=self._step,
            winner_id="tgn",
            mean_rewards={"tgn": 0.0},
            edges=[],
            stats=stats,
        )
