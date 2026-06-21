from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from multi_agent.config import MultiAgentConfig
from multi_agent.graph import Graph
from multi_agent.judge import Judge, StaticJudge
from multi_agent.utils import (
    AgentProposal,
    JudgeResult,
    ProposalBatch,
    accumulate_pair_counts,
    build_self_mask,
    make_text_lookup,
    role_sign,
    run_sync,
    score_and_sample_agent,
    score_pairs,
    split_by_cache,
)

# Pairs with |actual| below this are treated as neutral and skipped from
# the AUC denominator — the LLM judge doesn't reliably distinguish weak
# coherence from weak contradiction at low magnitudes, so including them
# adds noise without signal.
_AUC_NEUTRAL_EPS = 0.1
# Minimum labeled pair count before we emit an AUC. Below this the metric
# is too unstable to be informative; emit None so consumers can show
# "n/a" instead of a misleading number.
_AUC_MIN_PAIRS = 10


def _auc_prediction_groups(items: list[dict]) -> tuple[list[float], list[float]]:
    """Return valid positive/negative prediction groups for signed AUC."""
    pos: list[float] = []
    neg: list[float] = []
    for it in items:
        try:
            actual = float(it["actual"])
            predicted = float(it["predicted"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(actual) or math.isnan(predicted):
            continue
        if abs(actual) < _AUC_NEUTRAL_EPS:
            continue
        (pos if actual > 0 else neg).append(predicted)
    return pos, neg


def _signed_auc_from_groups(pos: list[float], neg: list[float]) -> float | None:
    if len(pos) + len(neg) < _AUC_MIN_PAIRS or not pos or not neg:
        return None
    wins = 0
    ties = 0
    for x in pos:
        for y in neg:
            if x > y:
                wins += 1
            elif x == y:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _signed_auc(items: list[dict]) -> float | None:
    """ROC AUC for graph/TGN link prediction (#7).

    ``items`` is ``field_revealed`` — one dict per judged pair with
    ``predicted`` (graph.field score, captured BEFORE the pair was used
    to extend the graph) and ``actual`` (the LLM judge score).

    Binary label: +1 if ``actual > 0`` (coherent), 0 if ``actual < 0``
    (contradictory). Near-zero ``actual`` is treated as neutral and
    skipped. Returns ``None`` when there aren't enough labeled pairs in
    both classes — the metric isn't meaningful below ``_AUC_MIN_PAIRS``.

    Implementation is the Mann–Whitney U identity:
    AUC = P(pred_pos > pred_neg) over all positive×negative pairs, with
    ties counted as 0.5. O(N²) but ``len(items) <= judge_budget`` (~64
    in production), so an n² scan is cheaper than pulling in sklearn.
    """
    pos, neg = _auc_prediction_groups(items)
    return _signed_auc_from_groups(pos, neg)


class PSROLoop:
    def __init__(
        self,
        config: MultiAgentConfig,
        judge: Judge | None = None,
        graph: Graph | None = None,
        tgn_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.config = config
        self.step_count: int = 0
        self._optimizers: dict = {}
        self._pending_optimizer_states: dict = {}
        self.judge: Judge = judge if judge is not None else StaticJudge(0.0)
        self._graph = graph
        # Optional optimizer for the attached TGN. When provided alongside
        # ``graph._tgn``, ``step`` will train the TGN's link head, message
        # encoder, and GRU updater on every batch's judged pairs (see the
        # ``train_step`` hook below). When None, TGN — if attached — stays
        # forward-only.
        self._tgn_optimizer = tgn_optimizer
        self.last_step_stats: dict[str, int] = {
            "imputed": 0,
            "judged": 0,
            "scorable": 0,
        }
        # Multiplicative meta-weights over strategies; agent ids lazy-filled
        # on first ``step`` call. ``sigma`` derives a distribution from this.
        self._meta_weights: dict[str, float] = {}
        # Monotonic edge-time counter — incremented once per emitted event so
        # TGN sees strictly increasing timestamps even across Trainer.step
        # calls. Persisted via to_snapshot so resume preserves the sequence
        # instead of restarting at zero.
        self._edge_clock: float = 0.0

    @property
    def sigma(self) -> dict[str, float]:
        """Current mixture over strategies, derived from ``_meta_weights``.

        Returns a dict ``agent_id → σ_j`` summing to 1. Blended with a
        uniform anchor of weight ``meta_eps`` so no strategy can collapse
        to zero even under sustained losses. Empty dict before the first
        ``step`` call (weights aren't initialized yet).
        """
        if not self._meta_weights:
            return {}
        total = sum(self._meta_weights.values())
        raw = {k: v / total for k, v in self._meta_weights.items()}
        eps = self.config.meta_eps
        n = len(raw)
        return {k: (1.0 - eps) * v + eps / n for k, v in raw.items()}

    def _get_optimizer(self, agent):
        """Return Adam for agents with trainable params, else ``None``.

        Lazy: first call constructs the optimizer (and consumes any stashed
        state_dict from a snapshot resume), subsequent calls return cached.
        ``CosineActor`` has no params — we cache ``None`` so we don't try
        to re-construct an empty optimizer every step.
        """
        if agent.agent_id not in self._optimizers:
            params = list(agent.parameters())
            opt = (
                torch.optim.Adam(params, lr=self.config.learning_rate)
                if params
                else None
            )
            if opt is not None:
                pending = getattr(self, "_pending_optimizer_states", None)
                if pending and agent.agent_id in pending:
                    opt.load_state_dict(pending.pop(agent.agent_id))
            self._optimizers[agent.agent_id] = opt
        return self._optimizers[agent.agent_id]

    def _propose(
        self,
        agents,
        query_embs,
        query_ids: list[str],
        pool_embs,
        pool_ids: list[str],
        k: int,
    ) -> ProposalBatch:
        self_cols = build_self_mask(query_ids, pool_ids)
        by_agent: dict[str, AgentProposal] = {}
        pair_counts: dict[tuple[str, str], int] = {}
        for agent in agents:
            proposal = score_and_sample_agent(
                agent, query_embs, pool_embs, pool_ids, self_cols, k
            )
            by_agent[agent.agent_id] = proposal
            accumulate_pair_counts(pair_counts, query_ids, proposal.proposals)
        return ProposalBatch(by_agent=by_agent, pair_counts=pair_counts)

    def _judge(
        self,
        scorable: list[tuple[str, str]],
        pair_counts: dict[tuple[str, str], int],
        text_of: Callable[[str], str],
        context_of: Callable[[str], str],
        score_cache: dict[tuple[str, str], float] | None,
    ) -> JudgeResult:
        score_by_pair, remaining = split_by_cache(scorable, score_cache)
        imputed, deferred = self._split_by_impute(remaining)
        score_by_pair.update(imputed)

        deferred.sort(key=lambda p: -pair_counts[p])
        budget = int(getattr(self.config, "judge_budget_per_batch", 0) or 0)
        to_judge = deferred[:budget] if budget > 0 else deferred
        skipped = deferred[budget:] if budget > 0 else []

        combined_scores = self._run_judge_pairs(to_judge, text_of, context_of)
        for pair, s in zip(to_judge, combined_scores):
            score_by_pair[pair] = s
            if score_cache is not None:
                score_cache[pair] = s

        return JudgeResult(
            score_by_pair=score_by_pair,
            judged_pairs=list(zip(to_judge, combined_scores)),
            stats={
                "cached": len(scorable) - len(remaining),
                "imputed": len(imputed),
                "judged": len(to_judge),
                "skipped": len(skipped),
            },
        )

    def _split_by_impute(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[dict[tuple[str, str], float], list[tuple[str, str]]]:
        """Split pairs into (imputed-by-graph, must-be-judged)."""
        if self._graph is None:
            return {}, list(pairs)
        imputed: dict[tuple[str, str], float] = {}
        deferred: list[tuple[str, str]] = []
        for qid, cid in pairs:
            y_hat = self._graph.impute(qid, cid)
            if y_hat is None:
                deferred.append((qid, cid))
            else:
                imputed[(qid, cid)] = y_hat
        return imputed, deferred

    def _run_judge_pairs(
        self,
        pairs: list[tuple[str, str]],
        text_of: Callable[[str], str],
        context_of: Callable[[str], str],
    ) -> list[float]:
        """Symmetrized judge call: score both directions, take max-by-abs."""
        if not pairs:
            return []
        text_pairs: list[tuple[str, str]] = []
        for qid, cid in pairs:
            text_pairs.append((text_of(qid), context_of(cid)))
            text_pairs.append((text_of(cid), context_of(qid)))
        raw = run_sync(score_pairs(self.judge, text_pairs, self.config.llm_concurrency))
        return [
            float(max(raw[2 * i], raw[2 * i + 1], key=abs)) for i in range(len(pairs))
        ]

    def _reward(
        self,
        agents,
        by_agent: dict[str, AgentProposal],
        judged_pairs: list[tuple[tuple[str, str], float]],
        query_ids: list[str],
    ) -> np.ndarray:
        """Per-query reward for each trained agent.

        Zero-sum between coherence (+y) and contradiction (-y); every
        other role gets 0. Only **judged** pairs count — imputed pairs
        are excluded so we never train on the graph's own predictions.
        """
        B = len(query_ids)
        rewards = np.zeros((B, len(agents)), dtype=np.float32)
        judged_score = dict(judged_pairs)
        for qi, qid in enumerate(query_ids):
            for ai, agent in enumerate(agents):
                sign = role_sign(getattr(agent, "role", None))
                if sign == 0.0:
                    continue
                parts = [
                    sign * float(judged_score[(qid, p)])
                    for p in by_agent[agent.agent_id].proposals[qi]
                    if (qid, p) in judged_score
                ]
                if parts:
                    rewards[qi, ai] = float(np.mean(parts))
        return rewards

    def _backward(
        self,
        rewards: np.ndarray,
        agents,
        by_agent: dict[str, AgentProposal],
    ) -> dict[str, float | dict[str, float]]:
        B = rewards.shape[0]
        trainable = [(ai, a) for ai, a in enumerate(agents) if self._get_optimizer(a)]
        trainable_idx = [ai for ai, _ in trainable]
        if trainable_idx:
            baseline = rewards[:, trainable_idx].mean(axis=1, keepdims=True)
        else:
            baseline = np.zeros((B, 1), dtype=np.float32)
        advantage = rewards - baseline

        for _, agent in trainable:
            self._get_optimizer(agent).zero_grad()

        device = by_agent[agents[0].agent_id].scores.device
        adv_t = torch.tensor(advantage, dtype=torch.float32, device=device)
        total_loss: torch.Tensor | None = None
        for ai, agent in trainable:
            proposal = by_agent[agent.agent_id]
            log_soft = proposal.scores.log_softmax(dim=1)
            # Per-row accumulation: rows with 0 valid candidates contribute
            # a zero scalar tied through log_soft to stay in the graph
            # (so .backward() works even when all rows are empty).
            row_log_probs = []
            for qi, ridx in enumerate(proposal.row_indices):
                if ridx.numel() == 0:
                    row_log_probs.append((log_soft[qi] * 0.0).sum())
                else:
                    row_log_probs.append(log_soft[qi, ridx].sum())
            log_probs_picked = torch.stack(row_log_probs)  # (B,)
            agent_loss = -(adv_t[:, ai] * log_probs_picked).sum() / B
            total_loss = agent_loss if total_loss is None else total_loss + agent_loss
        if total_loss is not None:
            total_loss.backward()

        for _, agent in trainable:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            self._get_optimizer(agent).step()

        per_agent_loss = {
            a.agent_id: float(1.0 - rewards[:, ai].mean())
            for ai, a in enumerate(agents)
        }
        return {
            "loss": float(1.0 - rewards.mean()),
            "per_agent_loss": per_agent_loss,
            "loss_spread": float(np.std(list(per_agent_loss.values()))),
        }

    def _results(
        self,
        query_ids: list[str],
        agents,
        by_agent: dict[str, AgentProposal],
        score_by_pair: dict[tuple[str, str], float],
        rewards: np.ndarray,
    ) -> list[dict]:
        out: list[dict] = []
        for qi, qid in enumerate(query_ids):
            out.append(
                {
                    "query_node": qid,
                    "proposals": {
                        a.agent_id: by_agent[a.agent_id].proposals[qi] for a in agents
                    },
                    "proposal_scores": {
                        a.agent_id: [
                            float(score_by_pair.get((qid, p), 0.0))
                            for p in by_agent[a.agent_id].proposals[qi]
                        ]
                        for a in agents
                    },
                    "rewards": [float(rewards[qi, ai]) for ai in range(len(agents))],
                }
            )
        return out

    def _ensure_meta_state(self, agents) -> None:
        for a in agents:
            if a.agent_id not in self._meta_weights:
                self._meta_weights[a.agent_id] = 1.0

    def _meta_reward(
        self,
        agents,
        by_agent: dict[str, AgentProposal],
        query_ids: list[str],
        judged_pairs: list[tuple[tuple[str, str], float]],
    ) -> dict[str, float]:
        """Per-strategy surprisal credit: how much each agent's proposals
        deviated from the graph's field prediction, split across co-proposers."""
        per_strategy = {a.agent_id: 0.0 for a in agents}
        if self._graph is None or not judged_pairs:
            return per_strategy
        proposers_by_pair: dict[tuple[str, str], list[str]] = {}
        for a in agents:
            for qi, qid in enumerate(query_ids):
                for cid in by_agent[a.agent_id].proposals[qi]:
                    proposers_by_pair.setdefault((qid, cid), []).append(a.agent_id)
        for (qid, cid), y in judged_pairs:
            surprisal = abs(float(y) - self._graph.field(qid, cid))
            proposers = proposers_by_pair.get((qid, cid), [])
            if not proposers:
                continue
            share = surprisal / len(proposers)
            for aid in proposers:
                per_strategy[aid] = per_strategy.get(aid, 0.0) + share
        return per_strategy

    def _update_mixture(self, meta_rewards: dict[str, float]) -> None:
        if self.config.meta_lr <= 0.0 or not meta_rewards:
            return
        vals = list(meta_rewards.values())
        scale = max((abs(v) for v in vals), default=0.0)
        norm = (
            {k: v / scale for k, v in meta_rewards.items()}
            if scale > 0.0
            else meta_rewards
        )
        eta = self.config.meta_lr
        for aid, r in norm.items():
            self._meta_weights[aid] = self._meta_weights.get(aid, 1.0) * float(
                np.exp(eta * r)
            )

    def _alignment_targets(
        self,
        node_ids: list[str],
        device: torch.device,
    ) -> tuple[list[str], torch.Tensor | None]:
        """Return raw embedding targets for TGN representation alignment.

        ``Trainer.step`` registers current-batch raw embeddings before PSRO
        scoring, so judged endpoints should already be present in
        ``graph._raw``. Missing rows are skipped defensively.
        """
        if self._graph is None or not node_ids:
            return [], None

        aligned_ids: list[str] = []
        raw_rows: list[torch.Tensor] = []
        for nid in node_ids:
            raw = self._graph._raw.get(nid)
            if raw is None:
                continue
            raw_rows.append(torch.as_tensor(raw, dtype=torch.float32, device=device))
            aligned_ids.append(nid)

        if not raw_rows:
            return [], None
        return aligned_ids, torch.stack(raw_rows)

    def step(
        self,
        population,
        query_embs,
        query_ids: list[str],
        pool_embs,
        pool_ids: list[str],
        node_texts: dict[str, str] | None = None,
        candidate_context: Callable[[str], str] | None = None,
        score_cache: dict[tuple[str, str], float] | None = None,
    ) -> list[dict]:
        """One PSRO step: propose → judge → reward → backward → results.

        Graph construction is a sequential stream of batches, so the
        public surface is sync. Judge fan-out concurrency (for remote
        LLM judges) is handled inside ``_judge`` via ``run_sync``.
        """
        self.step_count += 1
        text_of = make_text_lookup(node_texts)
        context_of = candidate_context if candidate_context is not None else text_of
        agents = list(population.agents)
        if not query_ids or not pool_ids:
            return []
        k = min(agents[0].k, len(pool_ids))
        self._ensure_meta_state(agents)

        batch = self._propose(agents, query_embs, query_ids, pool_embs, pool_ids, k)
        scorable = list(batch.pair_counts.keys())
        judged = self._judge(
            scorable, batch.pair_counts, text_of, context_of, score_cache
        )
        # Inner loop: reward the arms, update their parameters.
        rewards = self._reward(agents, batch.by_agent, judged.judged_pairs, query_ids)
        backward_stats = self._backward(rewards, agents, batch.by_agent)

        # Outer loop: surprisal-credit the arms, update the mixture over them.
        meta_rewards = self._meta_reward(
            agents, batch.by_agent, query_ids, judged.judged_pairs
        )
        self._update_mixture(meta_rewards)

        # Capture the graph's field prediction for judged pairs BEFORE
        # the post-judge extension below, otherwise field returns the
        # judged y exactly and surprisal looks like 0.
        field_revealed: list[dict] = []
        if self._graph is not None:
            for (qid, cid), y in judged.judged_pairs:
                p = float(self._graph.field(qid, cid))
                a = float(y)
                field_revealed.append(
                    {
                        "pair": [qid, cid],
                        "predicted": p,
                        "actual": a,
                        "surprisal": abs(p - a),
                    }
                )

        # Train the TGN's link head + message encoder + GRU updater on
        # the judged events BEFORE extend, so memory propagation happens
        # under autograd in train_step (not as a side effect of extend).
        # This is the architectural hook that turns TGN from a passive
        # observer into a co-trained reasoner.
        tgn_loss: float = 0.0
        tgn_link_loss: float = 0.0
        tgn_align_loss: float = 0.0
        tgn = self._graph.tgn if self._graph is not None else None
        if (
            self._graph is not None
            and tgn is not None
            and self._tgn_optimizer is not None
            and judged.judged_pairs
        ):
            events: list[tuple[str, str, float, float, float, float]] = []
            for (u, v), y in judged.judged_pairs:
                sign = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
                self._edge_clock += 1.0
                events.append((u, v, sign, self._edge_clock, abs(float(y)), float(y)))

            # Collect current-graph neighbourhood *IDs* for each event
            # endpoint BEFORE extend() adds the new edges. We pass IDs (not
            # tensors) so train_step can look up live memory per event —
            # repeated nodes that get updated mid-batch contribute their
            # autograd-connected memory when they next appear as a neighbour.
            nbr_ids_by_node: dict[str, list[str]] = {}
            graph_nodes = self._graph._raw
            for (u, v), _ in judged.judged_pairs:
                for nid in (u, v):
                    if nid in nbr_ids_by_node:
                        continue
                    ids = [
                        n
                        for n, _w in self._graph.get_neighbors(nid)
                        if n in graph_nodes
                    ]
                    if ids:
                        nbr_ids_by_node[nid] = ids

            self._tgn_optimizer.zero_grad()
            try:
                loss = tgn.train_step(
                    events,
                    nbr_ids_by_node=nbr_ids_by_node or None,
                )
                tgn_link_loss = float(loss.item())
                align_weight = float(getattr(self.config, "tgn_rep_align_weight", 0.0))
                if align_weight > 0.0:
                    touched = sorted(
                        {u for (u, _v), _ in judged.judged_pairs}
                        | {v for (_u, v), _ in judged.judged_pairs}
                    )
                    align_ids, raw_targets = self._alignment_targets(
                        touched, query_embs.device
                    )
                    if raw_targets is not None:
                        align_loss = tgn.representation_alignment_loss(
                            align_ids, raw_targets
                        )
                        tgn_align_loss = float(align_loss.item())
                        loss = loss + align_weight * align_loss
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tgn.parameters(), max_norm=1.0)
                    self._tgn_optimizer.step()
                tgn_loss = float(loss.item())
            finally:
                # train_step writes memory via set_no_detach to keep the
                # autograd graph alive within the batch. Memory MUST be
                # detached at the step boundary or the next batch's get()
                # returns tensors attached to a dead graph (second-backward
                # RuntimeError). Run in finally so backward/optimizer
                # failures don't corrupt subsequent batches.
                tgn.detach_all_memory()
                self._graph.clear_nbr_mems_cache()

        # Train the TGN's link head + message encoder + GRU updater on
        # the judged events BEFORE extend, so memory propagation happens
        # under autograd in train_step (not as a side effect of extend).
        # This is the architectural hook that turns TGN from a passive
        # observer into a co-trained reasoner.
        tgn_loss: float = 0.0
        if (
            self._graph is not None
            and self._graph._tgn is not None
            and self._tgn_optimizer is not None
            and judged.judged_pairs
        ):
            base_t = float(self._graph._edge_count) + 1.0
            events: list[tuple[str, str, float, float, float, float]] = []
            for i, ((u, v), y) in enumerate(judged.judged_pairs):
                sign = 1.0 if y > 0 else (-1.0 if y < 0 else 0.0)
                events.append(
                    (u, v, sign, base_t + i, abs(float(y)), float(y))
                )
            self._tgn_optimizer.zero_grad()
            loss = self._graph._tgn.train_step(events)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._graph._tgn.parameters(), max_norm=1.0
                )
                self._tgn_optimizer.step()
            self._graph._tgn.detach_all_memory()
            tgn_loss = float(loss.item())

        # Post-judge impute: now that q has its first neighborhood (from
        # the judged edges), re-impute any pairs we skipped for budget.
        score_by_pair, stats = self._impute_after_judge(
            judged, scorable, query_ids, query_embs
        )

        # #7: signed AUC of graph.field (TGN predict_link, when use_tgn=true)
        # vs LLM judge labels — computed from field_revealed which captured
        # predictions BEFORE the graph was extended with these edges, so
        # this is a forward-only held-out signal per step.
        auc_pos, auc_neg = _auc_prediction_groups(field_revealed)
        self.last_step_stats = {
            "scorable": len(scorable),
            **stats,
            **backward_stats,
            "meta_rewards": meta_rewards,
            "sigma": self.sigma,
            "field_revealed": field_revealed,
            "tgn_loss": tgn_loss,
        }
        return self._results(query_ids, agents, batch.by_agent, score_by_pair, rewards)

    def _impute_after_judge(
        self,
        judged: JudgeResult,
        scorable: list[tuple[str, str]],
        query_ids: list[str],
        query_embs: torch.Tensor,
    ) -> tuple[dict[tuple[str, str], float], dict[str, int]]:
        """Extend the graph with the batch's judged edges (and the fresh
        query nodes they touch), then retry impute on pairs that were
        skipped for budget. Returns the updated score map and stats."""
        if self._graph is None or not judged.judged_pairs:
            return judged.score_by_pair, judged.stats

        query_embs_np = query_embs.detach().cpu().numpy().astype(np.float32)
        edges = [(qid, cid, float(y)) for (qid, cid), y in judged.judged_pairs]
        self._graph.extend(query_ids, query_embs_np, edges)

        score_by_pair = dict(judged.score_by_pair)
        # Batched re-imputation: one TGN conv pass + link-head forward scores
        # every still-unresolved pair, instead of a per-pair forward. The
        # graph applies identical gating (observed edge / cold raw_fallback /
        # confidence threshold) as the scalar impute().
        #
        # Note: if agents proposed both (a,b) and (b,a) and only one direction
        # was judged, the reverse resolves here via the newly committed edge
        # (direct lookup). This is intentional — NLI is directional so both
        # pairs are valid, and resolving the reverse from the observed edge is
        # cheaper than a second LLM call. post_judge_resolved counts these
        # direct lookups alongside genuine 2-hop / TGN results.
        pending = [p for p in scorable if p not in score_by_pair]
        new_imputed = 0
        if pending:
            for pair, v in self._graph.impute_batch(pending).items():
                if v is not None:
                    score_by_pair[pair] = v
                    new_imputed += 1

        stats = dict(judged.stats)
        stats["imputed"] = stats.get("imputed", 0) + new_imputed
        stats["skipped"] = max(0, stats.get("skipped", 0) - new_imputed)
        stats["post_judge_resolved"] = new_imputed
        return score_by_pair, stats
