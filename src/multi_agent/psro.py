from __future__ import annotations

from typing import Callable

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


class PSROLoop:

    def __init__(
        self,
        config: MultiAgentConfig,
        judge: Judge | None = None,
        graph: Graph | None = None,
    ) -> None:
        self.config = config
        self.step_count: int = 0
        self._optimizers: dict = {}
        self.judge: Judge = judge if judge is not None else StaticJudge(0.0)
        self._graph = graph
        self.last_step_stats: dict[str, int] = {
            "imputed": 0,
            "judged": 0,
            "scorable": 0,
        }
        # Multiplicative meta-weights over strategies; agent ids lazy-filled
        # on first ``step`` call. ``sigma`` derives a distribution from this.
        self._meta_weights: dict[str, float] = {}

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

        ``CosineActor`` has no params — we cache ``None`` so we don't try
        to re-construct an empty optimizer every step.
        """
        if agent.agent_id not in self._optimizers:
            params = list(agent.parameters())
            self._optimizers[agent.agent_id] = (
                torch.optim.Adam(params, lr=self.config.learning_rate)
                if params
                else None
            )
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
            scores = by_agent[agent.agent_id].scores  # (B, N)
            indices = by_agent[agent.agent_id].indices  # (B, k)
            log_probs_picked = scores.log_softmax(dim=1).gather(1, indices).sum(dim=1)
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
        field_revealed: list[tuple[float, float]] = (
            [
                (self._graph.field(qid, cid), float(y))
                for (qid, cid), y in judged.judged_pairs
            ]
            if self._graph is not None
            else []
        )

        # Post-judge impute: now that q has its first neighborhood (from
        # the judged edges), re-impute any pairs we skipped for budget.
        score_by_pair, stats = self._impute_after_judge(
            judged, scorable, query_ids, query_embs
        )

        self.last_step_stats = {
            "scorable": len(scorable),
            **stats,
            **backward_stats,
            "meta_rewards": meta_rewards,
            "sigma": self.sigma,
            "field_revealed": field_revealed,
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
        new_imputed = 0
        for pair in scorable:
            if pair in score_by_pair:
                continue
            v = self._graph.impute(*pair)
            if v is not None:
                score_by_pair[pair] = v
                new_imputed += 1

        stats = dict(judged.stats)
        stats["imputed"] = stats.get("imputed", 0) + new_imputed
        stats["skipped"] = max(0, stats.get("skipped", 0) - new_imputed)
        return score_by_pair, stats
