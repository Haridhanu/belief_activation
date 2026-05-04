"""Streaming PSRO trainer.

The primary entry point is :class:`Trainer` — initialise it once with a
config + judge, then call :meth:`Trainer.step` per belief batch for as
long as you like. Each call returns a :class:`StepResult` carrying the
edges judged this step and per-step audit fields, so callers can wire
them into whatever observability sink they want.

:func:`run` is a thin convenience wrapper that drains a finite list of
batches and returns the aggregate result dict the notebook still expects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import numpy as np
import torch

from multi_agent.agent import AgentPopulation
from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.graph import Graph
from multi_agent.judge import Judge
from multi_agent.psro import PSROLoop


def _current_pool(graph: Graph, population, batch, query_embs):
    existing = graph.get_nodes()
    if not existing:
        return list(batch.ids), query_embs
    placed_ids = list(existing)
    placed_embs = population.embeddings_to_device(
        graph.get_representations_fast(placed_ids)
    )
    pool_ids = placed_ids + list(batch.ids)
    pool_embs = torch.cat([placed_embs, query_embs], dim=0)
    return pool_ids, pool_embs


def _judged_edges(
    step_results: list[dict], batch, graph: Graph
) -> list[tuple[str, str, float]]:
    graph_nodes = set(graph.get_nodes())
    batch_ids = set(batch.ids)
    directional: dict[tuple[str, str], list[float]] = {}
    for result in step_results:
        query_id = result["query_node"]
        if query_id not in batch_ids:
            continue
        for agent_id, proposals in result["proposals"].items():
            scores = result["proposal_scores"].get(agent_id, [])
            for neighbor_id, score in zip(proposals, scores):
                if neighbor_id == query_id or score == 0.0:
                    continue
                if neighbor_id not in batch_ids and neighbor_id not in graph_nodes:
                    continue
                key = tuple(sorted([query_id, neighbor_id]))
                directional.setdefault(key, []).append(float(score))
    return [(a, b, max(scores, key=abs)) for (a, b), scores in directional.items()]


@dataclass
class StepStats:
    step: int
    winner_id: str
    reward: float
    loss: float
    loss_spread: float
    per_agent_loss: dict[str, float]
    n_nodes: int
    n_coh: int
    n_dis: int
    judged: int
    scorable: int
    cached: int
    imputed: int
    skipped: int
    sigma: dict[str, float] = field(default_factory=dict)
    meta_rewards: dict[str, float] = field(default_factory=dict)
    field_revealed: list[tuple[float, float]] = field(default_factory=list)

    def format(self) -> str:
        return (
            f"  step{self.step:4d}: "
            f"winner={self.winner_id} reward={self.reward:.3f} "
            f"loss={self.loss:+.3f} "
            f"nodes={self.n_nodes} "
            f"edges+={self.n_coh}coh/{self.n_dis}dis "
            f"judge={self.judged}/{self.scorable} "
            f"(cached {self.cached}, imputed {self.imputed}, skipped {self.skipped})"
        )


@dataclass
class StepResult:
    """One batch's worth of work, in a form a sink can serialise."""

    step: int
    winner_id: str
    mean_rewards: dict[str, float]
    edges: list[tuple[str, str, float]]
    stats: StepStats


class EdgeSink(Protocol):
    def write(self, result: StepResult) -> None: ...


class JsonlEdgeLogger:
    """Append-only JSONL sink. One record per `Trainer.step` call.

    Use as a context manager so the file handle is flushed/closed cleanly.
    Field shape is intentionally flat so downstream tools (jq, pandas) can
    consume it without unwrapping.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "JsonlEdgeLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, result: StepResult) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlEdgeLogger used outside a `with` block")
        rec = {
            "step": result.step,
            "winner": result.winner_id,
            "mean_rewards": result.mean_rewards,
            "edges": [[a, b, w] for a, b, w in result.edges],
            "stats": asdict(result.stats),
        }
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()


class Trainer:
    """Stateful PSRO trainer. Init once, then call ``step(batch)`` forever.

    Holds the population, graph, PSRO loop, and audit caches across calls.
    Has no notion of epochs — the graph just grows as batches arrive.
    """

    def __init__(self, config: MultiAgentConfig, judge: Judge):
        self.config = config
        self.judge = judge
        self.population = AgentPopulation(config)
        self.graph = Graph(emb_dim=config.emb_dim)
        self.loop = PSROLoop(config, judge=judge, graph=self.graph)
        self.score_cache: dict[tuple[str, str], float] = {}
        self.node_texts: dict[str, str] = {}
        self.history: list[StepStats] = []
        self._step = 0

    def step(self, batch: Batch) -> StepResult:
        self._step += 1
        self.node_texts.update(dict(zip(batch.ids, batch.texts)))
        query_embs = self.population.embeddings_to_device(batch.embs)
        pool_ids, pool_embs = _current_pool(
            self.graph, self.population, batch, query_embs
        )

        step_results = self.loop.step(
            self.population,
            query_embs,
            batch.ids,
            pool_embs,
            pool_ids,
            node_texts=self.node_texts,
            score_cache=self.score_cache,
        )

        agent_ids = [a.agent_id for a in self.population.agents]
        mean_rewards = {
            aid: float(np.mean([r["rewards"][ai] for r in step_results]))
            for ai, aid in enumerate(agent_ids)
        }
        winner_id = max(mean_rewards, key=mean_rewards.get)
        for agent in self.population.agents:
            agent.rounds += 1
            agent.cum_reward += mean_rewards[agent.agent_id]
            if agent.agent_id == winner_id:
                agent.wins += 1

        edges = _judged_edges(step_results, batch, self.graph)
        n_coh = sum(1 for _, _, w in edges if w > 0)
        n_dis = sum(1 for _, _, w in edges if w < 0)
        self.graph.extend(batch.ids, batch.embs, edges)

        raw = self.loop.last_step_stats
        stats = StepStats(
            step=self._step,
            winner_id=winner_id,
            reward=mean_rewards[winner_id],
            loss=float(raw.get("loss", 0.0)),
            loss_spread=float(raw.get("loss_spread", 0.0)),
            per_agent_loss=dict(raw.get("per_agent_loss", {})),
            n_nodes=len(self.graph),
            n_coh=n_coh,
            n_dis=n_dis,
            judged=int(raw.get("judged", 0)),
            scorable=int(raw.get("scorable", 0)),
            cached=int(raw.get("cached", 0)),
            imputed=int(raw.get("imputed", 0)),
            skipped=int(raw.get("skipped", 0)),
            sigma=dict(raw.get("sigma", {})),
            meta_rewards=dict(raw.get("meta_rewards", {})),
            field_revealed=list(raw.get("field_revealed", [])),
        )
        self.history.append(stats)

        return StepResult(
            step=self._step,
            winner_id=winner_id,
            mean_rewards=mean_rewards,
            edges=edges,
            stats=stats,
        )

    def stream(
        self, batches: Iterable[Batch], sink: EdgeSink | None = None
    ) -> Iterator[StepResult]:
        """Yield a StepResult per batch; optionally tee to ``sink``."""
        for batch in batches:
            result = self.step(batch)
            if sink is not None:
                sink.write(result)
            yield result

    def rank(
        self, query_emb: np.ndarray, k: int | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """Score every graph node against ``query_emb`` per agent (no training).

        Returns ``{agent_id: [(node_id, score), ...]}`` sorted by score
        descending. ``k`` truncates each list (default: top-k from config).
        Use this to route an inference query through each role's trained
        policy — coherence, contradiction, semantic.
        """
        node_ids = self.graph.get_nodes()
        if not node_ids:
            return {a.agent_id: [] for a in self.population.agents}

        cand_embs = self.population.embeddings_to_device(
            self.graph.get_representations_fast(node_ids)
        )
        q = self.population.embeddings_to_device(np.asarray(query_emb))
        if q.ndim == 1:
            q_single = q
        else:
            q_single = q[0]

        top_k = k if k is not None else self.config.k
        out: dict[str, list[tuple[str, float]]] = {}
        for agent in self.population.agents:
            with torch.no_grad():
                raw = agent.score_candidates(q_single, cand_embs)
            scores = raw[0] if isinstance(raw, tuple) else raw
            scores = scores.detach().cpu().numpy()
            order = np.argsort(-scores)[:top_k]
            out[agent.agent_id] = [
                (node_ids[i], float(scores[i])) for i in order
            ]
        return out


def run(
    config: MultiAgentConfig,
    judge: Judge,
    beliefs: list[Batch],
    *,
    sink: EdgeSink | None = None,
) -> dict[str, Any]:
    """Bounded driver: replay ``beliefs`` for ``config.n_epochs`` epochs.

    Kept as the notebook-friendly aggregate API. Internally it just spins
    a fresh :class:`Trainer` per epoch (so epoch boundaries reset the
    graph, matching the prior behaviour) and drains it.
    """
    population: AgentPopulation | None = None
    graph: Graph | None = None
    loop: PSROLoop | None = None
    history: list[StepStats] = []
    epoch_boundaries: list[int] = []
    node_texts: dict[str, str] = {}

    for epoch in range(config.n_epochs):
        trainer = Trainer(config, judge)
        if population is None:
            population = trainer.population
        else:
            trainer.population = population
        for batch in beliefs:
            result = trainer.step(batch)
            if sink is not None:
                sink.write(result)
            if config.log_every and result.step % config.log_every == 0:
                print(result.stats.format())
        if epoch > 0:
            epoch_boundaries.append(len(history))
        history.extend(trainer.history)
        graph = trainer.graph
        loop = trainer.loop
        node_texts = trainer.node_texts

    assert population is not None and graph is not None and loop is not None
    return {
        "population": population,
        "graph": graph,
        "node_ids": graph.get_nodes(),
        "node_texts": node_texts,
        "step_history": history,
        "reward_history": [s.reward for s in history],
        "loss_history": [s.loss for s in history],
        "per_agent_loss_history": [s.per_agent_loss for s in history],
        "loss_spread_history": [s.loss_spread for s in history],
        "epoch_boundaries": epoch_boundaries,
        "winners": [s.winner_id for s in history],
        "loop": loop,
        "judge_calls": {
            "imputed": sum(s.imputed for s in history),
            "judged": sum(s.judged for s in history),
            "cached": sum(s.cached for s in history),
        },
    }
