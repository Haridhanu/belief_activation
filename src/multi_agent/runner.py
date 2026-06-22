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

import io as _io
import json
import logging
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
from multi_agent.snapshot import RouterSnapshot, SCHEMA_VERSION

logger = logging.getLogger(__name__)


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
    repr_divergence: float = 0.0
    post_judge_resolved: int = 0
    imputation_rate: float = 0.0
    mean_surprisal: float = 0.0
    tgn_loss: float = 0.0
    sigma: dict[str, float] = field(default_factory=dict)
    meta_rewards: dict[str, float] = field(default_factory=dict)
    field_revealed: list[dict[str, Any]] = field(default_factory=list)
    # #7: signed AUC of graph.field predictions vs LLM judge labels. None
    # when fewer than 10 labeled pairs or one class is empty.
    auc_signed: float | None = None
    auc_n_pos: int = 0
    auc_n_neg: int = 0

    def format(self) -> str:
        return (
            f"  step{self.step:4d}: "
            f"winner={self.winner_id} reward={self.reward:.3f} "
            f"loss={self.loss:+.3f} "
            f"nodes={self.n_nodes} "
            f"edges+={self.n_coh}coh/{self.n_dis}dis "
            f"judge={self.judged}/{self.scorable} "
            f"(cached {self.cached}, imputed {self.imputed}, skipped {self.skipped}) "
            f"impute_rate={self.imputation_rate:.3f} "
            f"surprisal={self.mean_surprisal:.3f} "
            f"tgn_loss={self.tgn_loss:.4f}"
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

        tgn = None
        tgn_optimizer: torch.optim.Optimizer | None = None
        if config.use_tgn:
            from multi_agent.tgn import TGNModule

            tgn = TGNModule(
                emb_dim=config.emb_dim,
                memory_dim=config.tgn_memory_dim,
                time_dim=config.tgn_time_dim,
                n_heads=config.tgn_n_attn_heads,
            ).to(torch.device(config.device))
            tgn_optimizer = torch.optim.Adam(tgn.parameters(), lr=config.tgn_lr)

        self.graph = Graph(
            emb_dim=config.emb_dim,
            _tgn=tgn,
            tgn_cold_start=config.tgn_cold_start,
            tgn_predict_threshold=config.tgn_predict_threshold,
        )
        self.tgn = tgn
        self.tgn_optimizer = tgn_optimizer

        if config.graph_substrate == "signed_hybrid":
            from multi_agent.signed_gnn import SignedGNN

            self.graph._sgnn = SignedGNN(
                in_dim=config.emb_dim,
                hidden=config.signed_gnn_hidden,
                layers=config.signed_gnn_layers,
            ).to(torch.device(config.device))
            self.graph.hybrid_density_threshold = config.hybrid_density_threshold

        self.loop = PSROLoop(
            config, judge=judge, graph=self.graph, tgn_optimizer=tgn_optimizer
        )
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
        # Register current-batch raw embeddings before PSRO scoring so
        # TGN raw_fallback can give field() a non-degenerate cold prior.
        # Cold impute() still defers to the judge instead of committing
        # raw-cosine edges.
        self.graph.extend(batch.ids, batch.embs, [])

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

        if self.config.graph_substrate == "signed_hybrid" and self.graph._sgnn is not None:
            self._refit_signed_gnn()

        raw = self.loop.last_step_stats
        field_revealed = list(raw.get("field_revealed", []))
        surprisals: list[float] = []
        for item in field_revealed:
            try:
                surprisal = float(item["surprisal"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(surprisal):
                surprisals.append(surprisal)
        scorable = int(raw.get("scorable", 0))

        repr_divergence = (
            self.graph.mean_representation_divergence() if self.tgn is not None else 0.0
        )
        raw["repr_divergence"] = repr_divergence
        if self.tgn is not None and self._step >= 10 and repr_divergence > 0.2:
            logger.warning(
                "TGN/raw representation divergence is high after %d steps: %.4f",
                self._step,
                repr_divergence,
            )
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
            scorable=scorable,
            cached=int(raw.get("cached", 0)),
            imputed=int(raw.get("imputed", 0)),
            skipped=int(raw.get("skipped", 0)),
            repr_divergence=float(raw.get("repr_divergence", 0.0)),
            post_judge_resolved=int(raw.get("post_judge_resolved", 0)),
            imputation_rate=float(raw.get("imputed", 0)) / max(scorable, 1),
            mean_surprisal=float(np.mean(surprisals)) if surprisals else 0.0,
            tgn_loss=float(raw.get("tgn_loss", 0.0)),
            sigma=dict(raw.get("sigma", {})),
            meta_rewards=dict(raw.get("meta_rewards", {})),
            field_revealed=field_revealed,
            auc_signed=raw.get("auc_signed"),
            auc_n_pos=int(raw.get("auc_n_pos", 0) or 0),
            auc_n_neg=int(raw.get("auc_n_neg", 0) or 0),
        )
        if self.config.use_tgn and stats.tgn_loss == 0.0 and self._step > 1:
            logger.warning(
                "WARNING: use_tgn=True but tgn_loss=0 — check judge_budget_per_batch"
            )
        self.history.append(stats)

        return StepResult(
            step=self._step,
            winner_id=winner_id,
            mean_rewards=mean_rewards,
            edges=edges,
            stats=stats,
        )

    def _refit_signed_gnn(self) -> None:
        """Refit the Signed-GNN on all committed signed edges and refresh the
        graph's id->row index + raw feature matrix used by ``_sgnn_predict``."""
        node_ids = self.graph.get_nodes()
        if not node_ids:
            return
        idx = {nid: i for i, nid in enumerate(node_ids)}
        feats = np.stack([self.graph._raw[nid][: self.config.emb_dim] for nid in node_ids])
        edges = [
            (idx[a], idx[b], float(w))
            for (a, b), w in self.graph._edges.items()
            if a in idx and b in idx
        ]
        self.graph._sgnn.fit(
            feats,
            edges,
            epochs=self.config.signed_gnn_epochs,
            lr=self.config.signed_gnn_lr,
        )
        self.graph._sgnn_index = idx
        self.graph._sgnn_features = feats

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
            out[agent.agent_id] = [(node_ids[i], float(scores[i])) for i in order]
        return out

    def to_snapshot(self, *, session_id: str) -> tuple[RouterSnapshot, bytes]:
        """Capture full Trainer state for cross-batch resume.

        Two artefacts: the JSON-serialisable `RouterSnapshot` and a
        torch-saved bytes blob containing the population state_dict and the
        per-agent Adam optimiser state_dicts.

        What's preserved: population weights, optimiser momentum, full Graph
        (raw embeddings, propagated z's, adjacency, signed edges, edge clock,
        edge timestamps, the four Bayesian hyperparameters), score_cache,
        node_texts, _meta_weights, step counter, agent stats
        (wins/rounds/cum_reward), trainer history.
        With TGN attached: TGN module weights, Adam momentum, per-node
        NodeMemory, and ``_ref_time`` — memory persists because it is
        path-dependent on the parameter trajectory (see the inline note in
        the weights-blob assembly below) and cannot be regenerated from
        edges alone.

        What's NOT preserved: random state (torch.manual_seed, np seed) — so
        proposals and gradient steps after a resume diverge from a continuous
        run, even though structural state matches.
        """
        for a, b in self.score_cache:
            if "|" in a or "|" in b:
                raise ValueError(
                    f"belief ID contains reserved separator '|': {a!r}, {b!r}"
                )
        for bid in self.graph._raw:
            if "|" in bid:
                raise ValueError(f"belief ID contains reserved separator '|': {bid!r}")

        cfg = self.config
        sigma = self.loop.sigma  # property; recomputed from _meta_weights

        if self.graph._tgn is not None:
            # mem_to_emb is trainable, so cached _z entries can go stale for
            # warm nodes that were not touched by the most recent Graph.extend.
            # Snapshot the current TGN-backed representation, not the cache.
            node_ids = self.graph.get_nodes()
            reps = self.graph.get_representations_fast(node_ids)
            graph_z = {
                bid: reps[i].astype(np.float32, copy=False)
                for i, bid in enumerate(node_ids)
            }
        else:
            graph_z = {
                bid: arr.astype(np.float32, copy=False)
                for bid, arr in self.graph._z.items()
            }
        graph_raw = {
            bid: arr.astype(np.float32, copy=False)
            for bid, arr in self.graph._raw.items()
        }
        # adjacency: dict[bid] -> set[bid]; serialise as sorted lists for stability
        graph_adj = {
            bid: sorted(neighbours) for bid, neighbours in self.graph._adj.items()
        }
        # edges: dict[(bid_a, bid_b)] -> weight; canonicalise key to "a|b" sorted
        graph_edges = {f"{a}|{b}": float(w) for (a, b), w in self.graph._edges.items()}
        # Graph.extend now records real edge clocks. The fallback keeps snapshots
        # loadable for legacy/incomplete in-memory Graph instances.
        graph_edge_timestamps = {
            f"{a}|{b}": int(self.graph._edge_timestamps.get((a, b), fallback_timestamp))
            for fallback_timestamp, ((a, b), _w) in enumerate(
                self.graph._edges.items(), start=1
            )
        }
        score_cache = {f"{a}|{b}": float(s) for (a, b), s in self.score_cache.items()}
        agent_pop_stats = {
            a.agent_id: {
                "wins": float(a.wins),
                "rounds": float(a.rounds),
                "cum_reward": float(a.cum_reward),
            }
            for a in self.population.agents
        }

        snap = RouterSnapshot(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            step=self._step,
            emb_dim=cfg.emb_dim,
            multi_agent_config={
                "emb_dim": cfg.emb_dim,
                "num_agents": cfg.num_agents,
                "k": cfg.k,
                "temperature": cfg.temperature,
                "agent_roles": dict(cfg.agent_roles or {}),
                "device": cfg.device,
                "use_tgn": cfg.use_tgn,
                "tgn_memory_dim": cfg.tgn_memory_dim,
                "tgn_time_dim": cfg.tgn_time_dim,
                "tgn_n_attn_heads": cfg.tgn_n_attn_heads,
                "tgn_lr": cfg.tgn_lr,
                "tgn_predict_threshold": cfg.tgn_predict_threshold,
                "tgn_rep_align_weight": cfg.tgn_rep_align_weight,
                "tgn_cold_start": cfg.tgn_cold_start,
            },
            graph_hyperparams={
                "attention_step": float(self.graph.attention_step),
                "prior_variance": float(self.graph.prior_variance),
                "obs_variance": float(self.graph.obs_variance),
                "confidence_floor": float(self.graph.confidence_floor),
            },
            bid_to_text=dict(self.node_texts),
            sigma={k: float(v) for k, v in sigma.items()},
            roles=dict(cfg.agent_roles or {}),
            history=[
                stats if isinstance(stats, dict) else asdict(stats)
                for stats in self.history
            ],
            meta_weights={k: float(v) for k, v in self.loop._meta_weights.items()},
            score_cache=score_cache,
            graph_z=graph_z,
            graph_raw=graph_raw,
            graph_adj=graph_adj,
            graph_edges=graph_edges,
            agent_pop_stats=agent_pop_stats,
            graph_edge_count=int(self.graph._edge_count),
            graph_edge_timestamps=graph_edge_timestamps,
        )

        weights_blob = {
            "state_dict": self.population.state_dict(),
            "optimizers": {
                aid: opt.state_dict()
                for aid, opt in self.loop._optimizers.items()
                if opt is not None
            },
        }
        # TGN parameters and per-node memory both persist via
        # ``TGNModule.state_dict``. Memory is path-dependent on the
        # parameter trajectory at the time each event was consumed, so it
        # cannot be regenerated from edges alone — dropping it would
        # silently break inference-only resumes. Callers that want a cold
        # session should call ``trainer.tgn.reset()`` explicitly.
        if self.tgn is not None:
            weights_blob["tgn"] = self.tgn.state_dict()
        if self.tgn_optimizer is not None:
            weights_blob["tgn_optimizer"] = self.tgn_optimizer.state_dict()
        weights_blob["_edge_clock"] = float(self.loop._edge_clock)
        buf = _io.BytesIO()
        torch.save(weights_blob, buf)
        return snap, buf.getvalue()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: RouterSnapshot,
        weights: bytes,
        judge: Judge,
    ) -> "Trainer":
        """Rebuild a Trainer at the state captured by `to_snapshot`."""
        from collections import defaultdict

        cfg_d = snapshot.multi_agent_config
        config = MultiAgentConfig(
            emb_dim=int(cfg_d["emb_dim"]),
            device=str(cfg_d.get("device", "cpu")),
            num_agents=int(cfg_d["num_agents"]),
            k=int(cfg_d["k"]),
            temperature=float(cfg_d.get("temperature", 0.3)),
            agent_roles=dict(cfg_d.get("agent_roles", {})),
            # TGN fields — defaults preserve baseline behavior for old
            # snapshots written before TGN persistence existed.
            use_tgn=bool(cfg_d.get("use_tgn", False)),
            tgn_memory_dim=int(cfg_d.get("tgn_memory_dim", 128)),
            tgn_time_dim=int(cfg_d.get("tgn_time_dim", 32)),
            tgn_n_attn_heads=int(cfg_d.get("tgn_n_attn_heads", 4)),
            tgn_lr=float(cfg_d.get("tgn_lr", 1e-3)),
            tgn_predict_threshold=float(cfg_d.get("tgn_predict_threshold", 0.2)),
            # Old snapshots predate alignment loss; resume them without
            # silently changing their optimization objective.
            tgn_rep_align_weight=float(cfg_d.get("tgn_rep_align_weight", 0.0)),
            tgn_cold_start=str(cfg_d.get("tgn_cold_start", "raw_fallback")),
        )
        trainer = cls(config, judge)

        # 1. Population weights (and optimizer state).
        loaded = torch.load(_io.BytesIO(weights), weights_only=True)
        trainer.population.load_state_dict(loaded["state_dict"])
        # Restore TGN parameters + optimizer momentum if both the snapshot
        # and the rebuilt trainer have TGN attached. Asymmetric cases
        # (only one side has TGN) are silently skipped — old snapshot
        # loaded into use_tgn=True trainer keeps freshly-initialised TGN;
        # new snapshot loaded into use_tgn=False trainer ignores the blob.
        if "tgn" in loaded and trainer.tgn is not None:
            trainer.tgn.load_state_dict(loaded["tgn"])
        if "tgn_optimizer" in loaded and trainer.tgn_optimizer is not None:
            trainer.tgn_optimizer.load_state_dict(loaded["tgn_optimizer"])
        if "_edge_clock" in loaded:
            trainer.loop._edge_clock = float(loaded["_edge_clock"])
        # Lazily build the optimizers on-demand the first time PSROLoop touches
        # them; here we just stash the saved state_dicts on the loop so it can
        # restore them when it constructs the optimisers.
        opt_states = loaded.get("optimizers", {}) or {}
        # Inference-only callers (those that only call .rank()) never construct
        # optimisers, so these stashed state_dicts are never consumed. They GC
        # with the Trainer; no leak.
        trainer.loop._pending_optimizer_states = dict(opt_states)

        # 2. Agent population stats (wins/rounds/cum_reward).
        for a in trainer.population.agents:
            stats = snapshot.agent_pop_stats.get(a.agent_id, {})
            a.wins = int(stats.get("wins", 0.0))
            a.rounds = int(stats.get("rounds", 0.0))
            a.cum_reward = float(stats.get("cum_reward", 0.0))

        # 3. Trainer-level state.
        trainer._step = int(snapshot.step)
        trainer.node_texts = dict(snapshot.bid_to_text)
        trainer.score_cache = {
            tuple(k.split("|", 1)): float(v) for k, v in snapshot.score_cache.items()
        }
        # history entries are kept as raw dicts; typed reconstruction is the
        # caller's responsibility.
        trainer.history = list(snapshot.history)

        # 4. PSROLoop meta-mixture.
        trainer.loop._meta_weights = {
            k: float(v) for k, v in snapshot.meta_weights.items()
        }
        trainer.loop.step_count = trainer._step

        # 5. Graph (raw embeddings, propagated z's, adjacency, edges).
        g = trainer.graph
        g._raw = {bid: arr.copy() for bid, arr in snapshot.graph_raw.items()}
        g._z = {bid: arr.copy() for bid, arr in snapshot.graph_z.items()}
        g._adj = defaultdict(set)
        for bid, neighbours in snapshot.graph_adj.items():
            g._adj[bid] = set(neighbours)
        g._edges = {}
        for key, w in snapshot.graph_edges.items():
            a, b = key.split("|", 1)
            g._edges[(min(a, b), max(a, b))] = float(w)
        graph_edge_timestamps = getattr(snapshot, "graph_edge_timestamps", {}) or {}
        g._edge_timestamps = {}
        # Older snapshots did not carry edge clocks, so reconstruct a stable
        # order only when the field is absent.
        for fallback_timestamp, key in enumerate(snapshot.graph_edges, start=1):
            a, b = key.split("|", 1)
            edge_key = (min(a, b), max(a, b))
            g._edge_timestamps[edge_key] = int(
                graph_edge_timestamps.get(key, fallback_timestamp)
            )
        g._edge_count = max(
            int(getattr(snapshot, "graph_edge_count", 0)),
            len(g._edges),
            max(g._edge_timestamps.values(), default=0),
        )
        g._z_tensor = None  # invalidate cached stack
        g._z_index = {}

        # 6. Graph Bayesian hyperparameters.
        hp = snapshot.graph_hyperparams
        g.attention_step = float(hp.get("attention_step", g.attention_step))
        g.prior_variance = float(hp.get("prior_variance", g.prior_variance))
        g.obs_variance = float(hp.get("obs_variance", g.obs_variance))
        g.confidence_floor = float(hp.get("confidence_floor", g.confidence_floor))

        return trainer


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
