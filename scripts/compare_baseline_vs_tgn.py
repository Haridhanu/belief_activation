"""Synthetic comparison: baseline (use_tgn=False) vs TGN+BlendedImputer.

Builds 18 belief embeddings in 3 well-separated clusters with deterministic
ground truth (same-cluster pairs ⇒ +0.8, cross-cluster ⇒ -0.8) and a
``StaticJudge`` that returns those values. Streams the beliefs as 6 batches
of 3 through a Trainer with a fixed judge budget per step.

Reports, per config (mean over N seeds):
- Total judge calls actually consumed
- Total pairs resolved by the graph (cached + imputed)
- Total pairs skipped (scorable but neither cached, imputed, nor judged)
- Held-out imputation accuracy: for every (q, c) pair *not* committed as an
  edge by end of run, sign(graph.field(q, c)) vs sign(ground_truth(q, c))

Run from repo root:
    uv run python scripts/compare_baseline_vs_tgn.py
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import StaticJudge
from multi_agent.runner import Trainer


EMB_DIM = 32
N_CLUSTERS = 3
PER_CLUSTER = 6
BATCH_SIZE = 3
JUDGE_BUDGET = 5
COHERENCE_Y = 0.8
CONTRADICT_Y = -0.8


def _build_world(seed: int) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """3 clusters × PER_CLUSTER beliefs each. Returns (ids, embs, cluster_of)."""
    rng = np.random.default_rng(seed)
    cluster_dirs = []
    for c in range(N_CLUSTERS):
        v = rng.standard_normal(EMB_DIM).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        cluster_dirs.append(v)

    ids: list[str] = []
    embs: list[np.ndarray] = []
    cluster_of: dict[str, int] = {}
    for c, mean in enumerate(cluster_dirs):
        for i in range(PER_CLUSTER):
            nid = f"c{c}_{i}"
            noise = rng.standard_normal(EMB_DIM).astype(np.float32) * 0.15
            v = mean + noise
            v /= np.linalg.norm(v) or 1.0
            ids.append(nid)
            embs.append(v)
            cluster_of[nid] = c
    return ids, np.stack(embs), cluster_of


def _ground_truth(cluster_of: dict[str, int], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return COHERENCE_Y if cluster_of[a] == cluster_of[b] else CONTRADICT_Y


def _judge_for(cluster_of: dict[str, int]) -> StaticJudge:
    """StaticJudge that returns ground-truth y for any pair given its texts (=ids)."""

    def score(q: str, c: str) -> float:
        if q in cluster_of and c in cluster_of:
            return _ground_truth(cluster_of, q, c)
        return 0.0

    return StaticJudge(score)


@dataclass
class RunStats:
    label: str
    seed: int
    total_scorable: int
    total_judged: int
    total_imputed: int
    total_cached: int
    total_skipped: int
    n_committed_edges: int
    held_out_accuracy: float
    held_out_n: int


def _run_one(
    *, label: str, seed: int, use_tgn: bool, ids: list[str], embs: np.ndarray,
    cluster_of: dict[str, int], steps: int = 1,
) -> RunStats:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=3,
        k=4,
        seed=seed,
        agent_roles={"agent_0": "coherence", "agent_1": "contradiction", "cosine": "semantic"},
        judge_budget_per_batch=JUDGE_BUDGET,
        use_tgn=use_tgn,
        tgn_memory_dim=32,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
    )
    trainer = Trainer(cfg, _judge_for(cluster_of))

    total = {"scorable": 0, "judged": 0, "imputed": 0, "cached": 0, "skipped": 0}
    for epoch in range(steps):
        for s in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[s : s + BATCH_SIZE]
            batch_embs = embs[s : s + BATCH_SIZE]
            batch = Batch(ids=batch_ids, embs=batch_embs, texts=batch_ids)
            res = trainer.step(batch)
            for k in total:
                total[k] += getattr(res.stats, k)

    # Held-out accuracy: every (q, c) pair *not* committed as an edge.
    correct = 0
    n_held = 0
    edge_keys = set(trainer.graph._edges.keys())
    for i, q in enumerate(ids):
        for c in ids[i + 1 :]:
            if (q, c) in edge_keys or (c, q) in edge_keys:
                continue
            yhat = trainer.graph.field(q, c)
            y_true = _ground_truth(cluster_of, q, c)
            if np.sign(yhat) == np.sign(y_true) and abs(yhat) > 1e-6:
                correct += 1
            n_held += 1

    return RunStats(
        label=label,
        seed=seed,
        total_scorable=total["scorable"],
        total_judged=total["judged"],
        total_imputed=total["imputed"],
        total_cached=total["cached"],
        total_skipped=total["skipped"],
        n_committed_edges=len(trainer.graph._edges),
        held_out_accuracy=correct / max(1, n_held),
        held_out_n=n_held,
    )


def _summarise(runs: list[RunStats]) -> dict[str, float]:
    """Mean over runs of every numeric field."""
    keys = [
        "total_scorable",
        "total_judged",
        "total_imputed",
        "total_cached",
        "total_skipped",
        "n_committed_edges",
        "held_out_accuracy",
        "held_out_n",
    ]
    return {k: float(np.mean([getattr(r, k) for r in runs])) for k in keys}


def _print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    keys = [
        "total_scorable",
        "total_judged",
        "total_imputed",
        "total_cached",
        "total_skipped",
        "n_committed_edges",
        "held_out_n",
        "held_out_accuracy",
    ]
    width_label = max(len(label) for label, _ in rows) + 2
    width_col = 14
    header = "metric".ljust(24) + "".join(label.rjust(width_col) for label, _ in rows)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = k.ljust(24)
        for _, summary in rows:
            v = summary[k]
            cell = f"{v:.4f}" if "accuracy" in k else f"{v:.1f}"
            row += cell.rjust(width_col)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds per config")
    parser.add_argument("--epochs", type=int, default=2, help="passes through the batch stream")
    args = parser.parse_args()

    print(
        f"Synthetic comparison: {N_CLUSTERS} clusters × {PER_CLUSTER} = "
        f"{N_CLUSTERS * PER_CLUSTER} beliefs, batch={BATCH_SIZE}, "
        f"judge_budget={JUDGE_BUDGET}, seeds={args.seeds}, epochs={args.epochs}"
    )
    print()

    base_runs: list[RunStats] = []
    tgn_runs: list[RunStats] = []
    for seed in range(args.seeds):
        ids, embs, cluster_of = _build_world(seed=1000 + seed)
        base_runs.append(
            _run_one(
                label="baseline",
                seed=seed,
                use_tgn=False,
                ids=ids,
                embs=embs,
                cluster_of=cluster_of,
                steps=args.epochs,
            )
        )
        tgn_runs.append(
            _run_one(
                label="tgn",
                seed=seed,
                use_tgn=True,
                ids=ids,
                embs=embs,
                cluster_of=cluster_of,
                steps=args.epochs,
            )
        )

    base_summary = _summarise(base_runs)
    tgn_summary = _summarise(tgn_runs)
    _print_table([("baseline", base_summary), ("tgn", tgn_summary)])

    print()
    print("delta (tgn - baseline):")
    for k in [
        "total_judged",
        "total_imputed",
        "total_cached",
        "total_skipped",
        "n_committed_edges",
        "held_out_accuracy",
    ]:
        d = tgn_summary[k] - base_summary[k]
        sign = "+" if d >= 0 else ""
        if "accuracy" in k:
            print(f"  {k}: {sign}{d:.4f}")
        else:
            print(f"  {k}: {sign}{d:.2f}")


if __name__ == "__main__":
    main()
