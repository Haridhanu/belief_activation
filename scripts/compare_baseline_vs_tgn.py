"""Synthetic comparison: baseline (use_tgn=False) vs TGN+BlendedImputer.

Builds N belief embeddings in K clusters with parameterised noise on the
embeddings *and* on the judge's output. Streams the beliefs through a
Trainer with a fixed judge budget per step.

Reports, per config (mean ± std over ``--seeds`` runs):
- Total judge calls actually consumed
- Total pairs resolved by the graph (cached + imputed)
- Total pairs skipped (scorable but neither cached, imputed, nor judged)
- Held-out imputation accuracy: for every (q, c) pair *not* committed as an
  edge by end of run, sign(graph.field(q, c)) vs sign(ground_truth(q, c))
- Held-out MAE: mean |graph.field(q, c) - ground_truth(q, c)| on held-out

Run from repo root:
    uv run python scripts/compare_baseline_vs_tgn.py                  # 18 nodes, clean
    uv run python scripts/compare_baseline_vs_tgn.py --hard            # 50 nodes, noisy
    uv run python scripts/compare_baseline_vs_tgn.py --clusters 4 \\
        --per-cluster 12 --embed-noise 0.5 --judge-noise 0.25
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
COHERENCE_Y = 0.8
CONTRADICT_Y = -0.8


def _build_world(
    seed: int, *, n_clusters: int, per_cluster: int, embed_noise: float
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """K clusters × ``per_cluster`` beliefs each. Returns (ids, embs, cluster_of)."""
    rng = np.random.default_rng(seed)
    cluster_dirs = []
    for c in range(n_clusters):
        v = rng.standard_normal(EMB_DIM).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        cluster_dirs.append(v)

    ids: list[str] = []
    embs: list[np.ndarray] = []
    cluster_of: dict[str, int] = {}
    for c, mean in enumerate(cluster_dirs):
        for i in range(per_cluster):
            nid = f"c{c}_{i}"
            noise = rng.standard_normal(EMB_DIM).astype(np.float32) * embed_noise
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


def _judge_for(
    cluster_of: dict[str, int], *, judge_noise: float, seed: int
) -> StaticJudge:
    """StaticJudge returning ground-truth y plus optional Gaussian noise.

    The noise is keyed on the unordered pair so repeated calls on the same
    pair return the same value — otherwise the score cache would mask the
    noise. Output is clamped to ``[-1, 1]``.
    """
    rng = np.random.default_rng(seed)
    pair_noise: dict[tuple[str, str], float] = {}

    def _eps(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        if key not in pair_noise:
            pair_noise[key] = float(rng.standard_normal()) * judge_noise
        return pair_noise[key]

    def score(q: str, c: str) -> float:
        if q in cluster_of and c in cluster_of:
            y = _ground_truth(cluster_of, q, c)
            if judge_noise > 0.0:
                y = max(-1.0, min(1.0, y + _eps(q, c)))
            return y
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
    held_out_mae: float
    held_out_n: int
    # Mapping: (q, c) → (yhat, y_true) for every pair *not* committed.
    # Used downstream to compute a "common held-out" accuracy across configs.
    held_out_predictions: dict[tuple[str, str], tuple[float, float]] = None  # type: ignore


def _run_one(
    *,
    label: str,
    seed: int,
    use_tgn: bool,
    ids: list[str],
    embs: np.ndarray,
    cluster_of: dict[str, int],
    steps: int,
    batch_size: int,
    judge_budget: int,
    judge_noise: float,
) -> RunStats:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=3,
        k=min(6, len(ids)),
        seed=seed,
        agent_roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        judge_budget_per_batch=judge_budget,
        use_tgn=use_tgn,
        tgn_memory_dim=32,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
    )
    judge = _judge_for(cluster_of, judge_noise=judge_noise, seed=seed + 7919)
    trainer = Trainer(cfg, judge)

    total = {"scorable": 0, "judged": 0, "imputed": 0, "cached": 0, "skipped": 0}
    for _ in range(steps):
        for s in range(0, len(ids), batch_size):
            batch_ids = ids[s : s + batch_size]
            batch_embs = embs[s : s + batch_size]
            batch = Batch(ids=batch_ids, embs=batch_embs, texts=batch_ids)
            res = trainer.step(batch)
            for k in total:
                total[k] += getattr(res.stats, k)

    # Held-out: every (q, c) pair *not* committed as an edge.
    correct = 0
    abs_err_sum = 0.0
    n_held = 0
    held_out_predictions: dict[tuple[str, str], tuple[float, float]] = {}
    edge_keys = set(trainer.graph._edges.keys())
    for i, q in enumerate(ids):
        for c in ids[i + 1 :]:
            if (q, c) in edge_keys or (c, q) in edge_keys:
                continue
            yhat = trainer.graph.field(q, c)
            y_true = _ground_truth(cluster_of, q, c)
            if np.sign(yhat) == np.sign(y_true) and abs(yhat) > 1e-6:
                correct += 1
            abs_err_sum += abs(yhat - y_true)
            n_held += 1
            held_out_predictions[(q, c)] = (yhat, y_true)

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
        held_out_mae=abs_err_sum / max(1, n_held),
        held_out_n=n_held,
        held_out_predictions=held_out_predictions,
    )


KEYS = [
    "total_scorable",
    "total_judged",
    "total_imputed",
    "total_cached",
    "total_skipped",
    "n_committed_edges",
    "held_out_n",
    "held_out_accuracy",
    "held_out_mae",
]


def _summarise(runs: list[RunStats]) -> dict[str, tuple[float, float]]:
    """Per-key (mean, std) over runs."""
    out: dict[str, tuple[float, float]] = {}
    for k in KEYS:
        vals = [getattr(r, k) for r in runs]
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _print_table(rows: list[tuple[str, dict[str, tuple[float, float]]]]) -> None:
    width_col = 18
    header = "metric".ljust(24) + "".join(label.rjust(width_col) for label, _ in rows)
    print(header)
    print("-" * len(header))
    for k in KEYS:
        row = k.ljust(24)
        for _, summary in rows:
            mean, std = summary[k]
            if "accuracy" in k or "mae" in k:
                cell = f"{mean:.3f}±{std:.3f}"
            else:
                cell = f"{mean:.1f}±{std:.1f}"
            row += cell.rjust(width_col)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds per config")
    parser.add_argument("--epochs", type=int, default=2, help="passes through the batch stream")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--per-cluster", type=int, default=6)
    parser.add_argument("--embed-noise", type=float, default=0.15)
    parser.add_argument(
        "--judge-noise",
        type=float,
        default=0.0,
        help="Gaussian std added to ground-truth judge output (clamped to [-1,1])",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--judge-budget", type=int, default=5)
    parser.add_argument(
        "--hard",
        action="store_true",
        help="preset: 5 clusters × 10 nodes, embed_noise=0.4, judge_noise=0.2, "
        "batch=5, budget=8",
    )
    args = parser.parse_args()

    if args.hard:
        args.clusters = 5
        args.per_cluster = 10
        args.embed_noise = 0.4
        args.judge_noise = 0.2
        args.batch_size = 5
        args.judge_budget = 8

    n_nodes = args.clusters * args.per_cluster
    print(
        f"Synthetic comparison: {args.clusters} clusters × {args.per_cluster} = "
        f"{n_nodes} beliefs, batch={args.batch_size}, "
        f"judge_budget={args.judge_budget}, "
        f"embed_noise={args.embed_noise}, judge_noise={args.judge_noise}, "
        f"seeds={args.seeds}, epochs={args.epochs}"
    )
    print()

    base_runs: list[RunStats] = []
    tgn_runs: list[RunStats] = []
    for seed in range(args.seeds):
        ids, embs, cluster_of = _build_world(
            seed=1000 + seed,
            n_clusters=args.clusters,
            per_cluster=args.per_cluster,
            embed_noise=args.embed_noise,
        )
        common = dict(
            ids=ids,
            embs=embs,
            cluster_of=cluster_of,
            steps=args.epochs,
            batch_size=args.batch_size,
            judge_budget=args.judge_budget,
            judge_noise=args.judge_noise,
        )
        base_runs.append(_run_one(label="baseline", seed=seed, use_tgn=False, **common))
        tgn_runs.append(_run_one(label="tgn", seed=seed, use_tgn=True, **common))

    base_summary = _summarise(base_runs)
    tgn_summary = _summarise(tgn_runs)
    _print_table([("baseline", base_summary), ("tgn", tgn_summary)])

    # Fair head-to-head: pairs *both* runs leave uncommitted.
    print()
    print("common held-out (pairs neither config committed):")
    common_acc_base: list[float] = []
    common_acc_tgn: list[float] = []
    common_mae_base: list[float] = []
    common_mae_tgn: list[float] = []
    common_ns: list[int] = []
    for b, t in zip(base_runs, tgn_runs):
        common_keys = set(b.held_out_predictions) & set(t.held_out_predictions)
        if not common_keys:
            continue
        common_ns.append(len(common_keys))
        b_corr = b_err = t_corr = t_err = 0
        b_es = t_es = 0.0
        for k in common_keys:
            yhb, yt = b.held_out_predictions[k]
            yht, _ = t.held_out_predictions[k]
            if np.sign(yhb) == np.sign(yt) and abs(yhb) > 1e-6:
                b_corr += 1
            if np.sign(yht) == np.sign(yt) and abs(yht) > 1e-6:
                t_corr += 1
            b_es += abs(yhb - yt)
            t_es += abs(yht - yt)
        n = len(common_keys)
        common_acc_base.append(b_corr / n)
        common_acc_tgn.append(t_corr / n)
        common_mae_base.append(b_es / n)
        common_mae_tgn.append(t_es / n)
    if common_ns:
        print(f"  shared pairs (mean): {np.mean(common_ns):.1f}")
        print(
            f"  baseline accuracy: {np.mean(common_acc_base):.3f}±{np.std(common_acc_base):.3f}"
        )
        print(
            f"  tgn      accuracy: {np.mean(common_acc_tgn):.3f}±{np.std(common_acc_tgn):.3f}"
        )
        print(
            f"  baseline MAE:      {np.mean(common_mae_base):.3f}±{np.std(common_mae_base):.3f}"
        )
        print(
            f"  tgn      MAE:      {np.mean(common_mae_tgn):.3f}±{np.std(common_mae_tgn):.3f}"
        )
        print(
            f"  Δ accuracy (tgn-base): {np.mean(common_acc_tgn) - np.mean(common_acc_base):+.4f}"
        )
        print(
            f"  Δ MAE      (tgn-base): {np.mean(common_mae_tgn) - np.mean(common_mae_base):+.4f}"
        )

    print()
    print("delta (tgn - baseline, mean only):")
    for k in [
        "total_judged",
        "total_imputed",
        "total_cached",
        "total_skipped",
        "n_committed_edges",
        "held_out_accuracy",
        "held_out_mae",
    ]:
        d = tgn_summary[k][0] - base_summary[k][0]
        sign = "+" if d >= 0 else ""
        if "accuracy" in k or "mae" in k:
            print(f"  {k}: {sign}{d:.4f}")
        else:
            print(f"  {k}: {sign}{d:.2f}")


if __name__ == "__main__":
    main()
