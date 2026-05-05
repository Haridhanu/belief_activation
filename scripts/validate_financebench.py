"""Validate baseline-vs-TGN on a real FinanceBench question.

Picks a prose-heavy question, sentence-splits its evidence into atomic
claims, embeds them with MiniLM (sentence-transformers), then runs the same
batch stream through ``Trainer`` twice — once with ``use_tgn=False`` and
once with ``use_tgn=True`` — sharing a single ``NLIJudge`` so judge calls
are deterministic across configs.

Reports:
- Total judge calls used, imputed, skipped, committed edges, scorable
- imputer_loss progression (only meaningful for TGN config)
- On the *common held-out* set (pairs neither config committed), NLI is
  consulted as ground truth and we report sign accuracy + MAE for each
  config's ``graph.field`` prediction. NLI calls on held-out pairs are
  capped via ``--gt-cap``.

This script downloads MiniLM (~80MB) and DeBERTa-large-mnli (~180MB) on
first run.

Examples:
    uv run python scripts/validate_financebench.py
    uv run python scripts/validate_financebench.py --qid 01290 --gt-cap 50
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import NLIJudge, StaticJudge
from multi_agent.runner import Trainer
from multi_agent.utils.financebench import (
    FinanceBenchQuestion,
    load_financebench,
    make_financebench_batches,
    prose_questions,
)


@dataclass
class RunSummary:
    label: str
    total_scorable: int
    total_judged: int
    total_imputed: int
    total_cached: int
    total_skipped: int
    n_committed_edges: int
    held_out_predictions: dict[tuple[str, str], float]


def _run(
    *,
    label: str,
    use_tgn: bool,
    judge,
    batches: list[Batch],
    seed: int,
    epochs: int,
    judge_budget: int,
) -> tuple[RunSummary, Trainer]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = MultiAgentConfig(
        emb_dim=batches[0].embs.shape[1],
        num_agents=3,
        k=8,
        seed=seed,
        agent_roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        judge_budget_per_batch=judge_budget,
        use_tgn=use_tgn,
        tgn_memory_dim=64,
        tgn_time_dim=16,
        tgn_n_attn_heads=2,
    )
    trainer = Trainer(cfg, judge)

    total = {"scorable": 0, "judged": 0, "imputed": 0, "cached": 0, "skipped": 0}
    losses: list[float] = []
    for ep in range(epochs):
        for batch in batches:
            res = trainer.step(batch)
            for k in total:
                total[k] += getattr(res.stats, k)
            if use_tgn:
                losses.append(
                    float(trainer.loop.last_step_stats.get("imputer_loss", 0.0))
                )

    if use_tgn and losses:
        head = np.mean(losses[: max(1, len(losses) // 3)])
        tail = np.mean(losses[-max(1, len(losses) // 3) :])
        print(
            f"  imputer_loss head→tail: {head:.4f} → {tail:.4f}  "
            f"(n_steps={len(losses)})"
        )

    held: dict[tuple[str, str], float] = {}
    edge_keys = set(trainer.graph._edges.keys())
    nodes = trainer.graph.get_nodes()
    for i, q in enumerate(nodes):
        for c in nodes[i + 1 :]:
            if (q, c) in edge_keys or (c, q) in edge_keys:
                continue
            held[(q, c)] = float(trainer.graph.field(q, c))

    return (
        RunSummary(
            label=label,
            total_scorable=total["scorable"],
            total_judged=total["judged"],
            total_imputed=total["imputed"],
            total_cached=total["cached"],
            total_skipped=total["skipped"],
            n_committed_edges=len(trainer.graph._edges),
            held_out_predictions=held,
        ),
        trainer,
    )


def _ground_truth_via_nli(
    pairs: Iterable[tuple[str, str]],
    text_of: dict[str, str],
    judge: NLIJudge,
    cap: int,
) -> dict[tuple[str, str], float]:
    """Use the NLI judge to score a sample of pairs as ground truth.

    Symmetrised (max-by-abs of both directions), exactly like the PSRO loop.
    """
    pairs = list(pairs)
    if cap > 0 and len(pairs) > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=cap, replace=False)
        pairs = [pairs[i] for i in idx]

    out: dict[tuple[str, str], float] = {}

    async def run_all() -> None:
        for q, c in pairs:
            yq = await judge.score(text_of[q], text_of[c])
            yc = await judge.score(text_of[c], text_of[q])
            out[(q, c)] = float(max(yq, yc, key=abs))

    asyncio.run(run_all())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", type=str, default=None, help="short id, e.g. 02024")
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--judge-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gt-cap",
        type=int,
        default=40,
        help="max NLI calls to evaluate held-out predictions",
    )
    parser.add_argument(
        "--judge",
        choices=["nli", "static"],
        default="nli",
        help="static is for offline smoke-testing only",
    )
    args = parser.parse_args()

    print("Loading FinanceBench…")
    questions = load_financebench()
    by_short = {q.short_id: q for q in questions}
    if args.qid is not None:
        if args.qid not in by_short:
            raise SystemExit(f"qid {args.qid} not found")
        question: FinanceBenchQuestion = by_short[args.qid]
    else:
        question = prose_questions(questions, min_beliefs=15)[3]  # Verizon 2024-pension
    print(
        f"Question: {question.short_id} — {question.company} {question.doc_period} "
        f"{question.doc_type}"
    )
    print(f"  {question.question}")
    print()

    print("Building belief batches (sentence-split + MiniLM embeddings)…")
    t0 = time.time()
    batches = make_financebench_batches(question, n_batches=args.n_batches, seed=args.seed)
    n_beliefs = sum(len(b.ids) for b in batches)
    print(f"  → {n_beliefs} beliefs in {len(batches)} batches  ({time.time() - t0:.1f}s)")
    print()

    text_of: dict[str, str] = {}
    for b in batches:
        text_of.update(dict(zip(b.ids, b.texts)))

    if args.judge == "nli":
        print("Loading NLI judge (first call may download DeBERTa-large)…")
        judge = NLIJudge()
    else:
        judge = StaticJudge(0.0)

    print("Running BASELINE (use_tgn=False)…")
    t0 = time.time()
    base, base_trainer = _run(
        label="baseline",
        use_tgn=False,
        judge=judge,
        batches=batches,
        seed=args.seed,
        epochs=args.epochs,
        judge_budget=args.judge_budget,
    )
    print(f"  baseline run: {time.time() - t0:.1f}s")
    print()

    print("Running TGN+BlendedImputer (use_tgn=True)…")
    t0 = time.time()
    tgn, tgn_trainer = _run(
        label="tgn",
        use_tgn=True,
        judge=judge,
        batches=batches,
        seed=args.seed,
        epochs=args.epochs,
        judge_budget=args.judge_budget,
    )
    print(f"  tgn run: {time.time() - t0:.1f}s")
    print()

    print("=" * 72)
    print(f"FinanceBench {question.short_id} — {question.label}")
    print(f"{n_beliefs} beliefs, {len(batches)} batches × {args.epochs} epochs, "
          f"judge_budget={args.judge_budget}, seed={args.seed}")
    print("=" * 72)
    print()

    width_col = 14
    print("metric".ljust(24) + "baseline".rjust(width_col) + "tgn".rjust(width_col)
          + "Δ".rjust(width_col))
    print("-" * (24 + width_col * 3))
    for k, label in [
        ("total_scorable", "scorable"),
        ("total_judged", "judged"),
        ("total_imputed", "imputed"),
        ("total_cached", "cached"),
        ("total_skipped", "skipped"),
        ("n_committed_edges", "edges"),
    ]:
        b = getattr(base, k)
        t = getattr(tgn, k)
        d = t - b
        sign = "+" if d >= 0 else ""
        print(f"{label}".ljust(24) + f"{b}".rjust(width_col)
              + f"{t}".rjust(width_col) + f"{sign}{d}".rjust(width_col))
    print()

    if args.gt_cap > 0 and args.judge == "nli":
        common = set(base.held_out_predictions) & set(tgn.held_out_predictions)
        print(
            f"Common held-out: {len(common)} pairs. Sampling up to "
            f"{args.gt_cap} for NLI ground-truth evaluation…"
        )
        t0 = time.time()
        gt = _ground_truth_via_nli(common, text_of, judge, args.gt_cap)
        print(f"  ground-truth eval: {time.time() - t0:.1f}s")

        b_acc = t_acc = 0
        b_mae = t_mae = 0.0
        n = 0
        for key, y_true in gt.items():
            y_b = base.held_out_predictions[key]
            y_t = tgn.held_out_predictions[key]
            if abs(y_true) < 1e-3:
                continue  # NLI ambiguous → skip
            n += 1
            if np.sign(y_b) == np.sign(y_true) and abs(y_b) > 1e-6:
                b_acc += 1
            if np.sign(y_t) == np.sign(y_true) and abs(y_t) > 1e-6:
                t_acc += 1
            b_mae += abs(y_b - y_true)
            t_mae += abs(y_t - y_true)
        if n > 0:
            print()
            print(f"NLI-graded held-out (n={n}):")
            print(f"  baseline   sign accuracy: {b_acc / n:.3f}   MAE: {b_mae / n:.3f}")
            print(f"  tgn+imputer sign accuracy: {t_acc / n:.3f}   MAE: {t_mae / n:.3f}")
            print(
                f"  Δ accuracy: {(t_acc - b_acc) / n:+.4f}   "
                f"Δ MAE: {(t_mae - b_mae) / n:+.4f}"
            )


if __name__ == "__main__":
    main()
