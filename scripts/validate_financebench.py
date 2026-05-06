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
from multi_agent.tgn_runner import TGNTrainer
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
    total_wall_time_s: float
    mean_step_ms: float
    p95_step_ms: float
    time_per_judge_call_s: float


def _run(
    *,
    label: str,
    engine: str,                 # "baseline" | "tgn_imputer" | "tgn_only"
    judge,
    batches: list[Batch],
    seed: int,
    epochs: int,
    judge_budget: int,
) -> tuple[RunSummary, Trainer | TGNTrainer]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_tgn = engine == "tgn_imputer"
    config_engine = "tgn_only" if engine == "tgn_only" else "psro"
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
        engine=config_engine,
        tgn_memory_dim=64,
        tgn_time_dim=16,
        tgn_n_attn_heads=2,
    )
    trainer: Trainer | TGNTrainer
    if engine == "tgn_only":
        trainer = TGNTrainer(cfg, judge)
    else:
        trainer = Trainer(cfg, judge)

    total = {"scorable": 0, "judged": 0, "imputed": 0, "cached": 0, "skipped": 0}
    losses: list[float] = []
    step_times_ms: list[float] = []
    t_run = time.time()
    for _ in range(epochs):
        for batch in batches:
            t_step = time.perf_counter()
            res = trainer.step(batch)
            step_times_ms.append((time.perf_counter() - t_step) * 1000.0)
            for k in total:
                total[k] += getattr(res.stats, k)
            if engine == "tgn_imputer":
                losses.append(
                    float(trainer.loop.last_step_stats.get("imputer_loss", 0.0))  # type: ignore[union-attr]
                )
            elif engine == "tgn_only":
                losses.append(float(res.stats.loss))
    total_wall_time_s = time.time() - t_run

    if losses and engine in ("tgn_imputer", "tgn_only"):
        head = np.mean(losses[: max(1, len(losses) // 3)])
        tail = np.mean(losses[-max(1, len(losses) // 3) :])
        loss_label = "imputer_loss" if engine == "tgn_imputer" else "link_loss"
        print(
            f"  {loss_label} head→tail: {head:.4f} → {tail:.4f}  "
            f"(n_steps={len(losses)})"
        )

    held: dict[tuple[str, str], float] = {}
    edge_keys = set(trainer.graph._edges.keys())
    nodes = trainer.graph.get_nodes()
    for i, q in enumerate(nodes):
        for c in nodes[i + 1 :]:
            if (q, c) in edge_keys or (c, q) in edge_keys:
                continue
            if engine == "tgn_only":
                # No graph.field equivalent — use TGN's predict_link directly.
                held[(q, c)] = float(trainer.tgn.predict_link(q, c))  # type: ignore[union-attr]
            else:
                held[(q, c)] = float(trainer.graph.field(q, c))

    mean_step_ms = float(np.mean(step_times_ms)) if step_times_ms else 0.0
    p95_step_ms = float(np.percentile(step_times_ms, 95)) if step_times_ms else 0.0
    time_per_judge_call_s = (
        total_wall_time_s / total["judged"] if total["judged"] > 0 else 0.0
    )

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
            total_wall_time_s=total_wall_time_s,
            mean_step_ms=mean_step_ms,
            p95_step_ms=p95_step_ms,
            time_per_judge_call_s=time_per_judge_call_s,
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

    print("Running BASELINE (engine=baseline)…")
    base, _ = _run(
        label="baseline",
        engine="baseline",
        judge=judge,
        batches=batches,
        seed=args.seed,
        epochs=args.epochs,
        judge_budget=args.judge_budget,
    )
    print(f"  baseline run: {base.total_wall_time_s:.1f}s")
    print()

    print("Running TGN+BlendedImputer (engine=tgn_imputer)…")
    imputer, _ = _run(
        label="tgn_imputer",
        engine="tgn_imputer",
        judge=judge,
        batches=batches,
        seed=args.seed,
        epochs=args.epochs,
        judge_budget=args.judge_budget,
    )
    print(f"  tgn_imputer run: {imputer.total_wall_time_s:.1f}s")
    print()

    print("Running TGN-only (engine=tgn_only)…")
    tgn_only, _ = _run(
        label="tgn_only",
        engine="tgn_only",
        judge=judge,
        batches=batches,
        seed=args.seed,
        epochs=args.epochs,
        judge_budget=args.judge_budget,
    )
    print(f"  tgn_only run: {tgn_only.total_wall_time_s:.1f}s")
    print()

    print("=" * 88)
    print(f"FinanceBench {question.short_id} — {question.label}")
    print(f"{n_beliefs} beliefs, {len(batches)} batches × {args.epochs} epochs, "
          f"judge_budget={args.judge_budget}, seed={args.seed}")
    print("=" * 88)
    print()

    runs = [("baseline", base), ("tgn_imputer", imputer), ("tgn_only", tgn_only)]
    width_col = 16
    print("metric".ljust(24) + "".join(label.rjust(width_col) for label, _ in runs))
    print("-" * (24 + width_col * 3))
    for k, label in [
        ("total_scorable", "scorable"),
        ("total_judged", "judged"),
        ("total_imputed", "imputed"),
        ("total_cached", "cached"),
        ("total_skipped", "skipped"),
        ("n_committed_edges", "edges"),
        ("total_wall_time_s", "wall_time_s"),
        ("mean_step_ms", "mean_step_ms"),
        ("p95_step_ms", "p95_step_ms"),
        ("time_per_judge_call_s", "s_per_judge_call"),
    ]:
        row = label.ljust(24)
        for _, r in runs:
            v = getattr(r, k)
            if "step_ms" in k:
                row += f"{v:.1f}".rjust(width_col)
            elif k == "total_wall_time_s":
                row += f"{v:.1f}".rjust(width_col)
            elif k == "time_per_judge_call_s":
                row += f"{v:.3f}".rjust(width_col)
            else:
                row += f"{v}".rjust(width_col)
        print(row)
    print()

    if args.gt_cap > 0 and args.judge == "nli":
        common = (
            set(base.held_out_predictions)
            & set(imputer.held_out_predictions)
            & set(tgn_only.held_out_predictions)
        )
        print(
            f"Common held-out (all three configs): {len(common)} pairs. "
            f"Sampling up to {args.gt_cap} for NLI ground-truth evaluation…"
        )
        t0 = time.time()
        gt = _ground_truth_via_nli(common, text_of, judge, args.gt_cap)
        print(f"  ground-truth eval: {time.time() - t0:.1f}s")

        b_acc = i_acc = t_acc = 0
        b_mae = i_mae = t_mae = 0.0
        n = 0
        for key, y_true in gt.items():
            y_b = base.held_out_predictions[key]
            y_i = imputer.held_out_predictions[key]
            y_t = tgn_only.held_out_predictions[key]
            if abs(y_true) < 1e-3:
                continue  # NLI ambiguous → skip
            n += 1
            if np.sign(y_b) == np.sign(y_true) and abs(y_b) > 1e-6:
                b_acc += 1
            if np.sign(y_i) == np.sign(y_true) and abs(y_i) > 1e-6:
                i_acc += 1
            if np.sign(y_t) == np.sign(y_true) and abs(y_t) > 1e-6:
                t_acc += 1
            b_mae += abs(y_b - y_true)
            i_mae += abs(y_i - y_true)
            t_mae += abs(y_t - y_true)
        if n > 0:
            print()
            print(f"NLI-graded held-out (n={n}):")
            print(f"  baseline    sign accuracy: {b_acc / n:.3f}   MAE: {b_mae / n:.3f}")
            print(f"  tgn_imputer sign accuracy: {i_acc / n:.3f}   MAE: {i_mae / n:.3f}")
            print(f"  tgn_only    sign accuracy: {t_acc / n:.3f}   MAE: {t_mae / n:.3f}")


if __name__ == "__main__":
    main()
