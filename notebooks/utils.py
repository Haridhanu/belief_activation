"""Notebook-level utilities shared across all notebooks in this directory.

Wraps multi_agent internals behind a minimal interface so notebooks don't
need to know about package layout details.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.utils.notebook import make_cosine_judge

if TYPE_CHECKING:
    from multi_agent.judge import Judge
    from multi_agent.runner import EdgeSink


def _infer_emb_dim(batches: list[Batch]) -> int:
    if not batches:
        raise ValueError("run_belief_activation requires at least one batch")

    emb_dim: int | None = None
    for i, batch in enumerate(batches):
        embs = np.asarray(batch.embs)
        if embs.ndim != 2:
            raise ValueError(f"batch {i} embeddings must be a 2D array")
        if len(batch.ids) != embs.shape[0] or len(batch.texts) != embs.shape[0]:
            raise ValueError(
                f"batch {i} ids/texts must match embedding row count "
                f"({embs.shape[0]})"
            )
        if emb_dim is None:
            emb_dim = int(embs.shape[1])
        elif emb_dim != int(embs.shape[1]):
            raise ValueError(
                f"batch {i} embedding dimension {embs.shape[1]} does not match "
                f"prior dimension {emb_dim}"
            )

    assert emb_dim is not None
    return emb_dim


def _coerce_judge(
    batches: list[Batch],
    judge: Judge | int | float | Callable[[str, str], float] | None,
) -> Judge:
    if judge is None:
        return make_cosine_judge(batches)
    if isinstance(judge, (int, float)) or callable(judge):
        from multi_agent.judge import StaticJudge  # deferred: transitive torch dep

        return StaticJudge(judge)
    return judge


def run_belief_activation(
    batches: list[Batch],
    *,
    config: MultiAgentConfig | None = None,
    judge: Judge | int | float | Callable[[str, str], float] | None = None,
    sink: EdgeSink | None = None,
    log_every: int | None = None,
    device: str = "cpu",
    num_agents: int = 3,
    k: int = 8,
    **config_overrides: Any,
) -> dict[str, Any]:
    """Run the activation loop from a notebook with minimal setup.

    ``batches`` can come from ``make_synthetic_batches``,
    ``make_sentence_batches``, or any custom ``Batch`` stream. By default the
    helper uses a deterministic cosine judge keyed by batch text, so notebook
    demos work offline. Pass ``judge=LLMJudge(...)`` or any object satisfying
    the ``Judge`` protocol to use a real judge.

    ``config`` may be supplied directly. Otherwise the embedding dimension is
    inferred from the batches and a conservative CPU config is created. Keyword
    overrides are applied with ``dataclasses.replace`` semantics, so invalid
    config fields fail fast.

    Note: ``device``, ``num_agents``, and ``k`` only take effect when ``config``
    is not supplied. When ``config=`` is passed, use ``**config_overrides`` to
    override specific fields.
    """
    emb_dim = _infer_emb_dim(batches)
    resolved_config = config or MultiAgentConfig(
        emb_dim=emb_dim,
        device=device,
        num_agents=num_agents,
        k=k,
    )

    updates = dict(config_overrides)
    if log_every is not None:
        updates["log_every"] = log_every
    if updates:
        resolved_config = replace(resolved_config, **updates)

    if resolved_config.emb_dim != emb_dim:
        raise ValueError(
            f"config.emb_dim={resolved_config.emb_dim} does not match "
            f"batch embedding dimension {emb_dim}"
        )

    resolved_judge = _coerce_judge(batches, judge)

    from multi_agent.runner import run  # deferred: transitive torch dep

    result = run(resolved_config, resolved_judge, batches, sink=sink)
    result["config"] = resolved_config
    result["judge"] = resolved_judge
    result["final_sigma"] = dict(result["loop"].sigma)
    return result
