from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

import torch

from multi_agent.utils.types import AgentProposal

if TYPE_CHECKING:
    from multi_agent.judge import Judge

T = TypeVar("T")


def safe_softmax(scores: torch.Tensor, temperature: float, dim: int) -> torch.Tensor:
    """Temperature-scaled softmax. Any row that produces NaN/Inf or sums to
    zero after cleanup is replaced with a uniform distribution along ``dim``."""
    probs = torch.softmax(scores / temperature, dim=dim)
    if not (torch.isnan(probs).any() or torch.isinf(probs).any()):
        return probs
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(dim=dim, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / probs.shape[dim])
    probs = torch.where(row_sums == 0, uniform, probs)
    return probs / probs.sum(dim=dim, keepdim=True)


def run_sync(coro: Awaitable[T]) -> T:
    """Run ``coro`` from sync code, even under a running loop (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


async def score_pairs(
    judge: Judge, pairs: list[tuple[str, str]], concurrency: int
) -> list[float]:
    """Score every pair. Uses ``judge.score_batch`` if the backend supports
    fused batched inference (e.g. local NLI); otherwise fans out individual
    ``score`` calls under a semaphore."""
    if not pairs:
        return []
    batch_fn = getattr(judge, "score_batch", None)
    if batch_fn is not None:
        return await batch_fn(pairs)
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str, c: str) -> float:
        async with sem:
            return await judge.score(q, c)

    return await asyncio.gather(*(_one(q, c) for q, c in pairs))


def make_text_lookup(
    node_texts: dict[str, str] | None,
) -> Callable[[str], str]:
    if node_texts is None:
        return lambda nid: nid
    return lambda nid: node_texts.get(nid, nid)


def build_self_mask(query_ids: list[str], pool_ids: list[str]) -> list[int | None]:
    """For each query, the pool column to mask (its own identifier), or None."""
    pool_idx_by_id = {pid: i for i, pid in enumerate(pool_ids)}
    return [pool_idx_by_id.get(qid) for qid in query_ids]


def score_and_sample_agent(
    agent,
    query_embs: torch.Tensor,
    pool_embs: torch.Tensor,
    pool_ids: list[str],
    self_cols: list[int | None],
    k: int,
) -> AgentProposal:
    """Score every (query, pool) pair for one agent, mask self-columns,
    sample ``k`` distinct proposals per query, and translate to identifiers."""
    scores = agent.score_candidates_batch(query_embs, pool_embs)  # (B, N)
    for qi, col in enumerate(self_cols):
        if col is not None:
            scores[qi, col] = float("-inf")
    probs = safe_softmax(scores, agent.temperature, dim=1)
    indices = torch.multinomial(probs, k, replacement=False)  # (B, k)
    proposals = [[pool_ids[j] for j in row] for row in indices.tolist()]
    return AgentProposal(scores=scores, indices=indices, proposals=proposals)


def accumulate_pair_counts(
    pair_counts: dict[tuple[str, str], int],
    query_ids: list[str],
    proposals: list[list[str]],
) -> None:
    """Add one vote per (query, proposal) pair this agent contributed,
    deduplicated so a single agent cannot double-count."""
    agent_pairs: set[tuple[str, str]] = set()
    for qi, qid in enumerate(query_ids):
        for p in proposals[qi]:
            agent_pairs.add((qid, p))
    for key in agent_pairs:
        pair_counts[key] = pair_counts.get(key, 0) + 1


def split_by_cache(
    scorable: list[tuple[str, str]],
    score_cache: dict[tuple[str, str], float] | None,
) -> tuple[dict[tuple[str, str], float], list[tuple[str, str]]]:
    """Split pairs into (already-cached scores, must-be-resolved)."""
    if score_cache is None:
        return {}, list(scorable)
    cached: dict[tuple[str, str], float] = {}
    remaining: list[tuple[str, str]] = []
    for pair in scorable:
        if pair in score_cache:
            cached[pair] = score_cache[pair]
        else:
            remaining.append(pair)
    return cached, remaining


def role_sign(role: str | None) -> float:
    """Zero-sum sign for an agent's role: +1 coherence, -1 contradiction, 0 else."""
    if role == "coherence":
        return 1.0
    if role == "contradiction":
        return -1.0
    return 0.0
