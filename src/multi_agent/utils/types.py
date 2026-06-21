from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class AgentProposal:
    """One agent's output for a batch of queries."""

    scores: torch.Tensor  # (B, N)
    indices: torch.Tensor  # (B, k_eff) — rectangular, may be 0-wide
    proposals: list[list[str]]  # pool identifiers, one list per query
    row_indices: list[torch.Tensor] = field(default_factory=list)  # ragged; len B


@dataclass(frozen=True)
class ProposalBatch:
    """All agents' proposals for a batch, plus the cross-agent pair counter."""

    by_agent: dict[str, AgentProposal]
    pair_counts: dict[tuple[str, str], int]


class JudgeResult(NamedTuple):
    """Outcome of one judge pass: every pair's score (cached/imputed/judged),
    the subset actually sent to the judge, and counts for telemetry."""

    score_by_pair: dict[tuple[str, str], float]
    judged_pairs: list[tuple[tuple[str, str], float]]
    stats: dict[str, int]
