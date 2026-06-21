"""Single configuration dataclass for the multi-agent retrieval system.

YAML-driven via ``MultiAgentConfig.from_yaml(path)``. All fields are flat
so a config file maps 1:1 to constructor kwargs — no nested sections to
remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class MultiAgentConfig:

    emb_dim: int = 768
    device: str = "cpu"

    num_agents: int = 3
    k: int = 8
    temperature: float = 0.3

    tournament_size: int = 3
    batch_size: int = 32
    learning_rate: float = 0.005

    llm_concurrency: int = 20

    judge_budget_per_batch: int = 0

    n_epochs: int = 1
    log_every: int = 0
    seed: int | None = None

    meta_lr: float = 0.1
    meta_eps: float = 0.05

    agent_roles: dict[str, str] | None = None

    # TGN integration: when use_tgn=True the TGN replaces the Bayesian
    # graph machinery. Agents continue to score (query, candidate) via
    # their attention + MLP heads, but `candidate_reps` come from the
    # TGN's projected memory (instead of the signed-attention `_z`).
    # Warm `Graph.impute` / `Graph.field` delegate to `tgn.predict_link`;
    # cold `raw_fallback` pairs use raw geometry only for field() and still
    # defer impute() to the judge.
    # All defaults below are no-ops when use_tgn=False.
    use_tgn: bool = False
    tgn_memory_dim: int = 128
    tgn_time_dim: int = 32
    tgn_n_attn_heads: int = 4
    tgn_lr: float = 1e-3
    tgn_predict_threshold: float = 0.2
    # Auxiliary loss weight for aligning mem_to_emb(memory) with raw
    # embedding coordinates. This keeps the TGN-backed candidate
    # representation meaningful to agents whose query vectors still live in
    # the original embedding space. The default is intentionally nonzero so
    # TGN-backed BA trains mem_to_emb by default; set to 0.0 to disable.
    # Applies only when use_tgn=True.
    tgn_rep_align_weight: float = 0.05

    # Cold-start handling.
    #
    # "raw_fallback" (default) — use the raw embedding for nodes whose
    #   memory has not yet been touched by any event, then switch to
    #   ``mem_to_emb(memory)`` once any event fires for that node. Default
    #   because it keeps cold nodes distinguishable from each other; the
    #   alternative collapses every cold node to the same vector.
    #
    # "pure" — always use ``mem_to_emb(memory)``, including at cold start
    #   when memory is all-zeros. Because ``mem_to_emb(0) == bias``, every
    #   untouched node returns the SAME representation; downstream ranking
    #   over an all-cold graph is degenerate. Only opt in when you intend
    #   to warm memory before any ranking call.
    tgn_cold_start: str = "raw_fallback"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultiAgentConfig":

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
