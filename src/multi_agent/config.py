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
    # TGN's projected memory (instead of the signed-attention `_z`),
    # and `Graph.impute` / `Graph.field` delegate to `tgn.predict_link`.
    # All defaults below are no-ops when use_tgn=False.
    use_tgn: bool = False
    tgn_memory_dim: int = 128
    tgn_time_dim: int = 32
    tgn_n_attn_heads: int = 4
    tgn_lr: float = 1e-3
    tgn_predict_threshold: float = 0.2

    # Cold-start handling — "pure" (option C) always uses
    # mem_to_emb(memory) as the candidate representation, even when memory
    # is all-zeros at session start. "raw_fallback" (option A) uses the
    # raw embedding for nodes whose memory has not yet been touched by
    # any event, then switches to projected memory once any event fires.
    tgn_cold_start: str = "pure"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultiAgentConfig":

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
