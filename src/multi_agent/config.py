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

    # TGN extension (all default to safe no-op values)
    use_tgn: bool = False
    tgn_memory_dim: int = 128
    tgn_time_dim: int = 32
    tgn_n_attn_heads: int = 4
    tgn_blend: float = 0.3
    time_decay: float = 0.1
    baseline_norm: float = 1.0
    impute_blend_lr: float = 1e-3

    # Engine selector — picks the inner training loop:
    #   "psro"     : current multi-agent + PSRO trainer (default, unchanged)
    #   "tgn_only" : TGN-only active-learning trainer (no agents, no Bayes)
    engine: str = "psro"
    tgn_only_lr: float = 1e-3
    # Lower threshold: link_head's Tanh output stays in roughly [-0.5, +0.5]
    # for most of training because pre-Tanh activations are small. 0.5 was
    # too conservative — the gate never opened. 0.2 lets confident-enough
    # predictions commit, with calibration (below) refining further.
    tgn_only_commit_threshold: float = 0.2
    tgn_only_candidate_k: int = 8
    # Calibration-based commit gate: track |pred| vs sign-correctness over
    # judged pairs; commit a predicted edge only when its magnitude lands
    # in a regime where empirical sign accuracy meets `target_accuracy`.
    # Falls back to `tgn_only_commit_threshold` until we have at least
    # `warmup` calibration samples.
    tgn_only_calibration_target: float = 0.7
    tgn_only_calibration_warmup: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultiAgentConfig":

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
