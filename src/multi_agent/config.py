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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultiAgentConfig":

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
