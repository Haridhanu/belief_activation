from __future__ import annotations
import numpy as np
import torch
import pytest


def test_config_tgn_defaults_off():
    from multi_agent.config import MultiAgentConfig
    cfg = MultiAgentConfig()
    assert cfg.use_tgn is False


def test_config_tgn_fields_round_trip():
    from multi_agent.config import MultiAgentConfig
    cfg = MultiAgentConfig(
        use_tgn=True,
        tgn_memory_dim=64,
        tgn_time_dim=16,
        tgn_n_attn_heads=2,
    )
    assert cfg.use_tgn is True
    assert cfg.tgn_memory_dim == 64
    assert cfg.tgn_time_dim == 16
    assert cfg.tgn_n_attn_heads == 2


def test_config_from_yaml_tgn_fields(tmp_path):
    import yaml
    from multi_agent.config import MultiAgentConfig
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump({"use_tgn": True, "tgn_memory_dim": 64}))
    cfg = MultiAgentConfig.from_yaml(p)
    assert cfg.use_tgn is True
    assert cfg.tgn_memory_dim == 64
