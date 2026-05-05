"""Tests for multi_agent.config."""

from __future__ import annotations

import pytest

from multi_agent.config import MultiAgentConfig


def test_defaults():
    cfg = MultiAgentConfig()
    assert cfg.emb_dim == 768
    assert cfg.device == "cpu"
    assert cfg.num_agents == 3
    assert cfg.k == 8
    assert cfg.temperature == 0.3
    assert cfg.tournament_size == 3
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.005
    assert cfg.llm_concurrency == 20


def test_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "emb_dim: 16\nnum_agents: 5\nk: 4\nbatch_size: 8\n",
    )
    cfg = MultiAgentConfig.from_yaml(p)
    assert cfg.emb_dim == 16
    assert cfg.num_agents == 5
    assert cfg.k == 4
    assert cfg.batch_size == 8
    # Unspecified keys take defaults.
    assert cfg.temperature == 0.3
    assert cfg.learning_rate == 0.005


def test_from_yaml_unknown_key_raises(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("emb_dim: 16\nbogus_field: 42\n")
    with pytest.raises(TypeError):
        MultiAgentConfig.from_yaml(p)


def test_config_tgn_defaults_to_disabled():
    cfg = MultiAgentConfig()
    assert cfg.use_tgn is False
    assert cfg.tgn_memory_dim == 128
    assert cfg.tgn_time_dim == 32
    assert cfg.tgn_blend == 0.3
    assert cfg.time_decay == 0.1


def test_config_tgn_fields_survive_yaml_round_trip(tmp_path):
    import yaml

    cfg = MultiAgentConfig(
        use_tgn=True, tgn_memory_dim=64, tgn_time_dim=16, tgn_blend=0.2, time_decay=0.05
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.dump(
            {
                "use_tgn": True,
                "tgn_memory_dim": 64,
                "tgn_time_dim": 16,
                "tgn_blend": 0.2,
                "time_decay": 0.05,
            }
        )
    )
    loaded = MultiAgentConfig.from_yaml(path)
    assert loaded.use_tgn is True
    assert loaded.tgn_memory_dim == 64
    assert loaded.tgn_time_dim == 16
    assert loaded.tgn_blend == 0.2
    assert loaded.time_decay == 0.05
