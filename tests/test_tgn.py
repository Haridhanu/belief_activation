from __future__ import annotations
import torch
import pytest


def test_time_encoder_output_shape():
    from multi_agent.tgn import TimeEncoder
    enc = TimeEncoder(time_dim=32)
    out = enc(torch.tensor(5.0))
    assert out.shape == (32,)


def test_time_encoder_has_no_learnable_parameters():
    from multi_agent.tgn import TimeEncoder
    enc = TimeEncoder(time_dim=32)
    assert sum(p.numel() for p in enc.parameters()) == 0


def test_time_encoder_different_times_differ():
    from multi_agent.tgn import TimeEncoder
    enc = TimeEncoder(time_dim=32)
    assert not torch.allclose(enc(torch.tensor(0.0)), enc(torch.tensor(5.0)))


def test_time_encoder_zero_time_pattern():
    # sin(0)=0, cos(0)=1, alternating
    from multi_agent.tgn import TimeEncoder
    enc = TimeEncoder(time_dim=4)
    out = enc(torch.tensor(0.0))
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.allclose(out, expected, atol=1e-6)


def test_message_encoder_output_shape():
    from multi_agent.tgn import TemporalMessageEncoder
    enc = TemporalMessageEncoder(memory_dim=32, time_dim=16)
    src = torch.zeros(32)
    dst = torch.zeros(32)
    t_enc = torch.ones(16)
    out = enc(src, dst, sign=1.0, time_enc=t_enc, weight=0.8)
    assert out.shape == (32,)


def test_message_encoder_sign_changes_output():
    from multi_agent.tgn import TemporalMessageEncoder
    enc = TemporalMessageEncoder(memory_dim=32, time_dim=16)
    src, dst, t_enc = torch.randn(32), torch.randn(32), torch.randn(16)
    pos = enc(src, dst, sign=1.0, time_enc=t_enc, weight=0.8)
    neg = enc(src, dst, sign=-1.0, time_enc=t_enc, weight=0.8)
    assert not torch.allclose(pos, neg)


def test_message_encoder_weight_changes_output():
    from multi_agent.tgn import TemporalMessageEncoder
    enc = TemporalMessageEncoder(memory_dim=32, time_dim=16)
    src, dst, t_enc = torch.randn(32), torch.randn(32), torch.randn(16)
    hi = enc(src, dst, sign=1.0, time_enc=t_enc, weight=1.0)
    lo = enc(src, dst, sign=1.0, time_enc=t_enc, weight=0.1)
    assert not torch.allclose(hi, lo)
