from __future__ import annotations
import io
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


def test_node_memory_unseen_returns_zeros():
    from multi_agent.tgn import NodeMemory
    mem = NodeMemory(memory_dim=16)
    out = mem.get("unknown")
    assert out.shape == (16,)
    assert torch.all(out == 0)


def test_node_memory_set_get_roundtrip():
    from multi_agent.tgn import NodeMemory
    mem = NodeMemory(memory_dim=16)
    val = torch.randn(16)
    mem.set("a", val)
    assert torch.allclose(mem.get("a"), val)


def test_node_memory_set_detaches_from_graph():
    from multi_agent.tgn import NodeMemory
    mem = NodeMemory(memory_dim=16)
    val = torch.randn(16, requires_grad=True)
    mem.set("a", val * 2.0)
    assert not mem.get("a").requires_grad


def test_node_memory_reset_clears_all():
    from multi_agent.tgn import NodeMemory
    mem = NodeMemory(memory_dim=16)
    mem.set("a", torch.ones(16))
    mem.reset()
    assert torch.all(mem.get("a") == 0)


def test_node_memory_get_batch_shape_and_zero_fallback():
    from multi_agent.tgn import NodeMemory
    mem = NodeMemory(memory_dim=16)
    mem.set("a", torch.ones(16))
    mem.set("b", torch.ones(16) * 2)
    batch = mem.get_batch(["a", "b", "unseen"])
    assert batch.shape == (3, 16)
    assert torch.allclose(batch[0], torch.ones(16))
    assert torch.all(batch[2] == 0)


def test_node_memory_state_dict_roundtrip():
    from multi_agent.tgn import NodeMemory
    mem = NodeMemory(memory_dim=16)
    mem.set("a", torch.randn(16))
    mem.set("b", torch.randn(16))
    sd = mem.state_dict()
    mem2 = NodeMemory(memory_dim=16)
    mem2.load_state_dict(sd)
    assert torch.allclose(mem.get("a"), mem2.get("a"))
    assert torch.allclose(mem.get("b"), mem2.get("b"))


def test_memory_updater_output_shape():
    from multi_agent.tgn import MemoryUpdater
    upd = MemoryUpdater(memory_dim=32)
    out = upd(torch.randn(32), torch.zeros(32))
    assert out.shape == (32,)


def test_memory_updater_changes_zero_memory():
    from multi_agent.tgn import MemoryUpdater
    upd = MemoryUpdater(memory_dim=32)
    out = upd(torch.randn(32), torch.zeros(32))
    assert not torch.allclose(out, torch.zeros(32))


def test_memory_updater_is_deterministic():
    from multi_agent.tgn import MemoryUpdater
    upd = MemoryUpdater(memory_dim=32)
    msg, mem = torch.randn(32), torch.randn(32)
    assert torch.allclose(upd(msg, mem), upd(msg, mem))


def test_aggregator_output_shape():
    from multi_agent.tgn import TemporalNeighborhoodAggregator
    agg = TemporalNeighborhoodAggregator(memory_dim=32, n_heads=4)
    out = agg(torch.randn(32), torch.randn(5, 32))
    assert out.shape == (32,)


def test_aggregator_single_neighbor():
    from multi_agent.tgn import TemporalNeighborhoodAggregator
    agg = TemporalNeighborhoodAggregator(memory_dim=32, n_heads=4)
    out = agg(torch.randn(32), torch.randn(1, 32))
    assert out.shape == (32,)


def test_aggregator_is_deterministic():
    from multi_agent.tgn import TemporalNeighborhoodAggregator
    torch.manual_seed(0)
    agg = TemporalNeighborhoodAggregator(memory_dim=32, n_heads=4)
    q, nbrs = torch.randn(32), torch.randn(3, 32)
    assert torch.allclose(agg(q, nbrs), agg(q, nbrs))


def test_aggregator_different_neighbors_different_output():
    from multi_agent.tgn import TemporalNeighborhoodAggregator
    torch.manual_seed(1)
    agg = TemporalNeighborhoodAggregator(memory_dim=32, n_heads=4)
    q = torch.randn(32)
    out1 = agg(q, torch.randn(3, 32))
    out2 = agg(q, torch.randn(3, 32))
    assert not torch.allclose(out1, out2)


def test_tgn_module_memory_zero_before_any_update():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    mems = tgn.get_memory(["a", "b", "c"])
    assert mems.shape == (3, 32)
    assert torch.all(mems == 0)


def test_tgn_module_update_makes_memory_nonzero():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.9)
    mems = tgn.get_memory(["a", "b"])
    assert not torch.all(mems == 0)


def test_tgn_module_second_update_changes_memory():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.9)
    mem_a_1 = tgn.get_memory(["a"])[0].clone()
    tgn.update("a", "c", sign=-1.0, timestamp=2.0, edge_weight=0.7)
    mem_a_2 = tgn.get_memory(["a"])[0]
    assert not torch.allclose(mem_a_1, mem_a_2)


def test_tgn_module_predict_link_in_signed_interval():
    """predict_link is now a trained signed predictor: output in [-1, 1]."""
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    score = tgn.predict_link("x", "y")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_tgn_link_loss_drives_predictions_toward_targets():
    """Repeated training on consistent (pair, y) pairs must move predictions
    toward y. This is the core contract of the new trainable link head."""
    import torch

    from multi_agent.tgn import TGNModule
    torch.manual_seed(0)
    tgn = TGNModule(emb_dim=32, memory_dim=16, time_dim=8, n_heads=2)
    # Seed memories so link_head has non-zero input
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.9)
    tgn.update("b", "c", sign=-1.0, timestamp=2.0, edge_weight=0.7)

    pairs = [("a", "b", 0.9), ("b", "c", -0.8)]
    optimizer = torch.optim.Adam(tgn.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(50):
        optimizer.zero_grad()
        loss = tgn.link_loss(pairs)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], (
        f"link_loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_tgn_link_loss_empty_pairs_returns_zero():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=32, memory_dim=16, time_dim=8, n_heads=2)
    loss = tgn.link_loss([])
    assert float(loss.item()) == 0.0


def test_tgn_module_predict_link_changes_after_update():
    from multi_agent.tgn import TGNModule
    torch.manual_seed(0)
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    before = tgn.predict_link("a", "b")
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.9)
    after = tgn.predict_link("a", "b")
    assert before != after


def test_tgn_module_reset_clears_memory():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.9)
    tgn.reset()
    assert torch.all(tgn.get_memory(["a", "b"]) == 0)


def test_tgn_module_project_to_emb_shape():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    mems = torch.randn(5, 32)
    projected = tgn.project_to_emb(mems)
    assert projected.shape == (5, 64)


def test_tgn_module_project_to_emb_is_detached():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    mems = torch.randn(3, 32, requires_grad=False)
    projected = tgn.project_to_emb(mems)
    assert not projected.requires_grad


def test_tgn_state_dict_roundtrip_restores_memories_and_predictions():
    from multi_agent.tgn import TGNModule
    torch.manual_seed(42)
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn.update("a", "b", sign=1.0,  timestamp=1.0, edge_weight=0.9)
    tgn.update("b", "c", sign=-1.0, timestamp=2.0, edge_weight=0.5)

    sd = tgn.state_dict()

    tgn2 = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn2.load_state_dict(sd)

    assert torch.allclose(
        tgn.get_memory(["a", "b", "c"]),
        tgn2.get_memory(["a", "b", "c"]),
    )
    assert abs(tgn.predict_link("a", "c") - tgn2.predict_link("a", "c")) < 1e-6


def test_tgn_state_dict_only_stores_seen_nodes():
    from multi_agent.tgn import TGNModule
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.9)
    sd = tgn.state_dict()
    assert set(sd["_node_memory"].keys()) == {"a", "b"}


def test_tgn_state_dict_torch_save_load_roundtrip():
    from multi_agent.tgn import TGNModule
    torch.manual_seed(7)
    tgn = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn.update("x", "y", sign=-1.0, timestamp=3.0, edge_weight=0.6)

    buf = io.BytesIO()
    torch.save(tgn.state_dict(), buf)
    buf.seek(0)

    tgn2 = TGNModule(emb_dim=64, memory_dim=32, time_dim=16, n_heads=2)
    tgn2.load_state_dict(torch.load(buf, weights_only=False))

    assert torch.allclose(tgn.get_memory(["x", "y"]), tgn2.get_memory(["x", "y"]))
