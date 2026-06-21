"""Tests for the snapshot module — wire-format primitives.

Storage-layer tests (`SnapshotStore`) live in this same file but are added
in a later task; the docstring will be updated when those land.
"""

from __future__ import annotations

import base64
import json

import fakeredis
import numpy as np
import pytest

from multi_agent.snapshot import (
    RouterSchemaMismatch,
    RouterSnapshot,
    SCHEMA_VERSION,
    SnapshotStore,
    encode_ndarray,
    decode_ndarray,
    hmac_sign,
    hmac_verify,
    validate_session_id,
)


def test_encode_decode_ndarray_roundtrip():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
    encoded = encode_ndarray(arr)
    assert encoded["dtype"] == "float32"
    assert encoded["shape"] == [3, 4]
    decoded = decode_ndarray(encoded)
    np.testing.assert_array_equal(decoded, arr)
    assert decoded.dtype == np.float32


def test_decode_ndarray_rejects_non_float32():
    bad = {"dtype": "float64", "shape": [3], "data": "AAAA"}
    with pytest.raises(ValueError, match="float32"):
        decode_ndarray(bad)


def test_encode_ndarray_rejects_non_float32():
    arr = np.zeros(4, dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        encode_ndarray(arr)


def test_decode_ndarray_rejects_truncated_payload():
    arr = np.arange(8, dtype=np.float32)
    encoded = encode_ndarray(arr)
    # Drop the last 8 bytes -> payload now too short for shape (8,).
    truncated_data = base64.b64decode(encoded["data"])[:-8]
    bad = {
        "dtype": "float32",
        "shape": encoded["shape"],
        "data": base64.b64encode(truncated_data).decode("ascii"),
    }
    with pytest.raises(ValueError, match="payload length"):
        decode_ndarray(bad)


def test_hmac_sign_verify_roundtrip():
    key = b"secret-key-bytes"
    payload = b"\x00\x01\x02hello"
    sig = hmac_sign(payload, session_id="sess-1", step=4, schema_version=1, key=key)
    assert hmac_verify(
        payload, sig, session_id="sess-1", step=4, schema_version=1, key=key
    )


def test_hmac_verify_rejects_wrong_session_id():
    key = b"secret-key-bytes"
    payload = b"data"
    sig = hmac_sign(payload, session_id="sess-1", step=4, schema_version=1, key=key)
    assert not hmac_verify(
        payload, sig, session_id="sess-2", step=4, schema_version=1, key=key
    )


def test_hmac_verify_rejects_tampered_payload():
    key = b"secret-key-bytes"
    sig = hmac_sign(b"data", session_id="s", step=1, schema_version=1, key=key)
    assert not hmac_verify(
        b"DATA", sig, session_id="s", step=1, schema_version=1, key=key
    )


def test_hmac_verify_rejects_wrong_step():
    key = b"k"
    sig = hmac_sign(b"d", session_id="s", step=1, schema_version=1, key=key)
    assert not hmac_verify(b"d", sig, session_id="s", step=2, schema_version=1, key=key)


def test_validate_session_id_accepts_normal_ids():
    assert validate_session_id("abc-DEF_123") == "abc-DEF_123"


def test_validate_session_id_accepts_namespaced_form():
    # Production session IDs arrive as `{key_id}:{public_session_id}` from
    # the gateway's _namespace_session_id (api/proxy.py).
    fixture = "dys_key_TESTKEYAAAAAAAAAAAA:semianalysis-corpus-004"
    assert validate_session_id(fixture) == fixture
    assert validate_session_id("dys_key_abc:session-1") == "dys_key_abc:session-1"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a/b",
        "../etc",
        "a b",
        "x" * 129,
        "héllo",
        # Namespaced form must have exactly one colon at the structural
        # boundary — these all violate that invariant.
        ":no-key-prefix",
        "no-public-session:",
        "key::session",
        "a:b:c",
        ":",
    ],
)
def test_validate_session_id_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        validate_session_id(bad)


@pytest.mark.parametrize("bad", [None, 42, [], {"x": 1}, b"bytes-not-str"])
def test_validate_session_id_rejects_non_str(bad):
    with pytest.raises(ValueError):
        validate_session_id(bad)


def _sample_snapshot() -> RouterSnapshot:
    return RouterSnapshot(
        schema_version=SCHEMA_VERSION,
        session_id="sess-1",
        step=3,
        emb_dim=4,
        multi_agent_config={
            "emb_dim": 4,
            "num_agents": 2,
            "k": 2,
            "temperature": 0.3,
            "agent_roles": {
                "agent_0": "coherence",
                "agent_1": "contradiction",
                "cosine": "semantic",
            },
            "device": "cpu",
        },
        graph_hyperparams={
            "attention_step": 0.2,
            "prior_variance": 1.0,
            "obs_variance": 0.05,
            "confidence_floor": 0.25,
        },
        bid_to_text={"b-1": "alpha", "b-2": "beta"},
        sigma={"agent_0": 0.4, "agent_1": 0.4, "cosine": 0.2},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        history=[],
        meta_weights={"agent_0": 1.1, "agent_1": 0.9, "cosine": 1.0},
        score_cache={"b-1|b-2": 0.42},
        graph_z={
            "b-1": np.ones(4, dtype=np.float32),
            "b-2": np.full(4, 2.0, dtype=np.float32),
        },
        graph_raw={
            "b-1": np.ones(4, dtype=np.float32),
            "b-2": np.full(4, 2.0, dtype=np.float32),
        },
        graph_adj={"b-1": ["b-2"], "b-2": ["b-1"]},
        graph_edges={"b-1|b-2": 0.7},
        agent_pop_stats={
            "agent_0": {"wins": 2.0, "rounds": 5.0, "cum_reward": 0.8},
            "agent_1": {"wins": 3.0, "rounds": 5.0, "cum_reward": 1.2},
            "cosine": {"wins": 0.0, "rounds": 5.0, "cum_reward": 0.5},
        },
    )


def test_router_snapshot_roundtrip():
    snap = _sample_snapshot()
    encoded = snap.to_meta_json()
    assert isinstance(encoded, bytes)
    restored = RouterSnapshot.from_meta_json(encoded)

    assert restored.schema_version == snap.schema_version
    assert restored.session_id == snap.session_id
    assert restored.step == snap.step
    assert restored.emb_dim == snap.emb_dim
    assert restored.bid_to_text == snap.bid_to_text
    assert restored.sigma == snap.sigma
    assert restored.roles == snap.roles
    assert restored.meta_weights == snap.meta_weights
    assert restored.score_cache == snap.score_cache
    assert restored.graph_adj == snap.graph_adj
    assert restored.graph_edges == snap.graph_edges
    assert restored.graph_edge_count == 1
    assert restored.graph_edge_timestamps == {"b-1|b-2": 1}
    assert restored.agent_pop_stats == snap.agent_pop_stats
    for bid, arr in snap.graph_z.items():
        np.testing.assert_array_equal(restored.graph_z[bid], arr)
    for bid, arr in snap.graph_raw.items():
        np.testing.assert_array_equal(restored.graph_raw[bid], arr)


def test_router_snapshot_backfills_missing_edge_clock_fields():
    snap = _sample_snapshot()
    obj = json.loads(snap.to_meta_json())
    obj.pop("graph_edge_count")
    obj.pop("graph_edge_timestamps")

    restored = RouterSnapshot.from_meta_json(json.dumps(obj).encode())

    assert restored.graph_edge_count == len(restored.graph_edges)
    assert restored.graph_edge_timestamps == {"b-1|b-2": 1}


def test_router_snapshot_rejects_old_schema():
    snap = _sample_snapshot()
    encoded = snap.to_meta_json()
    # Tamper the schema_version
    obj = json.loads(encoded)
    obj["schema_version"] = SCHEMA_VERSION - 1
    bad = json.dumps(obj).encode()
    with pytest.raises(RouterSchemaMismatch):
        RouterSnapshot.from_meta_json(bad)


def _store():
    return SnapshotStore(
        redis_client=fakeredis.FakeStrictRedis(),
        hmac_key=b"test-secret-key",
    )


def test_snapshot_store_publish_and_load_roundtrip():
    store = _store()
    snap = _sample_snapshot()
    weights = b"fake-torch-save-bytes"
    store.publish("sess-1", snap, weights, ttl_sec=60)

    loaded = store.load("sess-1")
    assert loaded is not None
    restored, w = loaded
    assert restored.session_id == snap.session_id
    assert restored.step == snap.step
    assert w == weights


def test_snapshot_store_load_returns_none_on_missing_keys():
    assert _store().load("nope") is None


def test_snapshot_store_load_returns_none_on_tampered_meta():
    store = _store()
    snap = _sample_snapshot()
    store.publish("sess-1", snap, b"w", ttl_sec=60)
    # Tamper the meta blob by writing different bytes under the same key.
    store._redis.set("s:sess-1:belief_activation:snapshot:meta", b"GARBAGE")
    assert store.load("sess-1") is None  # HMAC mismatch -> miss


def test_snapshot_store_load_returns_none_on_tampered_weights():
    store = _store()
    snap = _sample_snapshot()
    store.publish("sess-1", snap, b"w", ttl_sec=60)
    store._redis.set("s:sess-1:belief_activation:snapshot:weights", b"WRONG")
    assert store.load("sess-1") is None


def test_snapshot_store_delete():
    store = _store()
    store.publish("sess-1", _sample_snapshot(), b"w", ttl_sec=60)
    store.delete("sess-1")
    assert store.load("sess-1") is None


def test_snapshot_store_lock_acquire_release():
    store = _store()
    tok = store.acquire_lock("sess-1", ttl_sec=60)
    assert tok is not None
    # Second acquire while held returns None.
    assert store.acquire_lock("sess-1", ttl_sec=60) is None
    assert store.release_lock("sess-1", tok) is True
    # After release, can re-acquire.
    assert store.acquire_lock("sess-1", ttl_sec=60) is not None


def test_snapshot_store_release_with_wrong_token_does_not_release():
    store = _store()
    tok = store.acquire_lock("sess-1", ttl_sec=60)
    assert store.release_lock("sess-1", "not-the-token") is False
    # Original holder still holds.
    assert store.acquire_lock("sess-1", ttl_sec=60) is None


def test_snapshot_store_load_rejects_wrong_session_id_in_blob():
    """Defend against blob copy-paste between sessions: a valid-HMAC blob
    whose internal session_id differs from the lookup key must be rejected."""
    store_a = _store()
    # Publish under "sess-1" — the blob's internal session_id will be "sess-1"
    # because _sample_snapshot()'s fixture hardcodes that.
    snap = _sample_snapshot()  # session_id="sess-1"
    store_a.publish("sess-1", snap, b"w", ttl_sec=60)

    # Copy all four keys to sess-2 namespace, simulating blob theft/copy.
    redis = store_a._redis
    for src_key, dst_key in [
        (
            "s:sess-1:belief_activation:snapshot:meta",
            "s:sess-2:belief_activation:snapshot:meta",
        ),
        (
            "s:sess-1:belief_activation:snapshot:weights",
            "s:sess-2:belief_activation:snapshot:weights",
        ),
        (
            "s:sess-1:belief_activation:snapshot:hmac_meta",
            "s:sess-2:belief_activation:snapshot:hmac_meta",
        ),
        (
            "s:sess-1:belief_activation:snapshot:hmac_weights",
            "s:sess-2:belief_activation:snapshot:hmac_weights",
        ),
    ]:
        redis.set(dst_key, redis.get(src_key))

    # Lookup against sess-2 must fail (session_id in blob is "sess-1").
    assert store_a.load("sess-2") is None
