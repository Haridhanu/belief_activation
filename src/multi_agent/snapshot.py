"""Cryptographic and serialization primitives plus the snapshot envelope.

Currently exports:
- `encode_ndarray` / `decode_ndarray` — base64 + JSON wire format for float32 ndarrays.
- `hmac_sign` / `hmac_verify` — HMAC-SHA256 envelope helpers.
- `validate_session_id` — guard for any session id interpolated into a Redis key.
- `RouterSnapshot` (+ `RouterSchemaMismatch`) — the dataclass + JSON envelope.
- `SnapshotStore` — Redis I/O with HMAC-verified atomic publish and SETNX lock.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
HMAC_KEY_ENV_VAR = "BELIEF_ACTIVATION_SNAPSHOT_HMAC_KEY"

# Low-level: any identifier safe to interpolate into a Redis key. The
# library doesn't know or care whether callers use the namespaced gateway
# form, a bare public id, a UUID, or something else — only that the bytes
# can be embedded in a key segment without enabling injection.
_REDIS_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Structural: the perseverate-namespaced form `{key_id}:{public_session_id}`
# produced by the gateway's _namespace_session_id (api/proxy.py). Encoded
# here so callers in the daydream pipeline can validate against the exact
# wire format perseverate-api emits, without parsing it themselves.
_NAMESPACED_SESSION_ID_RE = re.compile(
    r"^(?P<key_id>[A-Za-z0-9_-]{1,63}):(?P<public_id>[A-Za-z0-9_-]{1,64})$"
)


def _validate_redis_safe_id(value: str, *, label: str) -> str:
    """Reject anything that isn't safe to interpolate into a Redis key.

    `^[A-Za-z0-9_-]{1,64}$` blocks dots, slashes, colons, spaces, and
    non-ASCII so values can't escape their Redis key segment.
    """
    if not isinstance(value, str) or not _REDIS_SAFE_ID_RE.match(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def validate_session_id(session_id: str) -> str:
    """Validate a session_id reaching the activation engine.

    Accepts either:
      - A bare Redis-safe id (single-tenant / test deploys), or
      - The gateway's namespaced form `{key_id}:{public_session_id}` where
        both segments independently pass `_validate_redis_safe_id`.

    The structural regex enforces *exactly one* colon at a fixed boundary —
    multiple, leading, or trailing colons are rejected so the value cannot
    collide with snapshot Redis keys (`s:{session_id}:belief_activation:…`)
    or escape into adjacent namespaces. Snapshot HMAC verification binds
    session_id into every payload signature as defense-in-depth.
    """
    if not isinstance(session_id, str):
        raise ValueError(f"invalid session_id: {session_id!r}")
    if _NAMESPACED_SESSION_ID_RE.match(session_id):
        return session_id
    return _validate_redis_safe_id(session_id, label="session_id")


def encode_ndarray(arr: np.ndarray) -> dict[str, Any]:
    """Encode a float32 ndarray as a JSON-safe dict.

    The wire format pins float32 for predictability and so that HMAC
    signatures bind to a definite byte layout. Non-float32 inputs are
    rejected; callers must cast explicitly.
    """
    if arr.dtype != np.float32:
        raise TypeError(f"encode_ndarray requires float32, got {arr.dtype}")
    return {
        "dtype": "float32",
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_ndarray(d: dict[str, Any]) -> np.ndarray:
    """Decode the wire-format dict back to a writable float32 ndarray.

    Validates dtype and that the decoded byte length matches `shape`
    before reshaping, so a truncated payload fails with a clear error
    rather than the generic message numpy emits from ``reshape``.
    """
    if d.get("dtype") != "float32":
        raise ValueError(
            f"only float32 is supported on the wire (got {d.get('dtype')!r})"
        )
    raw = base64.b64decode(d["data"])
    expected_bytes = int(np.prod(d["shape"])) * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"payload length {len(raw)} does not match shape {d['shape']} "
            f"(expected {expected_bytes} bytes for float32)"
        )
    arr = np.frombuffer(raw, dtype=np.float32).reshape(d["shape"])
    return arr.copy()


def hmac_sign(
    payload: bytes,
    *,
    session_id: str,
    step: int,
    schema_version: int,
    key: bytes,
) -> str:
    """Sign `(session_id, step, schema_version, sha256(payload))` with HS256."""
    digest = hashlib.sha256(payload).digest()
    msg = f"{session_id}|{step}|{schema_version}|".encode() + digest
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def hmac_verify(
    payload: bytes,
    signature: str,
    *,
    session_id: str,
    step: int,
    schema_version: int,
    key: bytes,
) -> bool:
    expected = hmac_sign(
        payload,
        session_id=session_id,
        step=step,
        schema_version=schema_version,
        key=key,
    )
    return hmac.compare_digest(expected, signature)


class RouterSchemaMismatch(Exception):
    """Raised when a snapshot's schema_version doesn't match the running code."""


@dataclass
class RouterSnapshot:
    schema_version: int
    session_id: str
    step: int
    emb_dim: int

    # Configuration needed to rebuild AgentPopulation skeleton on load.
    multi_agent_config: dict[str, Any]

    # Graph Bayesian hyperparameters — restored so resumed runs use identical
    # propagation dynamics as the original session.
    graph_hyperparams: dict[str, float]

    # Consumer-relevant identity & inference state.
    bid_to_text: dict[str, str]
    sigma: dict[str, float]
    roles: dict[str, str]

    # Producer-resume state (consumer can ignore everything below).
    history: list[dict[str, Any]]
    """Serialised StepStats records — one per training step. Each record is
    asdict(stats); restored as raw dicts (callers that need the typed form
    can rebuild StepStats themselves)."""
    meta_weights: dict[str, float]
    score_cache: dict[str, float]  # key = "a|b" sorted
    graph_z: dict[str, np.ndarray]
    graph_raw: dict[str, np.ndarray]
    graph_adj: dict[str, list[str]]
    graph_edges: dict[str, float]  # key = "a|b" sorted
    agent_pop_stats: dict[str, dict[str, float]]
    graph_edge_count: int = 0
    graph_edge_timestamps: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Backfill edge-clock fields for snapshots written before they existed."""
        if self.graph_edges and self.graph_edge_count < len(self.graph_edges):
            self.graph_edge_count = len(self.graph_edges)

        timestamps = {
            str(key): int(value)
            for key, value in self.graph_edge_timestamps.items()
            if key in self.graph_edges
        }
        next_ts = max(timestamps.values(), default=0) + 1
        for key in self.graph_edges:
            if key not in timestamps:
                timestamps[key] = next_ts
                next_ts += 1
        self.graph_edge_timestamps = timestamps

        max_ts = max(self.graph_edge_timestamps.values(), default=0)
        if self.graph_edge_count < max_ts:
            self.graph_edge_count = max_ts

    def to_meta_json(self) -> bytes:
        d = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "step": self.step,
            "emb_dim": self.emb_dim,
            "multi_agent_config": self.multi_agent_config,
            "graph_hyperparams": self.graph_hyperparams,
            "bid_to_text": self.bid_to_text,
            "sigma": self.sigma,
            "roles": self.roles,
            "history": self.history,
            "meta_weights": self.meta_weights,
            "score_cache": self.score_cache,
            "graph_z": {bid: encode_ndarray(arr) for bid, arr in self.graph_z.items()},
            "graph_raw": {
                bid: encode_ndarray(arr) for bid, arr in self.graph_raw.items()
            },
            "graph_adj": self.graph_adj,
            "graph_edges": self.graph_edges,
            "agent_pop_stats": self.agent_pop_stats,
            "graph_edge_count": self.graph_edge_count,
            "graph_edge_timestamps": self.graph_edge_timestamps,
        }
        return json.dumps(d, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_meta_json(cls, data: bytes) -> "RouterSnapshot":
        obj = json.loads(data)
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise RouterSchemaMismatch(
                f"snapshot schema_version={obj.get('schema_version')!r}, "
                f"runtime expects {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=obj["schema_version"],
            session_id=obj["session_id"],
            step=obj["step"],
            emb_dim=obj["emb_dim"],
            multi_agent_config=obj["multi_agent_config"],
            graph_hyperparams=obj["graph_hyperparams"],
            bid_to_text=obj["bid_to_text"],
            sigma=obj["sigma"],
            roles=obj["roles"],
            history=obj["history"],
            meta_weights=obj["meta_weights"],
            score_cache=obj["score_cache"],
            graph_z={bid: decode_ndarray(d) for bid, d in obj["graph_z"].items()},
            graph_raw={bid: decode_ndarray(d) for bid, d in obj["graph_raw"].items()},
            graph_adj=obj["graph_adj"],
            graph_edges=obj["graph_edges"],
            agent_pop_stats=obj["agent_pop_stats"],
            graph_edge_count=int(obj.get("graph_edge_count", 0)),
            graph_edge_timestamps={
                str(key): int(value)
                for key, value in obj.get("graph_edge_timestamps", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Redis key helpers
# ---------------------------------------------------------------------------


def _key_meta(session_id: str) -> str:
    return f"s:{session_id}:belief_activation:snapshot:meta"


def _key_weights(session_id: str) -> str:
    return f"s:{session_id}:belief_activation:snapshot:weights"


def _key_hmac_meta(session_id: str) -> str:
    return f"s:{session_id}:belief_activation:snapshot:hmac_meta"


def _key_hmac_weights(session_id: str) -> str:
    return f"s:{session_id}:belief_activation:snapshot:hmac_weights"


def _key_lock(session_id: str) -> str:
    return f"s:{session_id}:belief_activation:lock"


# ---------------------------------------------------------------------------
# SnapshotStore
# ---------------------------------------------------------------------------


class SnapshotStore:
    """Redis storage for `RouterSnapshot` blobs.

    Two payloads (meta JSON, torch-save weights bytes) are written atomically
    via MULTI/EXEC alongside their HMAC signatures. Reads verify HMACs and
    refuse to deserialise tampered blobs.
    """

    def __init__(self, redis_client, hmac_key: bytes) -> None:
        self._redis = redis_client
        self._hmac_key = hmac_key

    def publish(
        self,
        session_id: str,
        snapshot: RouterSnapshot,
        weights: bytes,
        ttl_sec: int = 3600,
        lock_token: str | None = None,
    ) -> bool:
        """Write snapshot atomically. Returns False (no-op) if lock_token is
        provided but no longer matches the live lock key — guards against
        stale workers publishing after their TTL expired."""
        validate_session_id(session_id)
        if lock_token is not None:
            current = self._redis.get(_key_lock(session_id))
            if current is None or current.decode() != lock_token:
                logger.warning(
                    "publish skipped for session=%s — lock token mismatch "
                    "(TTL expired or another worker holds the lock)",
                    session_id,
                )
                return False
        meta = snapshot.to_meta_json()
        sig_meta = hmac_sign(
            meta,
            session_id=session_id,
            step=snapshot.step,
            schema_version=snapshot.schema_version,
            key=self._hmac_key,
        )
        sig_weights = hmac_sign(
            weights,
            session_id=session_id,
            step=snapshot.step,
            schema_version=snapshot.schema_version,
            key=self._hmac_key,
        )
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(_key_meta(session_id), meta, ex=ttl_sec)
        pipe.set(_key_weights(session_id), weights, ex=ttl_sec)
        pipe.set(_key_hmac_meta(session_id), sig_meta, ex=ttl_sec)
        pipe.set(_key_hmac_weights(session_id), sig_weights, ex=ttl_sec)
        pipe.execute()
        return True

    def load(self, session_id: str) -> Optional[tuple[RouterSnapshot, bytes]]:
        validate_session_id(session_id)
        pipe = self._redis.pipeline(transaction=False)
        pipe.get(_key_meta(session_id))
        pipe.get(_key_weights(session_id))
        pipe.get(_key_hmac_meta(session_id))
        pipe.get(_key_hmac_weights(session_id))
        meta, weights, sig_meta, sig_weights = pipe.execute()
        if meta is None or weights is None or sig_meta is None or sig_weights is None:
            return None

        # We need the step and schema_version to verify, but they live inside the
        # meta JSON. Parse provisionally (no schema check yet) so we can verify
        # the HMAC bound to those values; if HMAC fails we return None and the
        # caller falls back. If schema_version is wrong, from_meta_json raises
        # later — that's a separate failure mode handled by the caller.
        try:
            obj = json.loads(meta)
            session_in_blob = obj["session_id"]
            step = int(obj["step"])
            schema_version = int(obj["schema_version"])
        except (ValueError, KeyError, TypeError):
            logger.error("snapshot meta blob unparseable for session=%s", session_id)
            return None
        if session_in_blob != session_id:
            logger.error(
                "snapshot session_id mismatch in blob: blob=%r expected=%r",
                session_in_blob,
                session_id,
            )
            return None

        sig_meta_str = sig_meta.decode() if isinstance(sig_meta, bytes) else sig_meta
        sig_weights_str = (
            sig_weights.decode() if isinstance(sig_weights, bytes) else sig_weights
        )
        if not hmac_verify(
            meta,
            sig_meta_str,
            session_id=session_id,
            step=step,
            schema_version=schema_version,
            key=self._hmac_key,
        ) or not hmac_verify(
            weights,
            sig_weights_str,
            session_id=session_id,
            step=step,
            schema_version=schema_version,
            key=self._hmac_key,
        ):
            logger.error(
                "snapshot HMAC verification failed for session=%s step=%s",
                session_id,
                step,
            )
            return None

        try:
            snap = RouterSnapshot.from_meta_json(meta)
        except RouterSchemaMismatch:
            raise
        except Exception:
            logger.error(
                "snapshot meta deserialization failed for session=%s",
                session_id,
                exc_info=True,
            )
            return None
        return snap, weights

    def delete(self, session_id: str) -> None:
        validate_session_id(session_id)
        self._redis.delete(
            _key_meta(session_id),
            _key_weights(session_id),
            _key_hmac_meta(session_id),
            _key_hmac_weights(session_id),
        )

    def acquire_lock(self, session_id: str, ttl_sec: int = 60) -> Optional[str]:
        validate_session_id(session_id)
        token = secrets.token_hex(16)
        ok = self._redis.set(_key_lock(session_id), token, nx=True, ex=ttl_sec)
        return token if ok else None

    def release_lock(self, session_id: str, token: str) -> bool:
        validate_session_id(session_id)
        # Compare-and-delete via Lua so we don't drop someone else's lock.
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] "
            "then return redis.call('DEL', KEYS[1]) else return 0 end"
        )
        result = self._redis.eval(script, 1, _key_lock(session_id), token)
        return bool(result)
