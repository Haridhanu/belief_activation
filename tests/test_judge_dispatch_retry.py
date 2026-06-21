"""Tests for the judge LLM dispatch retry layer (PLA-247).

Vertex AI returns 429 RESOURCE_EXHAUSTED under quota pressure. For small
corpora (e.g., 2-paper merges) where the merge orchestrator only has
one candidate pair, a single transient 429 propagates as 1/1 bridge
failures → 502. The retry wrapper backs off and re-issues the LLM call
so transient quota hiccups don't kill the whole merge.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from multi_agent import judge as judge_mod
from multi_agent.judge import _dispatch_with_retry, _is_rate_limit_error

# ── _is_rate_limit_error ──────────────────────────────────────────


class TestIsRateLimitError:
    def test_detects_vertex_429_resource_exhausted(self):
        # Matches the exact string shape from the PLA-247 production log:
        # ClientError("429 RESOURCE_EXHAUSTED. ...")
        exc = Exception(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
            "'message': 'Resource exhausted. Please try again later.', "
            "'status': 'RESOURCE_EXHAUSTED'}}"
        )
        assert _is_rate_limit_error(exc) is True

    def test_detects_message_with_only_429(self):
        assert _is_rate_limit_error(Exception("HTTP 429 too many requests")) is True

    def test_detects_message_with_only_resource_exhausted(self):
        assert _is_rate_limit_error(Exception("status=RESOURCE_EXHAUSTED")) is True

    def test_detects_openai_rate_limit_error_by_type_name(self):
        # We can't import openai in unit tests, so construct a class whose
        # __name__ matches what openai.RateLimitError exposes at runtime.
        class RateLimitError(Exception):
            pass

        assert _is_rate_limit_error(RateLimitError("rate limited")) is True

    def test_rejects_unrelated_errors(self):
        assert _is_rate_limit_error(ValueError("bad input")) is False
        assert _is_rate_limit_error(RuntimeError("500 Internal Server Error")) is False
        assert _is_rate_limit_error(TimeoutError("timed out")) is False

    def test_429_match_is_word_boundary_anchored(self):
        # Substrings containing "429" but not as a standalone token shouldn't
        # trigger a retry — e.g. an unrelated request ID, token count, or
        # timestamp. This is the tightening Claude bot flagged.
        assert _is_rate_limit_error(Exception("processed 4290 tokens")) is False
        assert _is_rate_limit_error(Exception("req-id=abc429xyz")) is False
        assert _is_rate_limit_error(Exception("HTTP 1429 nonstandard")) is False
        # And the canonical Vertex/HTTP shapes should still match.
        assert _is_rate_limit_error(Exception("HTTP 429 too many")) is True
        assert _is_rate_limit_error(Exception("status code: 429")) is True


# ── _dispatch_with_retry ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_on_first_success():
    """Happy path — no retry needed when the call succeeds immediately."""
    call_count = {"n": 0}

    async def _call() -> str:
        call_count["n"] += 1
        return '{"scores": []}'

    out = await _dispatch_with_retry(_call)
    assert out == '{"scores": []}'
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(monkeypatch):
    """Single transient 429 → backoff → retry → success.

    This is the core PLA-247 scenario: Vertex returns 429 once during a
    quota dip; with the retry wrapper, the call recovers transparently.
    """
    # Skip the actual sleep so the test is fast.
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    call_count = {"n": 0}

    async def _call() -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return '{"scores": []}'

    out = await _dispatch_with_retry(_call)
    assert out == '{"scores": []}'
    assert call_count["n"] == 2
    # Exactly one backoff was scheduled (between attempts 1 and 2).
    assert sleep_mock.await_count == 1


@pytest.mark.asyncio
async def test_propagates_after_exhausting_retries(monkeypatch):
    """All attempts hit 429 → the final exception still propagates so the
    bridge build is counted as failed (and the small-N tolerance in the
    orchestrator decides whether the whole merge fails)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    # Pin retry count to 2 so the test doesn't depend on env defaults.
    monkeypatch.setattr(judge_mod, "_JUDGE_DISPATCH_MAX_RETRIES", 2)

    call_count = {"n": 0}

    async def _call() -> str:
        call_count["n"] += 1
        raise Exception("429 RESOURCE_EXHAUSTED")

    with pytest.raises(Exception, match=r"429 RESOURCE_EXHAUSTED"):
        await _dispatch_with_retry(_call)
    # Attempts = max_retries + 1 (the final attempt isn't a retry).
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_rate_limit_errors(monkeypatch):
    """Errors that aren't 429 / RESOURCE_EXHAUSTED / RateLimitError must
    propagate immediately. We don't want the retry layer masking real
    failures (auth errors, malformed requests, network issues)."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    call_count = {"n": 0}

    async def _call() -> str:
        call_count["n"] += 1
        raise ValueError("bad prompt")

    with pytest.raises(ValueError, match=r"bad prompt"):
        await _dispatch_with_retry(_call)
    assert call_count["n"] == 1
    assert sleep_mock.await_count == 0


@pytest.mark.asyncio
async def test_backoff_grows_exponentially_with_equal_jitter(monkeypatch):
    """Sleep durations should follow the equal-jitter pattern around
    base, 2*base, 4*base — half of nominal as floor, full nominal as
    ceiling. This is what gives Vertex enough time to clear its rate
    bucket between retries, while de-correlating concurrent retries."""
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    monkeypatch.setattr(judge_mod, "_JUDGE_DISPATCH_MAX_RETRIES", 3)
    monkeypatch.setattr(judge_mod, "_JUDGE_DISPATCH_BACKOFF_BASE_S", 1.0)

    async def _call() -> str:
        raise Exception("429 RESOURCE_EXHAUSTED")

    with pytest.raises(Exception):
        await _dispatch_with_retry(_call)

    waits = [c.args[0] for c in sleep_mock.await_args_list]
    assert len(waits) == 3
    # Each wait sits in [nominal/2, nominal]: 1s → [0.5, 1.0],
    # 2s → [1.0, 2.0], 4s → [2.0, 4.0]. Magnitude still grows monotonically.
    assert 0.5 <= waits[0] <= 1.0
    assert 1.0 <= waits[1] <= 2.0
    assert 2.0 <= waits[2] <= 4.0


@pytest.mark.asyncio
async def test_jitter_decorrelates_parallel_retries(monkeypatch):
    """Two concurrent dispatchers hitting the same rate limit must not
    sleep for the same duration — that's the whole point of jitter.
    Without it, all BS_MERGE_BUILD_CONCURRENCY in-flight builds would
    retry in lockstep and re-collide on the upstream quota bucket."""
    sleep_calls: list[float] = []

    async def _capture_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr("asyncio.sleep", _capture_sleep)
    monkeypatch.setattr(judge_mod, "_JUDGE_DISPATCH_MAX_RETRIES", 1)
    monkeypatch.setattr(judge_mod, "_JUDGE_DISPATCH_BACKOFF_BASE_S", 2.0)

    async def _failing_call() -> str:
        raise Exception("429 RESOURCE_EXHAUSTED")

    # 50 concurrent dispatchers — enough that lockstep collision would be
    # mathematically near-impossible to confuse with jittered output.
    tasks = [_dispatch_with_retry(_failing_call) for _ in range(50)]
    for t in tasks:
        try:
            await t
        except Exception:
            pass

    # 50 dispatchers × 1 retry = 50 sleeps. Some must differ from each other.
    assert len(sleep_calls) == 50
    assert (
        len(set(sleep_calls)) > 1
    ), "all 50 sleeps were identical — jitter not de-correlating retries"


# ── Semaphore behavior (Cursor bot finding #1) ────────────────────


@pytest.mark.asyncio
async def test_semaphore_released_during_backoff_sleep(monkeypatch):
    """The judge concurrency semaphore must be RELEASED during the
    rate-limit backoff sleep, not held across it. Holding the slot
    across multi-second sleeps would starve parallel batches and push
    every in-flight bridge past `bridge_build_timeout_ms`.

    Approach: snapshot the semaphore's `_value` (free-slot count) inside
    the patched sleep. If the slot has been released during sleep, the
    value will be back to the initial capacity. If it's still held
    (pre-fix behavior), the value will be 0.
    """
    import asyncio

    sem = asyncio.Semaphore(1)
    sem_values_during_sleep: list[int] = []

    # Bypass actual delay so the test is fast. We don't recurse into
    # asyncio.sleep because we just record-and-return without ever
    # invoking the real function from within the patch.
    async def _instant_sleep(duration):
        # Inside _dispatch_with_retry, this fires AFTER the failing
        # coro_factory call has returned and the `async with semaphore`
        # block has exited. Record the current free-slot count.
        sem_values_during_sleep.append(sem._value)

    monkeypatch.setattr("asyncio.sleep", _instant_sleep)
    monkeypatch.setattr(judge_mod, "_JUDGE_DISPATCH_MAX_RETRIES", 2)

    async def _failing_call() -> str:
        # While we're inside _failing_call, the semaphore is acquired
        # (value=0). After the call raises and `async with` exits, the
        # value should be back to 1 before _instant_sleep is invoked.
        assert sem._value == 0, "semaphore should be held during the call"
        raise Exception("429 RESOURCE_EXHAUSTED")

    with pytest.raises(Exception, match=r"429"):
        await _dispatch_with_retry(_failing_call, semaphore=sem)

    # Two retries → two backoff sleeps → two snapshots. Both must show
    # the slot returned (value=1), proving the semaphore was released
    # before the sleep, not held across it.
    assert sem_values_during_sleep == [1, 1], (
        f"semaphore was not released during backoff sleep — slot values "
        f"during sleeps were {sem_values_during_sleep}, expected [1, 1]"
    )


@pytest.mark.asyncio
async def test_dispatch_without_semaphore_param_still_works():
    """Passing `semaphore=None` (the default) preserves the old behavior
    for any caller that doesn't have a concurrency limit to enforce."""
    call_count = {"n": 0}

    async def _call() -> str:
        call_count["n"] += 1
        return "ok"

    out = await _dispatch_with_retry(_call)
    assert out == "ok"
    assert call_count["n"] == 1


# ── Default budget fits within bridge timeout (Cursor bot finding #2) ─


def test_default_retry_budget_fits_within_default_bridge_timeout():
    """`bridge_build_timeout_ms` defaults to 8000 (8s). The default judge
    retry budget — max backoff sum across all retries — must fit within
    that budget with enough headroom for the LLM calls themselves.
    Otherwise `asyncio.wait_for` in BridgeService.build() cancels the
    retry loop before retries finish, defeating the layer entirely."""
    # Equal-jitter worst-case backoff sum (sum of upper bounds across
    # retries): sum_{i=0}^{N-1} base * 2^i.
    max_retries = judge_mod._JUDGE_DISPATCH_MAX_RETRIES
    base = judge_mod._JUDGE_DISPATCH_BACKOFF_BASE_S
    worst_case_backoff_s = sum(base * (2**i) for i in range(max_retries))

    # Reserve ≥3.5s for actual LLM attempts (Gemini Flash ~1-1.5s/call,
    # 3 attempts at default). The remaining budget for backoff must
    # leave that headroom within an 8s per-bridge timeout.
    DEFAULT_BRIDGE_TIMEOUT_S = 8.0
    MIN_HEADROOM_FOR_LLM_S = 3.5
    available_for_backoff_s = DEFAULT_BRIDGE_TIMEOUT_S - MIN_HEADROOM_FOR_LLM_S

    assert worst_case_backoff_s <= available_for_backoff_s, (
        f"default retry budget ({worst_case_backoff_s:.1f}s backoff) leaves "
        f"insufficient headroom for {max_retries + 1} LLM attempts within "
        f"the {DEFAULT_BRIDGE_TIMEOUT_S}s default bridge timeout. "
        f"asyncio.wait_for in BridgeService.build() will cancel the retry loop. "
        f"Reduce DISSONANCE_JUDGE_MAX_RETRIES or DISSONANCE_JUDGE_BACKOFF_BASE_S."
    )
