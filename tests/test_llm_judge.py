"""Tests for LLMJudge — batching, scoring, caching, and error handling.

Unit tests mock the Gemini client so no API credentials are needed.
Smoke tests hit the real Gemini 2.5 Flash API and are marked ``heavy``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_agent.judge import LLMJudge

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _resp(scores: list[dict]) -> MagicMock:
    """Fake Gemini response whose .text is a well-formed score payload."""
    m = MagicMock()
    m.text = json.dumps({"scores": scores})
    return m


def _neutral_resp(n: int) -> MagicMock:
    return _resp([{"pair": i, "entail": 0.0, "contradict": 0.0} for i in range(n)])


def _patch_client(responses):
    """
    Patch google.genai.Client so aio.models.generate_content is an AsyncMock.

    ``responses``: single MagicMock → return_value; list → side_effect (one
    item consumed per call, in call order).
    """
    mock_client = MagicMock()
    if isinstance(responses, list):
        mock_client.aio.models.generate_content = AsyncMock(side_effect=responses)
    else:
        mock_client.aio.models.generate_content = AsyncMock(return_value=responses)
    ctx = patch("google.genai.Client", return_value=mock_client)
    return ctx, mock_client


def _judge(tmp_path: Path, **kw) -> LLMJudge:
    """LLMJudge with an isolated cache dir so tests don't share state."""
    return LLMJudge(cache_dir=tmp_path, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_small_batch_makes_exactly_one_api_call(tmp_path):
    """Three pairs fit within pairs_per_request=20 → exactly one Gemini call."""
    pairs = [("A", "B"), ("C", "D"), ("E", "F")]
    resp = _resp(
        [
            {"pair": 0, "entail": 0.9, "contradict": 0.05},
            {"pair": 1, "entail": 0.05, "contradict": 0.9},
            {"pair": 2, "entail": 0.3, "contradict": 0.2},
        ]
    )
    ctx, mock_client = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path).score_batch(pairs)

    mock_client.aio.models.generate_content.assert_called_once()
    assert len(scores) == 3


@pytest.mark.asyncio
async def test_large_batch_splits_into_correct_number_of_chunks(tmp_path):
    """7 pairs with pairs_per_request=3 → ceil(7/3) = 3 API calls."""
    pairs = [(f"q{i}", f"c{i}") for i in range(7)]
    ctx, mock_client = _patch_client(
        [
            _neutral_resp(3),
            _neutral_resp(3),
            _neutral_resp(1),
        ]
    )
    with ctx:
        scores = await _judge(tmp_path, pairs_per_request=3).score_batch(pairs)

    assert mock_client.aio.models.generate_content.call_count == 3
    assert len(scores) == 7
    assert all(s == 0.0 for s in scores)


@pytest.mark.asyncio
async def test_chunk_boundary_exact_multiple(tmp_path):
    """6 pairs with pairs_per_request=3 → exactly 2 calls, none partial."""
    pairs = [(f"q{i}", f"c{i}") for i in range(6)]
    ctx, mock_client = _patch_client([_neutral_resp(3), _neutral_resp(3)])
    with ctx:
        scores = await _judge(tmp_path, pairs_per_request=3).score_batch(pairs)

    assert mock_client.aio.models.generate_content.call_count == 2
    assert len(scores) == 6


@pytest.mark.asyncio
async def test_empty_batch_skips_api_and_returns_empty_list(tmp_path):
    ctx, mock_client = _patch_client([])
    with ctx:
        scores = await _judge(tmp_path).score_batch([])

    mock_client.aio.models.generate_content.assert_not_called()
    assert scores == []


# ──────────────────────────────────────────────────────────────────────────────
# Prompt content
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_contains_all_pairs_with_local_chunk_indices(tmp_path):
    """The prompt bundled into the API call must list all pairs as Pair 0, Pair 1, …"""
    pairs = [("query alpha", "cand alpha"), ("query beta", "cand beta")]
    ctx, mock_client = _patch_client(
        _resp(
            [
                {"pair": 0, "entail": 0.8, "contradict": 0.1},
                {"pair": 1, "entail": 0.1, "contradict": 0.8},
            ]
        )
    )
    with ctx:
        await _judge(tmp_path).score_batch(pairs)

    call_kw = mock_client.aio.models.generate_content.call_args
    prompt = call_kw.kwargs["contents"]
    assert "query alpha" in prompt
    assert "cand alpha" in prompt
    assert "query beta" in prompt
    assert "cand beta" in prompt
    assert "Pair 0" in prompt
    assert "Pair 1" in prompt


@pytest.mark.asyncio
async def test_model_name_forwarded_to_api(tmp_path):
    """The model kwarg passed to LLMJudge must be forwarded in the API call."""
    ctx, mock_client = _patch_client(_neutral_resp(1))
    with ctx:
        await _judge(tmp_path, model="gemini-2.5-flash").score_batch([("q", "c")])

    call_kw = mock_client.aio.models.generate_content.call_args
    model_arg = call_kw.kwargs["model"]
    assert model_arg == "gemini-2.5-flash"


# ──────────────────────────────────────────────────────────────────────────────
# Score formula
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_score_is_entail_minus_contradict(tmp_path):
    """score = entail - contradict."""
    cases = [
        (0.8, 0.1, pytest.approx(0.7, abs=1e-6)),
        (0.0, 0.9, pytest.approx(-0.9, abs=1e-6)),
        (0.5, 0.5, pytest.approx(0.0, abs=1e-6)),
    ]
    resp = _resp(
        [{"pair": i, "entail": e, "contradict": c} for i, (e, c, _) in enumerate(cases)]
    )
    ctx, _ = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path, pairs_per_request=10).score_batch(
            [(f"q{i}", f"c{i}") for i in range(len(cases))]
        )

    for i, (_, _, expected) in enumerate(cases):
        assert scores[i] == expected, f"case {i}"


@pytest.mark.asyncio
async def test_score_is_clamped_to_minus_one_and_plus_one(tmp_path):
    """score stays within [-1.0, 1.0] at the boundary values."""
    resp = _resp(
        [
            {"pair": 0, "entail": 1.0, "contradict": 0.0},  # → +1.0 (ceiling)
            {"pair": 1, "entail": 0.0, "contradict": 1.0},  # → -1.0 (floor)
        ]
    )
    ctx, _ = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path).score_batch([("q0", "c0"), ("q1", "c1")])

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(-1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_call_with_same_pairs_hits_cache_not_api(tmp_path):
    resp = _resp([{"pair": 0, "entail": 0.95, "contradict": 0.02}])
    ctx, mock_client = _patch_client(resp)
    with ctx:
        j = _judge(tmp_path)
        scores1 = await j.score_batch([("the sky is blue", "sky is blue")])
        scores2 = await j.score_batch([("the sky is blue", "sky is blue")])

    mock_client.aio.models.generate_content.assert_called_once()
    assert scores1 == scores2
    assert j.stats["cache_hits"] == 1
    assert j.stats["cache_misses"] == 1


@pytest.mark.asyncio
async def test_partial_cache_overlap_calls_api_only_for_misses(tmp_path):
    """If 1 of 3 pairs is cached, the API receives a 2-pair chunk, not 3."""
    pair_a = ("q0", "c0")
    all_pairs = [pair_a, ("q1", "c1"), ("q2", "c2")]

    resp_first = _resp([{"pair": 0, "entail": 0.8, "contradict": 0.1}])
    resp_second = _resp(
        [
            {"pair": 0, "entail": 0.5, "contradict": 0.5},
            {"pair": 1, "entail": 0.2, "contradict": 0.7},
        ]
    )
    ctx, mock_client = _patch_client([resp_first, resp_second])
    with ctx:
        j = _judge(tmp_path)
        await j.score_batch([pair_a])  # caches pair_a
        scores = await j.score_batch(all_pairs)

    assert mock_client.aio.models.generate_content.call_count == 2
    # First call: 1 pair; second call: 2 pairs (q1, q2)
    second_prompt = mock_client.aio.models.generate_content.call_args_list[1]
    contents = second_prompt.kwargs["contents"]
    assert "q0" not in contents, "cached pair must not be re-sent to the API"
    assert "q1" in contents
    assert "q2" in contents
    assert len(scores) == 3


@pytest.mark.asyncio
async def test_cache_key_is_model_sensitive(tmp_path):
    """Different models must not share cache entries."""
    pair = ("query", "candidate")
    resp_a = _resp([{"pair": 0, "entail": 0.9, "contradict": 0.0}])
    resp_b = _resp([{"pair": 0, "entail": 0.1, "contradict": 0.8}])
    ctx, mock_client = _patch_client([resp_a, resp_b])
    with ctx:
        ja = _judge(tmp_path, model="gemini-2.5-flash")
        jb = _judge(tmp_path, model="gemini-2.5-pro")
        sa = await ja.score_batch([pair])
        sb = await jb.score_batch([pair])

    assert mock_client.aio.models.generate_content.call_count == 2
    assert sa[0] != sb[0]


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing edge cases
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_json_returns_zeros_without_raising(tmp_path):
    """Non-JSON from the model → all pairs in the chunk score 0.0, no exception."""
    bad = MagicMock()
    bad.text = "I cannot assist with that."
    ctx, _ = _patch_client(bad)
    with ctx:
        scores = await _judge(tmp_path).score_batch([("q0", "c0"), ("q1", "c1")])

    assert scores == [0.0, 0.0]


@pytest.mark.asyncio
async def test_missing_pair_index_in_response_scores_zero(tmp_path):
    """If the model omits a pair index in its response, that pair scores 0.0."""
    resp = _resp(
        [
            {"pair": 0, "entail": 0.9, "contradict": 0.05},
            # pair 1 absent
            {"pair": 2, "entail": 0.1, "contradict": 0.8},
        ]
    )
    ctx, _ = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path).score_batch(
            [("q0", "c0"), ("q1", "c1"), ("q2", "c2")]
        )

    assert scores[0] == pytest.approx(0.85, abs=1e-6)
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(-0.7, abs=1e-6)


@pytest.mark.asyncio
async def test_out_of_range_pair_index_is_silently_ignored(tmp_path):
    """A 'pair' index outside [0, chunk_size) must be dropped, not crash."""
    resp = _resp(
        [
            {"pair": 0, "entail": 0.9, "contradict": 0.0},
            {"pair": 99, "entail": 0.5, "contradict": 0.5},  # invalid
        ]
    )
    ctx, _ = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path).score_batch([("q", "c")])

    assert len(scores) == 1
    assert scores[0] == pytest.approx(0.9, abs=1e-6)


@pytest.mark.asyncio
async def test_null_entail_contradict_treated_as_zero(tmp_path):
    """'entail': null (or missing) must not crash — treated as 0.0."""
    resp = _resp([{"pair": 0, "entail": None, "contradict": None}])
    ctx, _ = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path).score_batch([("q", "c")])

    assert scores[0] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_null_pair_key_in_response_is_skipped(tmp_path):
    """A response entry with 'pair': null must be skipped without crashing."""
    resp = _resp([{"pair": None, "entail": 0.9, "contradict": 0.0}])
    ctx, _ = _patch_client(resp)
    with ctx:
        scores = await _judge(tmp_path).score_batch([("q", "c")])

    assert scores[0] == 0.0  # entry skipped → default


# ──────────────────────────────────────────────────────────────────────────────
# Stats counters
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_track_requests_pairs_and_cache_counters(tmp_path):
    ctx, _ = _patch_client(_neutral_resp(1))
    with ctx:
        j = _judge(tmp_path)
        await j.score_batch([("q", "c")])

    s = j.stats
    assert s["requests"] == 1
    assert s["pairs"] == 1
    assert s["cache_misses"] == 1
    assert s["cache_hits"] == 0


@pytest.mark.asyncio
async def test_stats_accumulate_across_calls(tmp_path):
    """Total pairs and cache counters must accumulate over multiple score_batch calls."""
    ctx, _ = _patch_client(_neutral_resp(2))
    with ctx:
        j = _judge(tmp_path)
        await j.score_batch([("q0", "c0"), ("q1", "c1")])
        await j.score_batch([("q0", "c0")])  # cache hit + 0 API calls

    s = j.stats
    assert s["pairs"] == 3  # 2 + 1
    assert s["requests"] == 1  # only the first call hit the API (second was all-cached)
    assert s["cache_misses"] == 2
    assert s["cache_hits"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Smoke tests — real Gemini 2.5 Flash (marked heavy, need ADC / GOOGLE_API_KEY)
# ──────────────────────────────────────────────────────────────────────────────

COHERENT: list[tuple[str, str]] = [
    ("She has three rescue dogs and a tabby cat at home.", "She has pets."),
    (
        "Global mean temperatures have risen by about 1°C since 1900.",
        "The climate is warming.",
    ),
    ("The team scored five goals and won the match easily.", "The team won the match."),
]

CONTRADICTORY: list[tuple[str, str]] = [
    (
        "She is a strict vegetarian who never eats meat.",
        "She enjoys eating steak every weekend.",
    ),
    ("The store is closed today for a national holiday.", "The store is open today."),
    (
        "The patient's blood pressure is dangerously low.",
        "The patient's blood pressure is dangerously high.",
    ),
]

UNRELATED: list[tuple[str, str]] = [
    (
        "She loves animals.",
        "Quantum entanglement is a non-local correlation between particles.",
    ),
    (
        "The team won the championship.",
        "Mountain ranges are formed by tectonic plate collisions.",
    ),
    ("She is a vegetarian.", "The library closes at 9 pm on weekdays."),
]

ENTAIL_FLOOR = 0.3
CONTRADICT_CEIL = -0.3
NEUTRAL_BAND = 0.4


@pytest.mark.asyncio
@pytest.mark.heavy
async def test_llm_judge_gemini_25_flash_buckets_belief_pairs_correctly(tmp_path):
    """Gemini 2.5 Flash must score coherent > neutral > contradictory.

    All 9 pairs fit in one chunk (pairs_per_request=20), so this also verifies
    that a single batched API call returns a correctly parsed, full result set.
    """
    judge = LLMJudge(model="gemini-2.5-flash", cache_dir=tmp_path)

    pairs: list[tuple[str, str]] = []
    labels: list[str] = []
    for premise, hypothesis in COHERENT:
        pairs.append((hypothesis, premise))
        labels.append("coherent")
    for premise, hypothesis in CONTRADICTORY:
        pairs.append((hypothesis, premise))
        labels.append("contradictory")
    for premise, hypothesis in UNRELATED:
        pairs.append((hypothesis, premise))
        labels.append("unrelated")

    scores = await judge.score_batch(pairs)

    coherent_scores = [s for s, lbl in zip(scores, labels) if lbl == "coherent"]
    contradict_scores = [s for s, lbl in zip(scores, labels) if lbl == "contradictory"]
    unrelated_scores = [s for s, lbl in zip(scores, labels) if lbl == "unrelated"]

    failures: list[str] = []
    for score, (premise, hypothesis) in zip(coherent_scores, COHERENT):
        if score < ENTAIL_FLOOR:
            failures.append(
                f"COHERENT {score:+.3f} < {ENTAIL_FLOOR}: {premise!r} ⊨ {hypothesis!r}"
            )
    for score, (premise, hypothesis) in zip(contradict_scores, CONTRADICTORY):
        if score > CONTRADICT_CEIL:
            failures.append(
                f"CONTRADICTORY {score:+.3f} > {CONTRADICT_CEIL}: {premise!r} ⊥ {hypothesis!r}"
            )
    for score, (premise, hypothesis) in zip(unrelated_scores, UNRELATED):
        if abs(score) > NEUTRAL_BAND:
            failures.append(
                f"UNRELATED |{score:+.3f}| > {NEUTRAL_BAND}: {premise!r} ∥ {hypothesis!r}"
            )

    if max(coherent_scores) <= max(unrelated_scores):
        failures.append(
            f"ordering: max(coherent)={max(coherent_scores):+.3f} "
            f"≤ max(unrelated)={max(unrelated_scores):+.3f}"
        )
    if min(contradict_scores) >= min(unrelated_scores):
        failures.append(
            f"ordering: min(contradictory)={min(contradict_scores):+.3f} "
            f"≥ min(unrelated)={min(unrelated_scores):+.3f}"
        )

    # 9 pairs < pairs_per_request=20 → must be exactly one API call
    assert (
        judge.stats["requests"] == 1
    ), f"expected 1 batched API call for 9 pairs, got {judge.stats['requests']}"

    assert not failures, "LLM judge smoke failures:\n  " + "\n  ".join(failures)
