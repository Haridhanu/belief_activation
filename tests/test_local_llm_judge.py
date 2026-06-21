"""Tests for LocalLLMJudge.

Unit tests mock the model and tokenizer so no GPU or downloaded weights are needed.
Tests for _extract_json are pure-function and have no external dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch

from multi_agent.judge import LocalLLMJudge

# ── _extract_json (pure function, no GPU) ────────────────────────────────────


def test_extract_json_fenced_with_language_tag():
    text = '```json\n{"entail": 0.9, "contradict": 0.0}\n```'
    assert json.loads(LocalLLMJudge._extract_json(text)) == {
        "entail": 0.9,
        "contradict": 0.0,
    }


def test_extract_json_fenced_without_language_tag():
    text = '```\n{"entail": 0.5, "contradict": 0.5}\n```'
    assert json.loads(LocalLLMJudge._extract_json(text)) == {
        "entail": 0.5,
        "contradict": 0.5,
    }


def test_extract_json_unfenced_with_leading_prose():
    text = 'Sure, here are the results: {"entail": 0.7, "contradict": 0.1}'
    parsed = json.loads(LocalLLMJudge._extract_json(text))
    assert parsed["entail"] == pytest.approx(0.7)


def test_extract_json_nested_braces_not_truncated():
    """rfind('}') must find the outermost closing brace, not stop at a nested one.

    If find('}') were used, extraction would stop at the first '}' inside the
    nested dict and produce invalid JSON.
    """
    text = '{"entail": 0.9, "contradict": 0.0, "meta": {"model": "test"}}'
    parsed = json.loads(LocalLLMJudge._extract_json(text))
    assert parsed["entail"] == pytest.approx(0.9)
    assert parsed["meta"] == {"model": "test"}


def test_extract_json_no_json_returns_original():
    text = "I cannot assist with that request."
    assert LocalLLMJudge._extract_json(text) == text


def test_extract_json_empty_string():
    assert LocalLLMJudge._extract_json("") == ""


# ── _parse_single (pure-ish, no GPU) ─────────────────────────────────────────


def test_parse_single_happy_path():
    judge = LocalLLMJudge.__new__(LocalLLMJudge)
    judge._parse_failures = 0
    text = json.dumps({"entail": 0.9, "contradict": 0.05})
    e, c = judge._parse_single(text)
    assert e == pytest.approx(0.9)
    assert c == pytest.approx(0.05)
    assert judge._parse_failures == 0


def test_parse_single_clamps_values():
    judge = LocalLLMJudge.__new__(LocalLLMJudge)
    judge._parse_failures = 0
    text = json.dumps({"entail": 1.5, "contradict": -0.3})
    e, c = judge._parse_single(text)
    assert e == pytest.approx(1.0)
    assert c == pytest.approx(0.0)


def test_parse_single_null_values_are_schema_error():
    """null entail/contradict is not a valid numeric score — counts as parse failure."""
    judge = LocalLLMJudge.__new__(LocalLLMJudge)
    judge._parse_failures = 0
    text = json.dumps({"entail": None, "contradict": None})
    e, c = judge._parse_single(text)
    assert e == 0.0 and c == 0.0
    assert judge._parse_failures == 1


def test_parse_single_missing_keys_are_schema_error():
    """JSON with wrong schema (missing entail/contradict) counts as parse failure."""
    judge = LocalLLMJudge.__new__(LocalLLMJudge)
    judge._parse_failures = 0
    # Old bundle format or renamed keys — wrong schema
    text = json.dumps({"scores": [{"pair": 0, "entail": 0.9, "contradict": 0.1}]})
    e, c = judge._parse_single(text)
    assert e == 0.0 and c == 0.0
    assert judge._parse_failures == 1


def test_parse_single_malformed_increments_failures():
    judge = LocalLLMJudge.__new__(LocalLLMJudge)
    judge._parse_failures = 0
    e, c = judge._parse_single("not json at all")
    assert e == 0.0 and c == 0.0
    assert judge._parse_failures == 1


# ── Helpers for mocked-model tests ──────────────────────────────────────────


def _make_judge(decode_responses: list[str]) -> LocalLLMJudge:
    """Return a LocalLLMJudge whose model and tokenizer are fully mocked.

    decode_responses is a list of per-pair output strings; decode() cycles
    through them in order (one call per output sequence).
    """
    judge = LocalLLMJudge(model_name="test-model", device="cpu")

    mock_tokenizer = MagicMock()
    # apply_chat_template called once per pair; returns [1, 10] tensor.
    mock_tokenizer.apply_chat_template.return_value = torch.zeros(
        1, 10, dtype=torch.long
    )
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.side_effect = decode_responses

    def _generate(**kwargs):
        # Return [N, input_len + 5] so out[input_len:] gives 5 decodable tokens.
        n = kwargs["input_ids"].shape[0]
        total_len = kwargs["input_ids"].shape[1] + 5
        return torch.zeros(n, total_len, dtype=torch.long)

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    mock_model.generate.side_effect = _generate
    mock_model.config.max_position_embeddings = 8192

    judge._tokenizer = mock_tokenizer
    judge._model = mock_model
    judge._actual_device = "cpu"
    return judge


# ── _score_chunk_sync: BatchEncoding regression ─────────────────────────────


def test_score_chunk_sync_batch_encoding_return_type():
    """Regression: apply_chat_template returning BatchEncoding dict (transformers>=4.44)
    must not crash with 'tokenizers.Encoding has no attribute shape'."""
    judge = LocalLLMJudge(model_name="test-model", device="cpu")

    from transformers import BatchEncoding

    mock_tokenizer = MagicMock()
    # Simulate transformers>=4.44: apply_chat_template returns a real BatchEncoding
    # (UserDict subclass, NOT a dict subclass — so isinstance(x, dict) is False).
    mock_tokenizer.apply_chat_template.return_value = BatchEncoding(
        {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
    )
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode.return_value = '{"entail": 0.8, "contradict": 0.1}'

    def _generate(**kwargs):
        n = kwargs["input_ids"].shape[0]
        return torch.zeros(n, kwargs["input_ids"].shape[1] + 5, dtype=torch.long)

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    mock_model.generate.side_effect = _generate
    mock_model.config.max_position_embeddings = 8192

    judge._tokenizer = mock_tokenizer
    judge._model = mock_model
    judge._actual_device = "cpu"

    # Must not raise; should return a numeric score
    scores = judge._score_chunk_sync([("The sky is blue.", "It has a blue sky.")])
    assert len(scores) == 1
    assert isinstance(scores[0], float)


# ── _score_chunk_sync: happy path ───────────────────────────────────────────


def test_score_chunk_sync_single_pair():
    """Single pair → correct entail-minus-contradict score."""
    judge = _make_judge([json.dumps({"entail": 0.9, "contradict": 0.05})])
    scores = judge._score_chunk_sync([("A", "B")])
    assert len(scores) == 1
    assert scores[0] == pytest.approx(0.85, abs=1e-6)
    assert judge._parse_failures == 0
    assert judge._total_chunks == 1


def test_score_chunk_sync_multiple_pairs_scored_independently():
    """N pairs → N independent decode calls, each returning its own JSON."""
    responses = [
        json.dumps({"entail": 0.9, "contradict": 0.0}),
        json.dumps({"entail": 0.0, "contradict": 0.8}),
        json.dumps({"entail": 0.1, "contradict": 0.15}),
    ]
    judge = _make_judge(responses)
    scores = judge._score_chunk_sync([("A", "B"), ("C", "D"), ("E", "F")])

    assert len(scores) == 3
    assert scores[0] == pytest.approx(0.9, abs=1e-6)
    assert scores[1] == pytest.approx(-0.8, abs=1e-6)
    assert scores[2] == pytest.approx(-0.05, abs=1e-6)
    assert judge._parse_failures == 0
    assert judge._total_chunks == 1  # one batched generate() call


def test_score_chunk_sync_batched_generate_called_once():
    """model.generate() is called exactly once for any chunk size."""
    n = 5
    responses = [json.dumps({"entail": 0.5, "contradict": 0.0})] * n
    judge = _make_judge(responses)
    judge._score_chunk_sync([("q", "c")] * n)
    assert judge._model.generate.call_count == 1


def test_score_chunk_sync_generate_receives_batch_tensor():
    """generate() input_ids batch dimension equals chunk size."""
    n = 4
    responses = [json.dumps({"entail": 0.5, "contradict": 0.1})] * n
    judge = _make_judge(responses)
    judge._score_chunk_sync([("q", "c")] * n)
    call_kwargs = judge._model.generate.call_args.kwargs
    assert call_kwargs["input_ids"].shape[0] == n


def test_score_chunk_sync_score_clamped_to_minus_one_plus_one():
    """entail - contradict is clamped to [-1.0, 1.0]."""
    responses = [
        json.dumps({"entail": 1.0, "contradict": 0.0}),
        json.dumps({"entail": 0.0, "contradict": 1.0}),
    ]
    scores = _make_judge(responses)._score_chunk_sync([("q0", "c0"), ("q1", "c1")])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(-1.0)


def test_score_chunk_sync_max_new_tokens_is_fixed():
    """generate() is always called with the fixed _MAX_NEW_TOKENS budget."""
    responses = [json.dumps({"entail": 0.5, "contradict": 0.1})] * 3
    judge = _make_judge(responses)
    judge._score_chunk_sync([("q", "c")] * 3)
    call_kwargs = judge._model.generate.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == LocalLLMJudge._MAX_NEW_TOKENS


# ── _score_chunk_sync: failure modes ────────────────────────────────────────


def test_score_chunk_sync_malformed_json_returns_neutral_per_pair():
    """Non-JSON output for a pair → 0.0 score; parse_failures counts each bad pair."""
    responses = ["not json", json.dumps({"entail": 0.8, "contradict": 0.0})]
    judge = _make_judge(responses)
    scores = judge._score_chunk_sync([("q0", "c0"), ("q1", "c1")])
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.8)
    assert judge._parse_failures == 1


def test_score_chunk_sync_all_pairs_over_budget_returns_neutral():
    """All pairs over context budget → neutral scores, generate() not called."""
    judge = _make_judge([])
    judge._model.config.max_position_embeddings = 15  # tiny context
    # apply_chat_template returns [1, 10]; 10 + 48 = 58 > 15
    scores = judge._score_chunk_sync([("q", "c")])
    assert scores == [0.0]
    assert judge._budget_exceeded == 1
    assert judge._parse_failures == 0
    judge._model.generate.assert_not_called()


def test_score_chunk_sync_one_oversized_pair_does_not_block_others():
    """One over-budget pair gets neutral; the remaining valid pairs are still scored."""
    short_response = json.dumps({"entail": 0.9, "contradict": 0.0})
    judge = _make_judge([short_response, short_response])

    # Make apply_chat_template return a long tensor for pair 0, short for pairs 1 and 2.
    long_ids = torch.zeros(
        1, 200, dtype=torch.long
    )  # 200 + 48 = 248, will exceed limit
    short_ids = torch.zeros(1, 5, dtype=torch.long)  # 5 + 48 = 53, fits in 100
    judge._tokenizer.apply_chat_template.side_effect = [long_ids, short_ids, short_ids]
    judge._model.config.max_position_embeddings = 100

    scores = judge._score_chunk_sync([("long", "pair"), ("q1", "c1"), ("q2", "c2")])

    assert scores[0] == pytest.approx(0.0)  # over-budget → neutral
    assert scores[1] == pytest.approx(0.9)  # scored normally
    assert scores[2] == pytest.approx(0.9)  # scored normally
    assert judge._budget_exceeded == 1
    # generate() was called once for the two viable pairs
    assert judge._model.generate.call_count == 1
    assert judge._model.generate.call_args.kwargs["input_ids"].shape[0] == 2


def test_score_chunk_sync_left_pads_unequal_length_viable_pairs():
    """Cover the left-padding path: two viable pairs of different lengths must
    be aligned to the longest with attention_mask zeros on the leading pad,
    and each pair must decode from the same input-length offset."""
    short_response = json.dumps({"entail": 0.7, "contradict": 0.0})
    long_response = json.dumps({"entail": 0.0, "contradict": 0.4})
    judge = _make_judge([short_response, long_response])

    short_ids = torch.full((1, 5), 7, dtype=torch.long)  # token id 7
    long_ids = torch.full((1, 12), 9, dtype=torch.long)  # token id 9
    judge._tokenizer.apply_chat_template.side_effect = [short_ids, long_ids]
    judge._tokenizer.pad_token_id = 0
    judge._model.config.max_position_embeddings = 8192

    scores = judge._score_chunk_sync([("short", "p"), ("longer", "p")])

    call_kwargs = judge._model.generate.call_args.kwargs
    input_ids = call_kwargs["input_ids"]
    attention_mask = call_kwargs["attention_mask"]

    # Both sequences padded to length 12 = max(5, 12).
    assert input_ids.shape == (2, 12)
    assert attention_mask.shape == (2, 12)

    # Row 0 (shorter): 7 zeros (pad) on the LEFT, then 5 real tokens of value 7.
    assert input_ids[0, :7].tolist() == [0] * 7
    assert input_ids[0, 7:].tolist() == [7] * 5
    assert attention_mask[0, :7].tolist() == [0] * 7
    assert attention_mask[0, 7:].tolist() == [1] * 5

    # Row 1 (longer): no padding, attention mask all ones.
    assert input_ids[1].tolist() == [9] * 12
    assert attention_mask[1].tolist() == [1] * 12

    # Both pairs were scored using the values from their respective decode strings.
    assert scores[0] == pytest.approx(0.7)
    assert scores[1] == pytest.approx(-0.4)


def test_parse_single_rejects_bool_as_numeric():
    """`isinstance(True, (int, float))` is True in CPython — bool subclasses
    int — so an unguarded numeric check would parse {"entail": true} as 1.0.
    Treat bools as a schema error instead."""
    judge = _make_judge([])
    e, c = judge._parse_single(json.dumps({"entail": True, "contradict": False}))
    assert (e, c) == (0.0, 0.0)
    assert judge._parse_failures == 1


def test_pair_prompt_format_renders_with_repr_args():
    """The prompt template's literal braces must survive format(); a malformed
    escape would raise KeyError or IndexError before generate() is ever called."""
    rendered = LocalLLMJudge._PAIR_PROMPT.format(q=repr("hello"), c=repr("world"))
    assert "'hello'" in rendered
    assert "'world'" in rendered
    assert "{{" not in rendered  # all double-braces consumed by .format()


def test_score_chunk_sync_adds_special_tokens_false():
    """apply_chat_template must be called with add_special_tokens=False to avoid
    duplicated BOS on chat models like Gemma."""
    judge = _make_judge([json.dumps({"entail": 0.5, "contradict": 0.0})])
    judge._score_chunk_sync([("q", "c")])
    call_kwargs = judge._tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("add_special_tokens") is False


# ── score_batch: chunking ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_score_batch_splits_into_chunks():
    """5 pairs with pairs_per_request=2 → 3 _score_chunk_sync calls → 3 generate() calls."""
    per_pair = json.dumps({"entail": 0.5, "contradict": 0.0})
    judge = _make_judge([per_pair] * 5)
    judge._pairs_per_request = 2

    scores = await judge.score_batch([(f"q{i}", f"c{i}") for i in range(5)])

    assert len(scores) == 5
    assert judge._total_pairs == 5
    assert judge._model.generate.call_count == 3  # ceil(5/2) = 3


@pytest.mark.asyncio
async def test_score_batch_empty_returns_empty():
    judge = _make_judge([])
    scores = await judge.score_batch([])
    assert scores == []
    assert judge._total_pairs == 0


# ── stats ────────────────────────────────────────────────────────────────────


def test_stats_before_load_has_zero_counters():
    judge = LocalLLMJudge()
    s = judge.stats
    assert s["pairs"] == 0
    assert s["chunks"] == 0
    assert s["parse_failures"] == 0
    assert s["budget_exceeded"] == 0
    assert s["load_time_sec"] == 0.0
    assert s["device"] is None
    assert "vram_allocated_bytes" not in s


def test_stats_after_scoring_reflect_actual_counts():
    judge = _make_judge([json.dumps({"entail": 0.8, "contradict": 0.1})])
    judge._score_chunk_sync([("q", "c")])

    s = judge.stats
    assert s["chunks"] == 1
    assert s["parse_failures"] == 0
    assert s["device"] == "cpu"
    assert "vram_allocated_bytes" not in s


def test_stats_parse_failures_accumulate():
    judge = _make_judge(["not json"] * 4)
    judge._score_chunk_sync([("q", "c"), ("q2", "c2")])
    judge._score_chunk_sync([("q", "c"), ("q2", "c2")])
    assert judge.stats["parse_failures"] == 4
