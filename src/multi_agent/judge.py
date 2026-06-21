from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import threading
from pathlib import Path
from typing import Awaitable, Callable, Protocol
import torch
import time

logger = logging.getLogger(__name__)


# Pin sampling seed across all judge LLM calls for cross-call determinism
# on top of temperature=0. Even at temp=0, Gemini drifts without a seed
# because of upstream batching / token-tie-breaking. This seed locks
# (prompt, model, seed) → identical output. Override via env var if a
# different seed is needed for A/B comparison or bisecting.
#
# NOTE: this stabilizes the judge call itself, but bs_merge has another
# determinism source — beam_runner's RNG is seeded from session-prefixed
# node IDs in `f"{start}|{end}"`, so identical sessions always yield the
# same beam paths but identical *content* across new sessions does not.
_JUDGE_SEED = int(os.getenv("BS_MERGE_JUDGE_SEED", "42"))


# Retry config for the judge LLM dispatch. Vertex AI returns 429
# RESOURCE_EXHAUSTED under quota pressure; a brief exponential backoff
# usually recovers within seconds. Without retry, a single quota error on
# the only candidate pair in a 2-paper merge propagates as 1/1 bridge
# failures → 502, even though the underlying service is healthy.
# (Linear: PLA-247.)
#
# Defaults are tuned to fit within the default per-bridge timeout
# (`bridge_build_timeout_ms` = 8000 in MergeRequest). With max_retries=2
# and base=1.5 + equal jitter:
#   nominal backoffs: 1.5s, 3.0s
#   jittered worst case (sum of upper bounds): 1.5 + 3.0 = 4.5s
#   leaves ~3.5s for 3 LLM attempts (~1.2s each — typical Gemini Flash).
# A bigger budget gets eaten by `asyncio.wait_for` in BridgeService.build()
# before retries finish, defeating the retry layer.
_JUDGE_DISPATCH_MAX_RETRIES = int(os.getenv("DISSONANCE_JUDGE_MAX_RETRIES", "2"))
_JUDGE_DISPATCH_BACKOFF_BASE_S = float(
    os.getenv("DISSONANCE_JUDGE_BACKOFF_BASE_S", "1.5")
)


# Anchor 429 matching at word boundaries so unrelated tokens that happen
# to contain "429" (request IDs, byte counts, timestamps) don't trigger
# spurious retries. RESOURCE_EXHAUSTED is a canonical Vertex status string
# and unambiguous on its own.
_RATE_LIMIT_429_PATTERN = re.compile(r"\b429\b")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if exc represents a transient rate-limit / quota condition that
    a brief backoff is likely to recover from (Vertex 429 / Gemini
    RESOURCE_EXHAUSTED or OpenAI RateLimitError)."""
    if type(exc).__name__ == "RateLimitError":  # openai.RateLimitError
        return True
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg:
        return True
    return bool(_RATE_LIMIT_429_PATTERN.search(msg))


async def _dispatch_with_retry(
    coro_factory: Callable[[], Awaitable[str]],
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Run an async LLM dispatch with exponential-backoff retry on
    rate-limit errors. `coro_factory` is a no-arg callable returning a
    fresh coroutine each call (a coroutine can only be awaited once).

    Uses "equal jitter" — actual wait is uniformly distributed in
    [nominal/2, nominal] — so concurrent dispatches that rate-limit
    together don't synchronize their retries and re-collide on the same
    upstream quota bucket. (`BS_MERGE_BUILD_CONCURRENCY` defaults to 8;
    without jitter, all 8 would retry in lockstep.)

    If `semaphore` is provided, it's acquired *per-attempt* — the slot
    is released during backoff sleeps so parallel dispatches that aren't
    rate-limited can make progress. Holding the slot across multi-second
    sleeps would starve the concurrency pool and risk pushing every
    in-flight bridge past `bridge_build_timeout_ms`.
    """
    for attempt in range(_JUDGE_DISPATCH_MAX_RETRIES + 1):
        try:
            if semaphore is not None:
                async with semaphore:
                    return await coro_factory()
            return await coro_factory()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == _JUDGE_DISPATCH_MAX_RETRIES:
                raise
            nominal_s = _JUDGE_DISPATCH_BACKOFF_BASE_S * (2**attempt)
            wait_s = nominal_s / 2 + random.uniform(0, nominal_s / 2)
            logger.warning(
                "DissonanceJudge rate-limited (attempt %d/%d, "
                "retrying in %.1fs): %s",
                attempt + 1,
                _JUDGE_DISPATCH_MAX_RETRIES + 1,
                wait_s,
                exc,
            )
            await asyncio.sleep(wait_s)
    # The loop returns or raises on every path; this line is unreachable.
    raise RuntimeError("dispatch_with_retry exited loop without returning")


# USD per 1M tokens. Keyed by model substring (first match wins). Update when
# Google publishes new prices or when we switch model tiers. The trend
# dashboard renders cumulative_cost_usd from these — wrong numbers here
# silently mislead config A/B comparisons, so prefer "missing" (returns 0)
# over "stale guess" when in doubt.
_MODEL_PRICING_USD_PER_1M = {
    # Gemini 2.5 Flash (text in / text out, non-thinking output rate).
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-flash-latest": {"input": 0.075, "output": 0.30},
    # Pro tier — synthesis-only path in perseverate.
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
}


def _pricing_for(model: str) -> dict[str, float]:
    """Return {input, output} USD/1M tokens for the given model id.

    Falls back to all-zero if the model isn't in the table — the cost
    field on telemetry shows up as $0.00, which is loud enough to flag
    the missing entry without crashing telemetry emit.
    """
    if not model:
        return {"input": 0.0, "output": 0.0}
    for key, prices in _MODEL_PRICING_USD_PER_1M.items():
        if key in model:
            return prices
    return {"input": 0.0, "output": 0.0}


def _loop_bound_semaphore(
    sem: asyncio.Semaphore | None,
    sem_loop: asyncio.AbstractEventLoop | None,
    max_concurrency: int,
) -> tuple[asyncio.Semaphore, asyncio.AbstractEventLoop]:
    """Return a semaphore bound to the current event loop.

    Activation calls run sync code through ``asyncio.run()``. Reusing the
    same judge object across those calls can otherwise reuse a semaphore
    created by a previous, now-closed loop.
    """
    loop = asyncio.get_running_loop()
    if sem is None or sem_loop is not loop:
        return asyncio.Semaphore(max_concurrency), loop
    return sem, sem_loop


class Judge(Protocol):
    async def score(self, query: str, candidate: str) -> float: ...


class StaticJudge:
    def __init__(
        self,
        score_fn: float | Callable[[str, str], float] = 0.5,
    ) -> None:
        self._fn = score_fn

    async def score(self, query: str, candidate: str) -> float:
        if callable(self._fn):
            return float(self._fn(query, candidate))
        return float(self._fn)


class LocalLLMJudge:
    """Judge using a locally-hosted instruction-tuned LLM (default: Gemma 4 E4B-it).

    On CUDA uses bitsandbytes int8 quantization (~8 GB VRAM); on MPS/CPU uses
    bfloat16. Prompt format is identical to LLMJudge so scores are comparable;
    no network calls once the model is loaded.

    The model must be pre-downloaded in the Docker image — VPC-egress-restricted
    Cloud Run environments cannot reach HuggingFace Hub at runtime.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        device: str | None = None,
        pairs_per_request: int = 16,  # GPU batch size; raise only after VRAM profiling
        load_in_8bit: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pairs_per_request = int(pairs_per_request)
        self._load_in_8bit = load_in_8bit
        self._tokenizer = None
        self._model = None
        self._actual_device: str | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        # Counters are bumped from worker threads (score_batch dispatches via
        # asyncio.to_thread). `n += k` is not atomic in CPython, so guard with
        # a lock to keep telemetry/alerting accurate under concurrent scoring.
        self._counter_lock = threading.Lock()
        self._load_time_sec: float = 0.0
        self._total_pairs: int = 0
        self._total_chunks: int = 0
        self._parse_failures: int = 0
        self._budget_exceeded: int = 0

    def _bump(self, attr: str, n: int = 1) -> None:
        lock = getattr(self, "_counter_lock", None)
        if lock is None:
            setattr(self, attr, getattr(self, attr, 0) + n)
            return
        with lock:
            setattr(self, attr, getattr(self, attr, 0) + n)

    @property
    def stats(self) -> dict:
        vram: dict = {}
        if self._actual_device and self._actual_device.startswith("cuda"):
            vram = {
                "vram_allocated_bytes": torch.cuda.memory_allocated(),
                "vram_peak_bytes": torch.cuda.max_memory_allocated(),
            }
        with self._counter_lock:
            counters = {
                "pairs": self._total_pairs,
                "chunks": self._total_chunks,
                "parse_failures": self._parse_failures,
                "budget_exceeded": self._budget_exceeded,
            }
        return {
            **counters,
            "load_time_sec": self._load_time_sec,
            "device": self._actual_device,
            **vram,
        }

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            if self._device is None:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            # AutoProcessor on Gemma 4 eagerly loads Gemma4VideoProcessor which
            # requires torchvision — unnecessary since this judge uses text only.
            # extra_special_tokens={} overrides the model's tokenizer_config.json
            # which incorrectly ships this as a list instead of a dict, causing
            # an AttributeError in transformers >= 4.51 on .keys().
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, extra_special_tokens={}
            )
            # Left-padding so all sequences in a batch share the same right-
            # aligned position for the generation prefix. EOS as pad is standard
            # for decoder-only models that ship without a dedicated pad token.
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            # device_map="auto" is CUDA-only — accelerate's infer_auto_device_map
            # ignores MPS, so using it on an M-series Mac silently loads everything
            # onto CPU. For non-CUDA paths, load normally and call .to(device).
            load_kwargs: dict = {}
            if self._device == "cuda":
                load_kwargs["device_map"] = "auto"
                if self._load_in_8bit:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                else:
                    load_kwargs["dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16

            t0 = time.monotonic()
            model = AutoModelForCausalLM.from_pretrained(
                self._model_name, **load_kwargs
            )
            if self._device != "cuda":
                model = model.to(self._device)
            self._load_time_sec = time.monotonic() - t0
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._actual_device = str(next(model.parameters()).device)

    async def score(self, query: str, candidate: str) -> float:
        return (await self.score_batch([(query, candidate)]))[0]

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await asyncio.to_thread(self._score_batch_sync, pairs)

    def _score_batch_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        self._ensure_loaded()
        scores: list[float] = []
        for start in range(0, len(pairs), self._pairs_per_request):
            chunk = pairs[start : start + self._pairs_per_request]
            scores.extend(self._score_chunk_sync(chunk))
            self._bump("_total_pairs", len(chunk))
        return scores

    # Per-pair prompt template — one call per pair so outputs are unambiguous.
    # Three few-shot examples (entailment, contradiction, unrelated) anchor the
    # model against drift toward "everything looks like entailment" — a sensitivity
    # noted on small decoder-only models like Gemma 4B.
    _PAIR_PROMPT = (
        "Score entailment and contradiction for this pair.\n"
        'Return strict JSON: {{"entail": 0..1, "contradict": 0..1}}\n'
        "- entail: probability A implies B is true\n"
        "- contradict: probability A implies B is false\n"
        "- Score near-zero on both if unrelated\n\n"
        "=== EXAMPLE 1 (entailment) ===\n"
        'A = "She owns three dogs."  B = "She has pets."\n'
        '{{"entail": 0.95, "contradict": 0.0}}\n\n'
        "=== EXAMPLE 2 (contradiction) ===\n"
        'A = "The room is empty."  B = "The room is full of people."\n'
        '{{"entail": 0.0, "contradict": 0.92}}\n\n'
        "=== EXAMPLE 3 (unrelated) ===\n"
        'A = "It is raining."  B = "She likes coffee."\n'
        '{{"entail": 0.02, "contradict": 0.02}}\n\n'
        "=== SCORE THIS ===\n"
        "A = {q}\nB = {c}"
    )
    # Output for a single pair is ~20 tokens; 48 gives headroom for fenced JSON.
    _MAX_NEW_TOKENS = 48

    def _score_chunk_sync(self, chunk: list[tuple[str, str]]) -> list[float]:
        model_ctx = getattr(self._model.config, "max_position_embeddings", 8192)

        # Tokenize each pair independently into a 1-D tensor.
        all_ids: list[torch.Tensor] = []
        for q, c in chunk:
            prompt = self._PAIR_PROMPT.format(q=repr(q), c=repr(c))
            messages = [{"role": "user", "content": prompt}]
            raw = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                # Gemma's chat template already inserts BOS; add_special_tokens=False
                # prevents a double-BOS. Only correct while model is pinned to Gemma.
                add_special_tokens=False,
            )
            # transformers>=4.44 returns BatchEncoding (UserDict); earlier versions return a Tensor.
            # Use hasattr(raw, "keys") rather than isinstance(raw, dict) because BatchEncoding
            # extends collections.UserDict which is NOT a subclass of dict.
            input_ids: torch.Tensor = raw["input_ids"] if hasattr(raw, "keys") else raw
            all_ids.append(input_ids[0])  # [seq_len]

        # Per-pair budget check: pairs whose prompt alone exceeds context are
        # neutralized individually; the rest are batched normally. This prevents
        # one unusually long belief from silencing valid pairs beside it.
        viable: list[int] = []
        for i, ids in enumerate(all_ids):
            if ids.shape[0] + self._MAX_NEW_TOKENS > model_ctx:
                self._bump("_budget_exceeded", 1)
                logger.warning(
                    "LocalLLMJudge: pair %d prompt length %d + %d new tokens "
                    "exceeds model context %d; using neutral score",
                    i,
                    ids.shape[0],
                    self._MAX_NEW_TOKENS,
                    model_ctx,
                )
            else:
                viable.append(i)

        if not viable:
            return [0.0] * len(chunk)

        viable_ids = [all_ids[i] for i in viable]
        max_input_len = max(t.shape[0] for t in viable_ids)

        # Left-pad all sequences to the same length so they form a rectangular
        # batch. attention_mask=0 on padding tells the model to ignore those
        # positions; left-padding keeps the generation prefix right-aligned.
        pad_id = self._tokenizer.pad_token_id
        input_ids_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        for ids in viable_ids:
            pad_len = max_input_len - ids.shape[0]
            input_ids_list.append(torch.cat([ids.new_full((pad_len,), pad_id), ids]))
            mask_list.append(
                torch.cat([ids.new_zeros(pad_len), ids.new_ones(ids.shape[0])])
            )

        input_ids = torch.stack(input_ids_list).to(self._actual_device)  # [N, L]
        attention_mask = torch.stack(mask_list).to(self._actual_device)  # [N, L]

        with self._generate_lock:
            with torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self._MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

        self._bump("_total_chunks", 1)

        # Decode only the newly generated tokens for each sequence. Because all
        # inputs were padded to max_input_len, new tokens start at that offset.
        # Scatter results back into a full-length list; over-budget slots stay 0.0.
        scores = [0.0] * len(chunk)
        for out_idx, chunk_idx in enumerate(viable):
            new_tokens = output_ids[out_idx, max_input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            e, c = self._parse_single(text, pair_idx=chunk_idx)
            scores[chunk_idx] = max(-1.0, min(1.0, e - c))
        return scores

    @staticmethod
    def _is_numeric(value) -> bool:
        # bool is a subclass of int — `isinstance(True, int)` is True — so a model
        # returning {"entail": true} would otherwise parse as 1.0. Exclude bool
        # explicitly to surface clearly-malformed responses as parse failures.
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _parse_single(self, text: str, *, pair_idx: int = 0) -> tuple[float, float]:
        try:
            data = json.loads(self._extract_json(text))
            e_raw = data.get("entail")
            c_raw = data.get("contradict")
            if not self._is_numeric(e_raw) or not self._is_numeric(c_raw):
                self._bump("_parse_failures", 1)
                logger.error(
                    "LocalLLMJudge pair %d schema error: expected numeric "
                    "entail/contradict, got entail=%r contradict=%r | raw: %.200s",
                    pair_idx,
                    e_raw,
                    c_raw,
                    text,
                )
                return 0.0, 0.0
            return max(0.0, min(1.0, float(e_raw))), max(0.0, min(1.0, float(c_raw)))
        except Exception as exc:
            self._bump("_parse_failures", 1)
            logger.error(
                "LocalLLMJudge pair %d parse failed: %s | raw: %.200s",
                pair_idx,
                exc,
                text,
            )
            return 0.0, 0.0

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences and extract the outermost {...} from model output."""
        # Non-greedy on the fence boundary so we stop at the first closing ```
        # rather than spanning across multiple fenced blocks.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            inner = fenced.group(1)
            s, e = inner.find("{"), inner.rfind("}")
            if s != -1 and e != -1 and e > s:
                return inner[s : e + 1]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text


class LLMJudge:
    # Redis key prefix to avoid collisions with other namespaces.
    _REDIS_PREFIX = "llm_judge:"
    # Default 30-day TTL; override via BELIEF_ACTIVATION_JUDGE_CACHE_TTL_SEC.
    _REDIS_TTL_SEC = int(
        os.environ.get("BELIEF_ACTIVATION_JUDGE_CACHE_TTL_SEC", str(86400 * 30))
    )

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        cache_dir: Path | str | None = None,
        pairs_per_request: int = 20,
        max_concurrency: int = 4,
        redis_client=None,
    ) -> None:
        self._model = model
        self._pairs_per_request = int(pairs_per_request)
        self._max_concurrency = int(max_concurrency)
        # Lazy: created on first use inside the running event loop so it is
        # always bound to the correct loop (asyncio.run() creates a fresh one).
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        # Redis client takes priority; disk cache is the local-dev fallback.
        self._redis = redis_client
        default_cache = Path.home() / ".cache" / "dyssonance" / "llm_judge"
        self._cache_dir = Path(cache_dir) if cache_dir else default_cache
        if self._redis is None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._client_lock = threading.Lock()
        # Telemetry counters — inspect on ``.stats``. A judge can be shared
        # across concurrent score_batch calls, so keep all counter snapshots
        # and mutations under one lock.
        self._counter_lock = threading.Lock()
        self._total_requests: int = 0
        self._total_pairs: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._thinking_tokens: int = 0

    def _bump_counter(self, attr: str, amount: int = 1) -> None:
        with self._counter_lock:
            setattr(self, attr, int(getattr(self, attr)) + int(amount or 0))

    def _bump_tokens(self, prompt: int, output: int, thinking: int) -> None:
        with self._counter_lock:
            self._input_tokens += int(prompt or 0)
            self._output_tokens += int(output or 0)
            self._thinking_tokens += int(thinking or 0)

    @property
    def stats(self) -> dict:
        prices = _pricing_for(self._model)
        # thinking is billed at the output rate when present (Gemini 2.5
        # pricing); since we set thinking_budget=0 it's almost always 0,
        # but include it so the cost stays correct if someone enables it.
        with self._counter_lock:
            requests = self._total_requests
            pairs = self._total_pairs
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
            in_tok = self._input_tokens
            out_tok = self._output_tokens
            think_tok = self._thinking_tokens
        cost_usd = (in_tok / 1_000_000) * prices["input"] + (
            (out_tok + think_tok) / 1_000_000
        ) * prices["output"]
        return {
            "requests": requests,
            "pairs": pairs,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "thinking_tokens": think_tok,
            "estimated_cost_usd": round(cost_usd, 6),
            "model": self._model,
        }

    def _ensure_client(self):
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is loop:
            return self._client
        with self._client_lock:
            if self._client is not None and self._client_loop is loop:
                return self._client
            from google import genai

            self._client = genai.Client()
            self._client_loop = loop
            return self._client

    def _cache_key(self, query: str, candidate: str) -> str:
        h = hashlib.sha1()
        h.update(self._model.encode("utf-8"))
        h.update(b"\x00")
        h.update(query.encode("utf-8"))
        h.update(b"\x00")
        h.update(candidate.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> float | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(f"{self._REDIS_PREFIX}{key}")
                if raw is None:
                    return None
                return float(json.loads(raw)["score"])
            except Exception as exc:
                logger.warning("LLMJudge Redis cache get failed: %s", exc)
                return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return float(data["score"])
        except Exception:
            return None

    def _cache_put(
        self,
        key: str,
        query: str,
        candidate: str,
        entail: float,
        contradict: float,
        score: float,
    ) -> None:
        record = {"entail": entail, "contradict": contradict, "score": score}
        if self._redis is not None:
            try:
                self._redis.set(
                    f"{self._REDIS_PREFIX}{key}",
                    json.dumps(record),
                    ex=self._REDIS_TTL_SEC,
                )
            except Exception as exc:
                logger.warning("LLMJudge Redis cache put failed: %s", exc)
            return
        path = self._cache_path(key)
        record["query"] = query
        record["candidate"] = candidate
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, path)

    async def score(self, query: str, candidate: str) -> float:
        return (await self.score_batch([(query, candidate)]))[0]

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float | None] = [None] * len(pairs)
        misses: list[tuple[int, str, str, str]] = []
        for i, (q, c) in enumerate(pairs):
            key = self._cache_key(q, c)
            # Redis GET is synchronous; offload to thread so we don't stall the loop.
            hit = (
                await asyncio.to_thread(self._cache_get, key)
                if self._redis is not None
                else self._cache_get(key)
            )
            if hit is not None:
                scores[i] = hit
                self._bump_counter("_cache_hits")
            else:
                misses.append((i, key, q, c))
                self._bump_counter("_cache_misses")

        if misses:
            batches = [
                misses[s : s + self._pairs_per_request]
                for s in range(0, len(misses), self._pairs_per_request)
            ]
            results = await asyncio.gather(*[self._score_chunk(b) for b in batches])
            for batch, batch_scores in zip(batches, results):
                for (idx, key, q, c), (entail, contradict) in zip(batch, batch_scores):
                    s = max(-1.0, min(1.0, entail - contradict))
                    scores[idx] = s
                    # Redis SET is synchronous; offload to thread (matches read path).
                    if self._redis is not None:
                        asyncio.create_task(
                            asyncio.to_thread(
                                self._cache_put, key, q, c, entail, contradict, s
                            )
                        )
                    else:
                        self._cache_put(key, q, c, entail, contradict, s)

        self._bump_counter("_total_pairs", len(pairs))
        return [0.0 if s is None else float(s) for s in scores]

    async def _score_chunk(
        self, chunk: list[tuple[int, str, str, str]]
    ) -> list[tuple[float, float]]:
        from google.genai import types

        self._sem, self._sem_loop = _loop_bound_semaphore(
            self._sem, self._sem_loop, self._max_concurrency
        )

        client = self._ensure_client()
        bundle = "\n".join(
            f"Pair {i}: A = {q!r}  B = {c!r}" for i, (_, _, q, c) in enumerate(chunk)
        )
        prompt = f"""For each pair (A, B) score entailment and contradiction.
Return strict JSON: {{"scores": [{{"pair": int, "entail": 0..1, "contradict": 0..1}}]}}.
- entail: probability that A implies B is true
- contradict: probability that A implies B is false
- A pair can score near-zero on both if A and B are unrelated
- Consider semantic meaning, not surface phrasing
- A revision that reverses an earlier claim counts as contradiction

=== EXAMPLES ===
Pair 0: A = "She owns three rescue dogs and a tabby cat."  B = "She has pets."
Pair 1: A = "The store closes at 6 pm on Sundays."  B = "The store is open all day Sunday."
Pair 2: A = "He prefers tea over coffee."  B = "Quantum mechanics is non-deterministic."
Pair 3: A = "Revenue grew 12% year-over-year."  B = "Revenue declined last year."
Pair 4: A = "The bridge is closed for repairs until March."  B = "The bridge is partially open to traffic."
{{"scores": [
  {{"pair": 0, "entail": 0.95, "contradict": 0.0}},
  {{"pair": 1, "entail": 0.0, "contradict": 0.92}},
  {{"pair": 2, "entail": 0.0, "contradict": 0.02}},
  {{"pair": 3, "entail": 0.0, "contradict": 0.94}},
  {{"pair": 4, "entail": 0.0, "contradict": 0.88}}
]}}

=== SCORE THESE ===
{bundle}
"""
        async with self._sem:
            resp = await client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    seed=_JUDGE_SEED,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        self._bump_counter("_total_requests")
        # Token accounting for cost telemetry. usage_metadata is on every
        # successful response; missing only on rare client-side failures
        # (we still count the request above, but no tokens).
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            self._bump_tokens(
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
                getattr(usage, "thoughts_token_count", 0) or 0,
            )
        out: list[tuple[float, float]] = [(0.0, 0.0)] * len(chunk)
        try:
            scores = json.loads(resp.text)["scores"]
        except Exception as exc:
            logger.warning("_score_chunk JSON parse failed: %s", exc)
            return out
        for s in scores:
            p = s.get("pair")
            if p is None or not (0 <= p < len(chunk)):
                continue
            e = float(s.get("entail", 0.0) or 0.0)
            c = float(s.get("contradict", 0.0) or 0.0)
            out[p] = (max(0.0, min(1.0, e)), max(0.0, min(1.0, c)))
        return out


# ─────────────────────────────────────────────────────────────────────
# DissonanceJudge — v1 cross-graph dissonance classifier
# ─────────────────────────────────────────────────────────────────────
#
# Used by the bridge-controller during Layer-3 boundary candidate
# classification (companion spec §8.4 / §11.5 — "v1 LLM stand-in for
# cross-graph contradiction agent"). Returns signed scores compatible
# with the existing Judge Protocol; v2 (PSRO-trained contradiction
# agent) is a drop-in replacement once trained on v1's labels.
#
# Mirrors LLMJudge's batching + caching + concurrency exactly; differs
# only in:
#  - cross-graph-specific prompt (frames the question as "two beliefs
#    from different sources" rather than entailment-style scoring)
#  - provider abstraction (gemini default for prod parity, openai for
#    environments without google-genai access)
#  - returns single signed score per pair (not separate entail +
#    contradict — collapsed into the score directly because cross-
#    graph use cases consume the signed value)
#
# Architectural notes:
#  - Persistent disk cache keyed on (model, text_a, text_b) — repeated
#    classifications are free
#  - Score range: [-1.0, +1.0]
#       +1.0 = strongly coherent (boundary pair supports both texts as true)
#        0.0 = unrelated / ambiguous
#       -1.0 = strongly dissonant (texts contradict or imply mutual
#              exclusivity given other context)
#  - Threshold for routing to dissonance entries vs coherence:
#        score < DISSONANCE_THRESHOLD (default -0.2) → dissonance
#        score > COHERENCE_THRESHOLD  (default +0.2) → coherence
#        else → ambiguous (passes through to beam search as-is)


class DissonanceJudge:
    """LLM-based cross-graph dissonance classifier (v1).

    Compatible with the Judge Protocol — v2 PSRO contradiction agent
    will be a drop-in replacement (same async score / score_batch
    signatures, same float [-1, 1] output range).

    Designed for production scale via:
        - Persistent disk cache (SHA1 keyed on model + pair texts)
        - N-pairs-per-request batching (default 20)
        - asyncio.Semaphore for concurrency control (default 4)
        - Provider-abstracted (gemini / openai)
        - JSON-mode response for reliable parsing

    Example:
        judge = DissonanceJudge(model="gemini-flash-latest", provider="gemini")
        scores = await judge.score_batch([
            ("Alice will be on leave June-August.",
             "Project X requires the lead engineer June-September."),
            ("Bob is the tech lead.", "Bob is the project manager."),
        ])
        # scores[0] ≈ -0.85 (dissonance — Alice can't lead during her leave)
        # scores[1] ≈ +0.30 (mild coherence — Bob has both roles)
    """

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        provider: str = "gemini",  # "gemini" | "openai"
        cache_dir: Path | str | None = None,
        pairs_per_request: int = 20,
        max_concurrency: int = 4,
    ) -> None:
        if provider not in ("gemini", "openai"):
            raise ValueError(f"provider must be 'gemini' or 'openai', got {provider!r}")
        self._model = model
        self._provider = provider
        self._pairs_per_request = int(pairs_per_request)
        self._max_concurrency = int(max_concurrency)
        # Lazy-initialized in the running event loop (different from
        # __init__'s loop when wrapped by asyncio.run).
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        default_cache = Path.home() / ".cache" / "dyssonance" / "dissonance_judge"
        self._cache_dir = Path(cache_dir) if cache_dir else default_cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._client_lock = threading.Lock()
        # Cumulative perf counters — inspect via ``.stats``.
        self._total_requests: int = 0
        self._total_pairs: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._parse_failures: int = 0

    @property
    def stats(self) -> dict:
        return {
            "requests": self._total_requests,
            "pairs": self._total_pairs,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "parse_failures": self._parse_failures,
            "provider": self._provider,
            "model": self._model,
        }

    def _ensure_client(self):
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is loop:
            return self._client
        with self._client_lock:
            if self._client is not None and self._client_loop is loop:
                return self._client
            if self._provider == "gemini":
                from google import genai

                self._client = genai.Client()
            elif self._provider == "openai":
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            self._client_loop = loop
            return self._client

    def _cache_key(self, text_a: str, text_b: str) -> str:
        # Order-invariant: classify (a, b) and (b, a) as the same pair —
        # dissonance is symmetric.
        a, b = sorted([text_a, text_b])
        h = hashlib.sha1()
        h.update(b"dissonance:")
        h.update(self._model.encode("utf-8"))
        h.update(b"\x00")
        h.update(a.encode("utf-8"))
        h.update(b"\x00")
        h.update(b.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> float | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return float(data["score"])
        except Exception:
            return None

    def _cache_put(
        self, key: str, text_a: str, text_b: str, score: float, reason: str
    ) -> None:
        """Persist (score, reason) under the SHA-derived cache key.

        Belief texts can contain PII (names, dates, salary numbers, etc.).
        By default we DO NOT write the raw text into the cache record —
        the SHA-derived `key` already pins the entry to the input pair,
        and read paths only consume `score` and `reason`. Set the env var
        `DISSONANCE_JUDGE_DEBUG_CACHE=1` to additionally store text_a /
        text_b for debugging; never enable in production / multi-tenant
        environments.
        """
        path = self._cache_path(key)
        record: dict = {"score": score, "reason": reason}
        if os.environ.get("DISSONANCE_JUDGE_DEBUG_CACHE", "").lower() in (
            "1",
            "true",
        ):
            record["text_a"] = text_a
            record["text_b"] = text_b
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        os.replace(tmp, path)

    async def score(self, query: str, candidate: str) -> float:
        """Single-pair convenience wrapper. The Protocol's `query` and
        `candidate` parameter names come from the activation-engine
        entailment use case; for cross-graph dissonance, both arguments
        are just two belief texts and the classification is symmetric."""
        return (await self.score_batch([(query, candidate)]))[0]

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a batch of (text_a, text_b) pairs.

        Returns a list of floats aligned with the input order. Each
        score is in [-1, 1]:
            +1.0 = strongly coherent
             0.0 = unrelated / ambiguous
            -1.0 = strongly dissonant

        Cache hits are returned without LLM calls. Misses are batched
        (pairs_per_request per LLM call) and dispatched concurrently
        via asyncio.Semaphore (max_concurrency).
        """
        scores: list[float | None] = [None] * len(pairs)
        misses: list[tuple[int, str, str, str]] = []
        for i, (a, b) in enumerate(pairs):
            key = self._cache_key(a, b)
            hit = self._cache_get(key)
            if hit is not None:
                scores[i] = hit
                self._cache_hits += 1
            else:
                misses.append((i, key, a, b))
                self._cache_misses += 1

        if misses:
            batches = [
                misses[s : s + self._pairs_per_request]
                for s in range(0, len(misses), self._pairs_per_request)
            ]
            results = await asyncio.gather(*[self._score_chunk(b) for b in batches])
            for batch, batch_scores in zip(batches, results):
                for (idx, key, a, b), (score, reason) in zip(batch, batch_scores):
                    scores[idx] = score
                    self._cache_put(key, a, b, score, reason)

        self._total_pairs += len(pairs)
        return [0.0 if s is None else float(s) for s in scores]

    async def score_batch_with_reasons(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[float, str]]:
        """Variant that returns (score, reason) tuples — useful for v2
        training-data generation (the reason gives the PSRO contradiction
        agent additional supervision signal beyond the scalar)."""
        # Run score_batch first (populates cache), then re-read cache
        # for reason text.
        await self.score_batch(pairs)
        out = []
        for a, b in pairs:
            key = self._cache_key(a, b)
            path = self._cache_path(key)
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    out.append((float(data["score"]), str(data.get("reason", ""))))
                    continue
                except Exception:
                    pass
            out.append((0.0, ""))
        return out

    async def _score_chunk(
        self, chunk: list[tuple[int, str, str, str]]
    ) -> list[tuple[float, str]]:
        """LLM call for a single chunk of (idx, key, text_a, text_b) pairs.
        Returns list of (score, reason) tuples aligned with chunk order."""
        self._sem, self._sem_loop = _loop_bound_semaphore(
            self._sem, self._sem_loop, self._max_concurrency
        )

        client = self._ensure_client()
        bundle = "\n".join(
            f"Pair {i}:\n  Belief A: {a}\n  Belief B: {b}"
            for i, (_, _, a, b) in enumerate(chunk)
        )
        prompt = (
            "You are classifying pairs of beliefs from DIFFERENT sources to "
            "decide whether they are coherent (consistent), dissonant "
            "(contradictory or mutually exclusive), or unrelated.\n\n"
            "For each pair below, return a score in [-1.0, 1.0]:\n"
            "  +1.0 = strongly coherent (one supports the other; they imply "
            "compatible facts)\n"
            "   0.0 = unrelated or ambiguous (don't constrain each other)\n"
            "  -1.0 = strongly dissonant (one denies the other; they imply "
            "mutually exclusive facts; satisfying one violates the other)\n\n"
            "Examples:\n"
            '  A: "Alice is on leave June-August."\n'
            '  B: "Project X requires Alice\'s full involvement June-September."\n'
            "  → score: -0.85 (dissonant — Alice can't be both)\n\n"
            '  A: "Bob is the tech lead."\n'
            '  B: "Bob has 5 years of experience."\n'
            "  → score: +0.20 (coherent but only mildly — both can be true)\n\n"
            '  A: "Alice prefers Python."\n'
            '  B: "The launch is on July 15."\n'
            "  → score: 0.0 (unrelated)\n\n"
            "Return strict JSON:\n"
            '  {"scores": [{"pair": int, "score": float, '
            '"reason": str}]}\n'
            "where reason is at most 12 words explaining the classification.\n\n"
            f"{bundle}\n"
        )

        # Concurrency limit is enforced inside _dispatch_llm → _dispatch_with_retry
        # so the semaphore slot is released during rate-limit backoff sleeps.
        # Wrapping this call in an outer `async with self._sem` would hold the
        # slot across multi-second sleeps and starve parallel batches.
        text = await self._dispatch_llm(client, prompt)

        self._total_requests += 1

        # Parse + align
        out: list[tuple[float, str]] = [(0.0, "")] * len(chunk)
        try:
            scores = json.loads(text)["scores"]
        except Exception as exc:
            logger.warning("DissonanceJudge JSON parse failed: %s", exc)
            self._parse_failures += 1
            return out

        for s in scores:
            p = s.get("pair")
            if p is None or not (0 <= p < len(chunk)):
                continue
            score = float(s.get("score", 0.0) or 0.0)
            score = max(-1.0, min(1.0, score))
            reason = str(s.get("reason", "") or "")[:200]  # truncate
            out[p] = (score, reason)
        return out

    async def _dispatch_llm(self, client, prompt: str) -> str:
        """Provider-specific LLM dispatch. Returns the raw response text
        (expected JSON). Both Gemini and OpenAI return JSON via their
        structured output modes. Wrapped in exponential-backoff retry for
        transient rate-limit errors (Vertex 429 / OpenAI RateLimitError).

        The judge concurrency semaphore (`self._sem`) is passed *into* the
        retry wrapper so it's acquired per-attempt rather than held across
        backoff sleeps. Holding the slot during multi-second sleeps would
        starve parallel batches and push every in-flight bridge past its
        timeout. Callers must NOT wrap this method in an outer
        `async with self._sem` block (the retry layer owns the slot)."""
        if self._provider == "gemini":
            from google.genai import types

            # temp=0 + fixed seed → deterministic output. Even with temp=0
            # Gemini can drift between calls without a pinned seed because
            # of upstream batching / token-tie-breaking. The seed locks
            # that down so identical (prompt, model, seed) → identical output.
            async def _call() -> str:
                resp = await client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        seed=_JUDGE_SEED,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return resp.text

            return await _dispatch_with_retry(_call, semaphore=self._sem)
        elif self._provider == "openai":

            async def _call() -> str:
                resp = await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    seed=_JUDGE_SEED,
                    max_tokens=2000,
                )
                return resp.choices[0].message.content

            return await _dispatch_with_retry(_call, semaphore=self._sem)
        else:
            raise ValueError(f"unknown provider: {self._provider!r}")
