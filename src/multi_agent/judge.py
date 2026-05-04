from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable, Protocol
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
from google.genai import types


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


class NLIJudge:
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
        confidence_floor: float = 0.0,
        log_path: Path | str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._confidence_floor = float(confidence_floor)
        self._log_path = Path(log_path) if log_path else None
        self._log_lock = threading.Lock() if log_path else None
        self._tokenizer = None
        self._model = None
        self._entail_idx: int | None = None
        self._contradict_idx: int | None = None
        self._load_lock = threading.Lock()
        # Cumulative perf counters — inspect on ``.stats``.
        self._total_seconds: float = 0.0
        self._total_forward_passes: int = 0
        self._total_nli_pairs: int = 0
        self._total_judge_calls: int = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            if self._device is None:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name, torch_dtype=dtype
            )
            model.eval()
            model.to(self._device)

            # Resolve label indices from the config so we're robust to
            # ordering differences across MNLI checkpoints.
            id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
            entail_idx = next(i for i, lbl in id2label.items() if "entail" in lbl)
            contradict_idx = next(
                i for i, lbl in id2label.items() if "contradict" in lbl
            )

            self._tokenizer = tokenizer
            self._model = model
            self._entail_idx = entail_idx
            self._contradict_idx = contradict_idx

    @staticmethod
    def _split_neighborhood(candidate: str) -> list[str]:
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        sents = [ln[2:].strip() if ln.startswith("- ") else ln for ln in lines]
        return [s for s in sents if s]

    @property
    def stats(self) -> dict:
        return {
            "judge_calls": self._total_judge_calls,
            "nli_pairs": self._total_nli_pairs,
            "forward_passes": self._total_forward_passes,
            "seconds": self._total_seconds,
        }

    async def score(self, query: str, candidate: str) -> float:
        return (await self.score_batch([(query, candidate)]))[0]

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:

        return await asyncio.to_thread(self._score_batch_sync, pairs)

    def _score_batch_sync(self, pairs: list[tuple[str, str]]) -> list[float]:

        self._ensure_loaded()

        assert self._tokenizer is not None and self._model is not None
        assert self._entail_idx is not None and self._contradict_idx is not None

        premises: list[str] = []
        hypotheses: list[str] = []
        offsets: list[tuple[int, int]] = []
        for q, c in pairs:
            sents = self._split_neighborhood(c)
            start = len(premises)
            premises.extend(sents)
            hypotheses.extend([q] * len(sents))
            offsets.append((start, len(premises)))

        scores = [0.0] * len(pairs)
        if not premises:
            self._total_judge_calls += len(pairs)
            if self._log_path is not None:
                for (q, c), s in zip(pairs, scores):
                    self._append_log(q, c, s)
            return scores

        t0 = time.perf_counter()
        n = len(premises)
        entail = torch.empty(n, dtype=torch.float32)
        contradict = torch.empty(n, dtype=torch.float32)
        passes = 0
        for start in range(0, n, self._batch_size):
            end = min(start + self._batch_size, n)
            enc = self._tokenizer(
                premises[start:end],
                hypotheses[start:end],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            entail[start:end] = probs[:, self._entail_idx]
            contradict[start:end] = probs[:, self._contradict_idx]
            passes += 1

        floor = self._confidence_floor
        entail_gated = entail.where(entail >= floor, torch.zeros_like(entail))
        contradict_gated = contradict.where(
            contradict >= floor, torch.zeros_like(contradict)
        )

        for i, (s, e) in enumerate(offsets):
            if s == e:
                continue
            me = float(entail_gated[s:e].max().item())
            mc = float(contradict_gated[s:e].max().item())
            sc = max(-1.0, min(1.0, me - mc))
            scores[i] = sc

        self._total_seconds += time.perf_counter() - t0
        self._total_forward_passes += passes
        self._total_nli_pairs += n
        self._total_judge_calls += len(pairs)
        if self._log_path is not None:
            for (q, c), s in zip(pairs, scores):
                self._append_log(q, c, s)
        return scores

    def _append_log(self, query: str, candidate: str, score: float) -> None:
        record = {"query": query, "candidate": candidate, "score": score}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        assert self._log_lock is not None and self._log_path is not None
        with self._log_lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(line)


class LLMJudge:
    def __init__(
        self,
        model: str = "gemini-flash-latest",
        cache_dir: Path | str | None = None,
        pairs_per_request: int = 20,
        max_concurrency: int = 4,
    ) -> None:
        self._model = model
        self._pairs_per_request = int(pairs_per_request)
        self._sem = asyncio.Semaphore(max_concurrency)
        default_cache = Path.home() / ".cache" / "dyssonance" / "llm_judge"
        self._cache_dir = Path(cache_dir) if cache_dir else default_cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._client_lock = threading.Lock()
        # Cumulative perf counters — inspect on ``.stats``.
        self._total_requests: int = 0
        self._total_pairs: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    @property
    def stats(self) -> dict:
        return {
            "requests": self._total_requests,
            "pairs": self._total_pairs,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            from google import genai

            self._client = genai.Client()
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
        path = self._cache_path(key)
        record = {
            "query": query,
            "candidate": candidate,
            "entail": entail,
            "contradict": contradict,
            "score": score,
        }
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
            hit = self._cache_get(key)
            if hit is not None:
                scores[i] = hit
                self._cache_hits += 1
            else:
                misses.append((i, key, q, c))
                self._cache_misses += 1

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
                    self._cache_put(key, q, c, entail, contradict, s)

        self._total_pairs += len(pairs)
        return [0.0 if s is None else float(s) for s in scores]

    async def _score_chunk(
        self, chunk: list[tuple[int, str, str, str]]
    ) -> list[tuple[float, float]]:

        client = self._ensure_client()
        bundle = "\n".join(
            f"Pair {i}: A = {q!r}  B = {c!r}" for i, (_, _, q, c) in enumerate(chunk)
        )
        prompt = f"""For each pair below, score whether A and B entail, contradict, or are
neutral. Return strict JSON:
{{"scores": [{{"pair": int, "entail": 0..1, "contradict": 0..1}}]}}.
- Consider semantic contradiction even if phrasing differs.
- A revision/update that reverses an earlier claim counts as contradiction.

{bundle}
"""
        async with self._sem:
            resp = await client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        self._total_requests += 1
        out: list[tuple[float, float]] = [(0.0, 0.0)] * len(chunk)
        try:
            scores = json.loads(resp.text)["scores"]
        except Exception:
            return out
        for s in scores:
            p = s.get("pair")
            if p is None or not (0 <= p < len(chunk)):
                continue
            e = float(s.get("entail", 0.0) or 0.0)
            c = float(s.get("contradict", 0.0) or 0.0)
            out[p] = (max(0.0, min(1.0, e)), max(0.0, min(1.0, c)))
        return out
