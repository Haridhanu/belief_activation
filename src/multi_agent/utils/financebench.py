"""FinanceBench loader and belief-batch builder.

Pulls the open eval set (``PatronusAI/financebench``, ~150 questions) and,
for one question at a time, sentence-splits its evidence pages into
candidate beliefs, embeds them with a local sentence-transformer, and
chunks the result into ``Batch`` objects ready for ``Trainer.step``.

Embeddings are cached on disk so repeated runs of the same question are
instant.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from multi_agent.benchmarks import Batch
from multi_agent.utils.notebook import _DEFAULT_CACHE, _encode_cached


@dataclass
class FinanceBenchQuestion:
    qid: str
    company: str
    doc_name: str
    doc_period: int
    doc_type: str
    question: str
    answer: str
    justification: str
    evidence_pages: list[str]

    @property
    def short_id(self) -> str:
        return self.qid.rsplit("_", 1)[-1]

    @property
    def label(self) -> str:
        return f"{self.company} {self.doc_period} {self.doc_type}"


def load_financebench() -> list[FinanceBenchQuestion]:
    """Pull the open FinanceBench eval split (~150 questions)."""
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/financebench", split="train")
    out: list[FinanceBenchQuestion] = []
    for ex in ds:
        out.append(
            FinanceBenchQuestion(
                qid=ex["financebench_id"],
                company=ex["company"],
                doc_name=ex["doc_name"],
                doc_period=ex["doc_period"],
                doc_type=ex["doc_type"],
                question=ex["question"],
                answer=ex["answer"],
                justification=ex.get("justification") or "",
                evidence_pages=[
                    p["evidence_text_full_page"] for p in ex["evidence"] or []
                ],
            )
        )
    return out


def split_into_beliefs(
    text: str, *, min_words: int = 5, min_alpha_ratio: float = 0.5
) -> list[str]:
    """Sentence-split + drop tabular junk.

    SEC filings inline a lot of tables. Filter sentences shorter than
    ``min_words`` words and any whose alphabetic-character ratio dips
    below ``min_alpha_ratio`` (kills lines that are mostly numbers).
    """
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in parts:
        s = s.strip()
        words = s.split()
        if len(words) < min_words:
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in s)
        if alpha / max(len(s), 1) < min_alpha_ratio:
            continue
        out.append(s)
    return out


class BeliefExtractor(Protocol):
    """Turn one page of evidence into a list of atomic claims."""

    def extract(self, page: str, question: FinanceBenchQuestion) -> list[str]: ...


@dataclass
class SentenceExtractor:
    """Naive sentence-split + length/alpha filter. Fast, free, noisy.

    Use ``GeminiExtractor`` for high-quality belief extraction.
    """

    min_words: int = 5
    min_alpha_ratio: float = 0.5

    def extract(self, page: str, question: FinanceBenchQuestion) -> list[str]:
        return split_into_beliefs(
            page,
            min_words=self.min_words,
            min_alpha_ratio=self.min_alpha_ratio,
        )


_GEMINI_PROMPT = """You are extracting atomic, self-contained claims from a passage of an SEC filing.

Each claim must be **independently interpretable**: a reader who has not seen \
the passage should still understand exactly *who*, *what*, *when*, and *which*. \
If a claim still depends on the surrounding passage to be understood, drop it.

Resolve every reference using the passage:
- **Pronouns** ("we", "it", "they", "the Company", "its") → the literal name "{company}".
- **Demonstratives** ("such cases", "these complaints", "this agreement", \
"such occurrences") → name the specific antecedent (e.g. "the rebate-pricing \
class actions", "the December 2022 opioid settlement agreement").
- **Definite noun phrases** ("the court", "the agreement", "the judgment", \
"the matter", "the case") → name the specific entity (e.g. "the U.S. District \
Court for the Northern District of Ohio", "the December 2022 multistate \
settlement", "the August 2022 $651 million judgment", "United States ex rel. \
Behnke v. CVS Caremark Corporation").
- **Time anaphora** ("later", "subsequently", "during the period") → use the \
explicit date if recoverable; otherwise drop the claim.

Also drop:
- Case captions, docket / MDL numbers, and court names as standalone fragments.
- Citation references like "(MDL No. 2804)" or "United States ex rel. ___ v. ___".
- Headers, table-row text (unless naturally a sentence), and questions.
- Generic boilerplate like "Such occurrences could result in damages." — too \
underspecified to be meaningful.

Prefer claims about concrete facts, events, agreements, obligations, risks, \
parties, dates, and quantities.

Examples of good claims (keep):
- "{company} is appealing the August 2022 $651 million judgment in the Ohio \
federal opioid trial."
- "Under the December 2022 multistate settlement, {company} would pay up to \
$4.3 billion in opioid remediation over 10 years beginning in 2023."
- "In September 2022, CVS Pharmacy, Inc. agreed to settle all opioid claims \
brought by the State of West Virginia."

Examples of bad claims (drop):
- "The agreement is contingent upon sufficient participation." (which agreement?)
- "Such occurrences could result in {company} entering into settlements." (which occurrences?)
- "The court ordered injunctive relief in this matter." (which court? which matter?)
- "{company} is defending itself against these claims." (which claims?)

Output one claim per line. No numbering, no bullets, no preamble, no commentary.

COMPANY: {company} ({doc_period} {doc_type})

PASSAGE:
{text}

CLAIMS:
"""


class GeminiExtractor:
    """LLM atomic-claim extractor backed by ``google.genai``.

    Each ``(model, company, doc_period, doc_type, page_text)`` tuple is
    hashed and cached on disk under ``cache_dir`` so re-extraction is
    free. The first call lazily constructs a ``genai.Client()`` exactly
    like ``judge.GeminiJudge`` does — same auth surface (Vertex via env
    or AI Studio API key).
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        cache_dir: Path | str | None = None,
    ):
        self.model = model
        base = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
        self.cache_dir = base / "beliefs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._lock = threading.Lock()

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            from google import genai

            self._client = genai.Client()
            return self._client

    def _key(self, page: str, q: FinanceBenchQuestion) -> str:
        h = hashlib.sha256()
        for piece in (
            self.model,
            _GEMINI_PROMPT,
            q.company,
            str(q.doc_period),
            q.doc_type,
            page,
        ):
            h.update(piece.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:24]

    def _parse(self, raw: str) -> list[str]:
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•]\s+", "", line)
            line = re.sub(r"^\d+[.)]\s+", "", line)
            line = line.strip()
            if len(line.split()) < 4:
                continue
            out.append(line)
        seen, deduped = set(), []
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)
        return deduped

    def extract(self, page: str, question: FinanceBenchQuestion) -> list[str]:
        key = self._key(page, question)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                cache_path.unlink(missing_ok=True)

        client = self._ensure_client()
        prompt = _GEMINI_PROMPT.format(
            company=question.company,
            doc_period=question.doc_period,
            doc_type=question.doc_type,
            text=page,
        )
        resp = client.models.generate_content(model=self.model, contents=prompt)
        beliefs = self._parse(getattr(resp, "text", "") or "")

        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(beliefs, ensure_ascii=False))
        tmp.replace(cache_path)
        return beliefs


def count_beliefs(question: FinanceBenchQuestion) -> int:
    """How many candidate beliefs sentence-split would yield for this question.

    SEC filings are heavily tabular; tables get filtered out, so most
    questions yield only a handful of prose sentences. Use this to
    pre-screen prose-heavy questions worth running.
    """
    return sum(len(split_into_beliefs(p)) for p in question.evidence_pages)


def prose_questions(
    questions: list[FinanceBenchQuestion], *, min_beliefs: int = 10
) -> list[FinanceBenchQuestion]:
    """Filter to prose-heavy questions (>= ``min_beliefs`` candidate sentences),
    sorted by belief count descending."""
    scored = [(count_beliefs(q), q) for q in questions]
    scored = [(n, q) for n, q in scored if n >= min_beliefs]
    scored.sort(key=lambda nq: -nq[0])
    return [q for _, q in scored]


def make_financebench_batches(
    question: FinanceBenchQuestion,
    *,
    n_batches: int = 5,
    seed: int = 0,
    extractor: BeliefExtractor | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path | str | None = None,
) -> list[Batch]:
    """Extract beliefs from a question's evidence, embed (cached), chunk.

    ``extractor`` defaults to the cheap ``SentenceExtractor`` for backwards
    compatibility. Pass a ``GeminiExtractor`` (or any object implementing
    ``BeliefExtractor``) for high-quality atomic claims.
    """
    if extractor is None:
        extractor = SentenceExtractor()
    sentences: list[str] = []
    for page in question.evidence_pages:
        sentences.extend(extractor.extract(page, question))
    seen, deduped = set(), []
    for s in sentences:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    sentences = deduped
    if not sentences:
        return []

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sentences))
    sentences = [sentences[i] for i in perm]

    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    embs = _encode_cached(sentences, model_name, seed, cache)

    ids = [f"{question.short_id}_s{i}" for i in range(len(sentences))]
    chunks = np.array_split(np.arange(len(sentences)), n_batches)
    return [
        Batch(
            ids=[ids[i] for i in chunk],
            embs=embs[chunk],
            texts=[sentences[i] for i in chunk],
        )
        for chunk in chunks
        if len(chunk) > 0
    ]
