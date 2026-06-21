"""ContraDoc benchmark — self-contradictions in long documents.

Each positive example is a document with an inserted or replaced claim
(``evidence``) that contradicts one or more original sentences
(``ref sentences``). The retrieval task: given the ``evidence`` as the
anchor, retrieve the original ``ref sentences``. Correct placement wires
the two with a *dissonant* edge (negative weight), which the answerer
uses to pick the contradictory claim.

Dataset source: https://github.com/ddhruvkr/CONTRADOC — single JSON file
with ``pos`` (449 contradictory docs) and ``neg`` (442 clean docs) keyed
by doc id. We only use ``pos`` rows that include ``ref sentences``.
"""

from __future__ import annotations

import glob
import json
import re
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

from multi_agent.benchmarks import Batch, Benchmark, Dataset, Query

ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
SOURCE_URL = "https://raw.githubusercontent.com/ddhruvkr/CONTRADOC/main/ContraDoc.json"


class ContraDoc(Benchmark):
    """ContraDoc adapter with lazy fetch + sentence-level indexing."""

    name = "contradoc"

    def __init__(
        self,
        data_dir: Path | None = None,
        max_documents: int | None = 40,
        min_sentences: int = 20,
    ) -> None:
        self.data_dir = (
            Path(data_dir) if data_dir else Path(__file__).parent / "contradoc_data"
        )
        self.max_documents = max_documents
        self.min_sentences = min_sentences

    # ------------------------------------------------------------------
    # Benchmark contract
    # ------------------------------------------------------------------

    def load(self, pattern: str = "contradoc_*.json") -> list[Dataset]:
        if not list(glob.glob(str(self.data_dir / pattern))):
            self._prepare()

        datasets: list[Dataset] = []
        for path in sorted(glob.glob(str(self.data_dir / pattern))):
            with open(path) as f:
                payload = json.load(f)
            queries = [
                Query(
                    description=q["description"],
                    correct_idx=int(q["correct_idx"]),
                    metadata={
                        "anchor_idx": q.get("anchor_idx"),
                        "anchor_text": q.get("anchor_text"),
                        "contra_type": q.get("contra_type"),
                        "scope": q.get("scope"),
                    },
                )
                for q in payload["queries"]
            ]
            datasets.append(
                Dataset(
                    id=payload["id"],
                    label=payload["label"],
                    candidates=payload["candidates"],
                    queries=queries,
                    cand_embs=np.asarray(payload["cand_embs"], dtype=np.float32),
                    query_embs=np.asarray(payload["query_embs"], dtype=np.float32),
                )
            )
        return datasets

    def format_query(self, q: Query) -> str:
        return q.description

    def score(self, prediction: str, target: str) -> float:
        return float(prediction.strip().lower() == target.strip().lower())

    def build_prompt(
        self,
        query: Query,
        context_texts: list[str],
        *,
        context_ids: list[str] | None = None,
        dataset: Dataset | None = None,
        edges: list[tuple[str, str, float]] | None = None,
    ) -> str:
        anchor_id, candidates = self._filter_candidates(
            query, context_texts, context_ids
        )
        candidates = self._rerank_by_proximity(anchor_id, candidates, edges)
        anchor = (query.metadata or {}).get("anchor_text", "")
        # Dissonance prior from the graph — the judge has already scored
        # every retrieved candidate against the anchor. Expose the
        # magnitude inline so the small answerer has a structural hint
        # to cross-check against its own reading.
        diss_by_cid = self._dissonance_by_cid(anchor_id, edges)
        lines = [
            "You are checking a long document for internal contradictions.",
            "A contradicting sentence DIRECTLY STATES THE OPPOSITE of the anchor "
            "(e.g. anchor says X happened, contradictor says X did not happen, "
            "or asserts the opposing outcome). Unrelated events or distant plot "
            "points are NOT contradictions.",
            "",
            f'ANCHOR CLAIM: "{anchor}"',
            "",
            "Numbered candidate sentences (ordered by proximity to the anchor). "
            "``dissonance`` is a judge-based prior: more negative = stronger "
            "contradiction signal. Use it as guidance, but verify by reading.",
        ]
        for n, (cid, text) in enumerate(candidates, start=1):
            diss = diss_by_cid.get(cid)
            tag = f" [dissonance={diss:+.2f}]" if diss is not None else ""
            lines.append(f"  [{n}]{tag} {text}")

        lines.append("")
        lines.append(
            "Which numbered candidate directly contradicts the ANCHOR CLAIM? "
            'Reply with only the bracketed number, e.g. "[3]".'
        )
        return "\n".join(lines)

    def answer(
        self,
        query: Query,
        context_texts: list[str],
        client,
        model: str = "gpt-4o-mini",
        *,
        context_ids: list[str] | None = None,
        dataset: Dataset | None = None,
        edges: list[tuple[str, str, float]] | None = None,
    ) -> str:
        prompt = self.build_prompt(
            query,
            context_texts,
            context_ids=context_ids,
            dataset=dataset,
            edges=edges,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        anchor_id, candidates = self._filter_candidates(
            query, context_texts, context_ids
        )
        candidates = self._rerank_by_proximity(anchor_id, candidates, edges)
        m = re.search(r"\[?(\d+)\]?", raw)
        if m:
            n = int(m.group(1))
            if 1 <= n <= len(candidates):
                return candidates[n - 1][1]
        return raw

    def query_node_id(self, q: Query, ds: Dataset) -> str | None:
        anchor_idx = (q.metadata or {}).get("anchor_idx")
        if anchor_idx is None:
            return None
        return str(int(anchor_idx))

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_candidates(
        query: Query,
        context_texts: list[str],
        context_ids: list[str] | None,
    ) -> tuple[str | None, list[tuple[str, str]]]:
        """Drop the anchor from context so the LLM can't echo it back."""
        anchor_idx = (query.metadata or {}).get("anchor_idx")
        anchor_id = str(int(anchor_idx)) if anchor_idx is not None else None
        ids = context_ids or [str(i) for i in range(len(context_texts))]
        out: list[tuple[str, str]] = []
        for cid, text in zip(ids, context_texts):
            if anchor_id is not None and cid == anchor_id:
                continue
            out.append((cid, text))
        return anchor_id, out

    @staticmethod
    def _dissonance_by_cid(
        anchor_id: str | None,
        edges: list[tuple[str, str, float]] | None,
    ) -> dict[str, float]:
        """Strongest dissonant edge weight to the anchor, keyed by candidate id."""
        if not edges or anchor_id is None:
            return {}
        out: dict[str, float] = {}
        for a, b, w in edges:
            if w >= 0:
                continue
            if a == anchor_id:
                other = b
            elif b == anchor_id:
                other = a
            else:
                continue
            if other not in out or abs(w) > abs(out[other]):
                out[other] = float(w)
        return out

    @staticmethod
    def _anchor_conflicts(
        anchor_id: str | None,
        candidates: list[tuple[str, str]],
        edges: list[tuple[str, str, float]] | None,
    ) -> list[tuple[int, float]]:
        """For each candidate, the strongest dissonant edge to the anchor."""
        if not edges or anchor_id is None:
            return []
        touching: dict[str, float] = {}
        for a, b, w in edges:
            if w >= 0:
                continue
            if a == anchor_id:
                other = b
            elif b == anchor_id:
                other = a
            else:
                continue
            if other not in touching or abs(w) > abs(touching[other]):
                touching[other] = float(w)
        ranked: list[tuple[int, float]] = []
        for n, (cid, _text) in enumerate(candidates, start=1):
            if cid in touching:
                ranked.append((n, touching[cid]))
        ranked.sort(key=lambda p: -abs(p[1]))
        return ranked

    @staticmethod
    def _rerank_by_proximity(
        anchor_id: str | None,
        candidates: list[tuple[str, str]],
        edges: list[tuple[str, str, float]] | None,
    ) -> list[tuple[str, str]]:
        """Sort candidates by document-position proximity to the anchor,
        with dissonance magnitude as tie-breaker.

        ContraDoc contradictions are inserted/replaced sentences — the
        altered claim sits in the same narrative slot as the original, so
        the contradictor is almost always the anchor's near-neighbor in
        sentence order. Sorting by ``|Δpos|`` surfaces that candidate at
        ``[1]`` instead of leaving it mid-list in walk order, which is
        what tripped the 8B answerer into picking a distant setup-arc
        sentence.
        """
        if anchor_id is None:
            return candidates
        anchor_pos = int(anchor_id)
        diss: dict[str, float] = {}
        for a, b, w in edges or []:
            if w >= 0:
                continue
            if a == anchor_id:
                other = b
            elif b == anchor_id:
                other = a
            else:
                continue
            diss[other] = max(diss.get(other, 0.0), abs(float(w)))

        def key(pair: tuple[str, str]) -> tuple[int, float]:
            cid, _ = pair
            return (abs(int(cid) - anchor_pos), -diss.get(cid, 0.0))

        return sorted(candidates, key=key)

    def get_batches(self, ds: Dataset, batch_size: int) -> Iterator[Batch]:
        ids = [str(i) for i in range(len(ds.candidates))]
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            sl = slice(start, end)
            yield Batch(
                ids=ids[sl],
                embs=ds.cand_embs[sl],
                texts=ds.candidates[sl],
            )

    # ------------------------------------------------------------------
    # Lazy fetch + prepare
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        print(
            f"[ContraDoc] no cached data in {self.data_dir} — preparing "
            f"(max_documents={self.max_documents})"
        )
        raw = self._fetch_raw()
        rows = self._select_rows(raw)
        if not rows:
            raise RuntimeError("ContraDoc: no eligible documents found.")
        encoder = _load_encoder()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for doc_id, r in rows:
            sentences = _split_sentences(r["text"])
            if len(sentences) < self.min_sentences:
                continue
            evidence_idx = _locate(sentences, r["evidence"])
            ref_sentences = r.get("ref sentences") or []
            ref_indices = [_locate(sentences, s) for s in ref_sentences]
            ref_indices = [i for i in ref_indices if i is not None]
            if evidence_idx is None or not ref_indices:
                continue
            # ``_locate`` falls back to fuzzy matching when the evidence
            # sentence replaced (not inserted-alongside) the ref sentence,
            # and can collide on the same index — the anchor text and the
            # "contradicting" evidence become the same string, leaving
            # nothing for the pipeline to retrieve. Drop those queries.
            ref_indices = [i for i in ref_indices if i != evidence_idx]
            if not ref_indices:
                continue
            # One query per ref sentence: "what contradicts this?" → evidence.
            queries: list[dict] = []
            for ri in ref_indices:
                queries.append(
                    {
                        "description": (
                            f"Which sentence in the document contradicts this "
                            f'claim: "{sentences[ri]}"?'
                        ),
                        "correct_idx": evidence_idx,
                        "anchor_idx": ri,
                        "anchor_text": sentences[ri],
                        "contra_type": r.get("contra_type"),
                        "scope": r.get("scope"),
                    }
                )
            cand_embs = _embed(sentences, encoder).astype(np.float32)
            query_embs = _embed([q["description"] for q in queries], encoder).astype(
                np.float32
            )
            out = {
                "id": doc_id,
                "label": f"ContraDoc {doc_id} ({len(sentences)} sents)",
                "candidates": sentences,
                "queries": queries,
                "cand_embs": cand_embs.tolist(),
                "query_embs": query_embs.tolist(),
            }
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(doc_id))
            with open(self.data_dir / f"contradoc_{safe_id}.json", "w") as f:
                json.dump(out, f)
            print(
                f"  cached {doc_id}: {len(sentences)} sents, "
                f"evidence=[{evidence_idx}], refs={ref_indices}"
            )
            kept += 1
            if self.max_documents is not None and kept >= self.max_documents:
                break

    def _fetch_raw(self) -> dict:
        cache = self.data_dir / "_raw_ContraDoc.json"
        if cache.exists():
            with open(cache) as f:
                return json.load(f)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ContraDoc] downloading {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        with open(cache, "w") as f:
            json.dump(payload, f)
        return payload

    def _select_rows(self, raw: dict) -> list[tuple[str, dict]]:
        """Keep positive rows with ``ref sentences`` so we have labeled pairs."""
        pos = raw.get("pos") or {}
        out: list[tuple[str, dict]] = []
        for doc_id, row in pos.items():
            if not row.get("text") or not row.get("evidence"):
                continue
            if not row.get("ref sentences"):
                continue
            out.append((doc_id, row))
        return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_SENT_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z0-9\"'])")


def _normalize(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip()).lower()


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for chunk in _SENT_RE.split(text or ""):
        stripped = chunk.strip()
        if stripped:
            out.append(stripped)
    return out


def _locate(sentences: list[str], target: str) -> int | None:
    """Find ``target`` in ``sentences``.

    Prefer exact normalized match; fall back to substring containment if
    the target is long enough to be distinctive (>= 30 chars normalized),
    then to the sentence with the highest word-overlap Jaccard. Returns
    ``None`` when nothing clears a minimum similarity bar.
    """
    nt = _normalize(target)
    if not nt:
        return None
    normed = [_normalize(s) for s in sentences]
    for i, ns in enumerate(normed):
        if ns == nt:
            return i
    if len(nt) >= 30:
        for i, ns in enumerate(normed):
            if nt in ns:
                return i
    t_words = set(nt.split())
    if not t_words:
        return None
    best_i, best_j = None, 0.0
    for i, ns in enumerate(normed):
        s_words = set(ns.split())
        if not s_words:
            continue
        j = len(t_words & s_words) / len(t_words | s_words)
        if j > best_j:
            best_j = j
            best_i = i
    return best_i if best_j >= 0.5 else None


_encoder = None


def _load_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER_MODEL)
    return _encoder


def _embed(texts: list[str], encoder) -> np.ndarray:
    if not texts:
        return np.zeros(
            (0, encoder.get_sentence_embedding_dimension()), dtype=np.float32
        )
    return encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
