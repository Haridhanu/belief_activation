"""Retrieval benchmark adapters + an ABC every benchmark extends.

A ``Benchmark`` exposes everything the runner needs that *varies* across
benchmarks:

- ``name`` — used as the artifact directory name (``runs_out/<name>/``).
- ``load()`` — returns a list of ``Dataset`` (each is one candidate-set +
  query-set to route over). **Implementations are responsible for their
  own fetch + cache** — first call populates ``<benchmark>_data/`` from
  upstream, subsequent calls read straight from disk.
- ``format_query(q)`` — turns a ``Query`` into the text shown to the LLM
  judge at training time and to the answerer at eval time.
- ``answer(...)`` — asks an LLM to pick/generate an answer from a
  retrieval context.
- ``score(prediction, target)`` — numeric grade of a prediction in [0, 1].
- ``belief_prompt(numbered_sentences)`` — *optional*. Override to enable
  LLM belief extraction over raw context. Default returns ``None`` →
  benchmark uses raw text candidates.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class Query:
    """A single retrieval question with a known correct candidate."""

    description: str
    correct_idx: int
    metadata: dict[str, Any] | None = None


@dataclass
class Dataset:
    """One candidate-pool + its query set, with pre-computed embeddings."""

    id: str
    label: str
    candidates: list[str]
    queries: list[Query]
    cand_embs: np.ndarray  # (n_candidates, emb_dim)
    query_embs: np.ndarray  # (n_queries, emb_dim)
    candidate_meta: list[dict[str, Any]] | None = None


@dataclass
class Batch:
    """A chunk of beliefs to place into the growing graph.

    ``ids`` are globally unique inside the dataset (stringified candidate
    indices) so the graph can keep them stable as it grows.
    """

    ids: list[str]
    embs: np.ndarray  # (len(ids), emb_dim)
    texts: list[str]


class Benchmark(ABC):
    """Contract every benchmark extends."""

    name: str

    @abstractmethod
    def load(self) -> list[Dataset]:
        """Return ready-to-use datasets, fetching + caching if needed."""

    @abstractmethod
    def format_query(self, q: Query) -> str: ...

    @abstractmethod
    def score(self, prediction: str, target: str) -> float: ...

    @abstractmethod
    def build_prompt(
        self,
        query: Query,
        context_texts: list[str],
        *,
        context_ids: list[str] | None = None,
        dataset: "Dataset | None" = None,
        edges: list[tuple[str, str, float]] | None = None,
    ) -> str:
        """Construct the full user-message prompt sent to the answerer LLM.

        ``context_ids`` / ``dataset`` let benchmarks with per-candidate
        metadata re-order or annotate the context. ``edges`` carries
        signed edges among the retrieved nodes (positive = coherent,
        negative = dissonant) so benchmarks can render contradictions as
        competing claims.
        """

    @abstractmethod
    def answer(
        self,
        query: Query,
        context_texts: list[str],
        client,
        model: str = "gpt-4o-mini",
        *,
        context_ids: list[str] | None = None,
        dataset: "Dataset | None" = None,
        edges: list[tuple[str, str, float]] | None = None,
    ) -> str: ...

    @abstractmethod
    def get_batches(self, ds: Dataset, batch_size: int) -> Iterator[Batch]: ...

    def belief_prompt(self, numbered_sentences: str) -> str | None:
        """Optional LLM belief-extraction prompt.

        Override to return a strict-JSON prompt that converts raw context
        sentences into atomic beliefs. Default ``None`` → benchmark uses
        raw text candidates (no extraction step).
        """
        return None

    def coherence_edges(
        self, ds: Dataset, node_ids: list[str]
    ) -> list[tuple[str, str, float]]:
        """Benchmark-defined structural edges to seed the graph with.

        Override for datasets with non-semantic coherence (temporal order,
        shared-entity overlap, etc.). Defaults to empty.
        """
        return []

    def query_node_id(self, q: Query, ds: Dataset) -> str | None:
        """Optional: pin the eval query to an existing graph node.

        Return the node id whose graph-aware representation should be
        used as ``query_emb`` for retrieval (and rendered as the seed
        in the proposal viz). ``None`` falls back to the encoded query
        description. For contradiction benchmarks this is the anchor
        sentence id, so retrieval and the visualization operate on the
        same vector.
        """
        return None


from multi_agent.benchmarks.contradoc import ContraDoc  # noqa: E402

registry: dict[str, type[Benchmark]] = {
    "contradoc": ContraDoc,
}


__all__ = [
    "Batch",
    "Benchmark",
    "Dataset",
    "Query",
    "ContraDoc",
    "registry",
]
