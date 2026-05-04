"""Notebook helpers: synthetic data + plots over ``StepStats`` history.

Kept out of the core library to avoid pulling matplotlib/pandas into
runtime paths. Install with ``pip install -e ".[notebook]"``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from multi_agent.benchmarks import Batch

_DEFAULT_CACHE = Path.home() / ".cache" / "multi_agent_notebook"


def _encode_cached(
    sentences: list[str],
    model_name: str,
    seed: int,
    cache_dir: Path,
) -> np.ndarray:
    """Encode ``sentences`` with the named sentence-transformer, caching the
    resulting embeddings on disk keyed by a hash of the inputs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1()
    h.update(model_name.encode())
    h.update(str(seed).encode())
    for s in sentences:
        h.update(s.encode())
        h.update(b"\x00")
    path = cache_dir / f"emb_{h.hexdigest()[:16]}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embs = np.asarray(
        model.encode(sentences, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    np.save(path, embs)
    return embs


_BELIEF_CORES: dict[str, list[str]] = {
    "A+": [
        "Artificial intelligence will eventually surpass human intelligence",
        "AI systems will become smarter than people",
        "Machine learning models will exceed human cognitive abilities",
        "In the future, AI will outperform humans at all tasks",
        "Advanced AI will outthink the smartest humans",
        "Intelligent machines will surpass human reasoning",
        "AI's capabilities will grow beyond any human's",
        "Computer intelligence will rival and exceed ours",
    ],
    "A-": [
        "AI will never truly match human intelligence",
        "Machines cannot replicate human reasoning",
        "Human cognition remains beyond the reach of AI",
        "Computers will not outthink people",
        "AI systems will always lack real understanding",
        "Human intelligence cannot be surpassed by software",
        "Machine learning has fundamental limits that humans do not",
        "Artificial intelligence will not exceed human minds",
    ],
    "B+": [
        "Remote work is more productive than office work",
        "Working from home improves individual output",
        "Distributed teams outperform co-located ones",
        "Flexible remote schedules boost productivity",
        "Home offices yield better work results",
        "Remote employees are more efficient than in-office ones",
        "Telework produces more output per hour",
        "People work harder outside the office",
    ],
    "B-": [
        "Remote work is less productive than office work",
        "Working from home reduces individual output",
        "In-person teams outperform distributed ones",
        "Office presence is essential for productivity",
        "Remote employees accomplish less than on-site ones",
        "Productivity suffers when people work from home",
        "Telework produces less output per hour",
        "People work harder in the office than at home",
    ],
    "C+": [
        "Organic food is healthier than conventional food",
        "Eating organic improves long-term health",
        "Organic produce contains more nutrients",
        "Pesticide-free food leads to better health",
        "Organic diets reduce disease risk",
        "Whole organic foods help prevent illness",
        "Conventional food is less nutritious than organic",
        "Organic eating supports overall wellness",
    ],
    "C-": [
        "Organic food is no healthier than conventional food",
        "Eating organic offers no health benefit",
        "Organic produce has the same nutrients as conventional",
        "Pesticide-free claims do not improve health outcomes",
        "Organic diets do not reduce disease risk",
        "Conventional food is as nutritious as organic",
        "The organic label does not indicate healthiness",
        "There is no measurable benefit to eating organic",
    ],
}


def _lc(s: str) -> str:
    """Lowercase the first letter so the sentence can be embedded in a wrapper."""
    return s[:1].lower() + s[1:] if s else s


_BELIEF_WRAPPERS: list = [
    lambda s: f"{s}.",
    lambda s: f"Research suggests that {_lc(s)}.",
    lambda s: f"Many people believe that {_lc(s)}.",
    lambda s: f"Evidence supports the claim that {_lc(s)}.",
    lambda s: f"Some argue that {_lc(s)}.",
]


def make_sentence_batches(
    n_batches: int = 20,
    n_sentences: int | None = None,
    seed: int = 0,
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path | str | None = None,
) -> list[Batch]:
    """Hand-curated belief sentences in three opposing topic pairs (A±, B±, C±),
    embedded with a local sentence-transformer. 6 topics × 8 cores × 5 wrappers
    = 240 sentences; set ``n_sentences`` to cap.

    Embeddings are cached under ``cache_dir`` (default ``~/.cache/multi_agent_notebook``)
    keyed by a hash of the sentence list + model + seed, so repeat runs are instant."""
    all_sentences: list[str] = []
    topic_of: list[int] = []
    topic_keys = sorted(_BELIEF_CORES.keys())
    for t_idx, topic in enumerate(topic_keys):
        for core in _BELIEF_CORES[topic]:
            for wrapper in _BELIEF_WRAPPERS:
                all_sentences.append(wrapper(core))
                topic_of.append(t_idx)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_sentences))
    sentences = [all_sentences[i] for i in perm]
    topics = [topic_keys[topic_of[i]] for i in perm]
    if n_sentences is not None:
        sentences = sentences[:n_sentences]
        topics = topics[:n_sentences]

    resolved = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    embs = _encode_cached(sentences, model_name, seed, resolved)

    ids = [f"b{i}" for i in range(len(sentences))]
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


def make_synthetic_batches(
    n_nodes: int = 200,
    n_batches: int = 20,
    n_topic_pairs: int = 3,
    emb_dim: int = 64,
    noise: float = 0.25,
    seed: int = 0,
) -> list[Batch]:
    """Clustered belief embeddings with ground-truth coherence / contradiction.

    Generates ``2 * n_topic_pairs`` topics whose prototype vectors come in
    antipodal pairs (topic ``2k`` and ``2k+1`` point in opposite directions).
    Each node is sampled near one topic prototype, so pairs within a topic
    are coherent (cosine ≈ +1), pairs across an opposing topic are
    contradictory (cosine ≈ -1), and pairs across unrelated topics are
    neutral (cosine ≈ 0). Use ``make_cosine_judge`` to reveal this
    structure as the NLI-style signal.
    """
    rng = np.random.default_rng(seed)
    n_topics = 2 * n_topic_pairs
    raw_proto = rng.normal(size=(n_topic_pairs, emb_dim)).astype(np.float32)
    raw_proto /= np.linalg.norm(raw_proto, axis=1, keepdims=True)
    prototypes = np.empty((n_topics, emb_dim), dtype=np.float32)
    for k in range(n_topic_pairs):
        prototypes[2 * k] = raw_proto[k]
        prototypes[2 * k + 1] = -raw_proto[k]
    topic_of = rng.integers(0, n_topics, size=n_nodes)
    jitter = noise * rng.normal(size=(n_nodes, emb_dim)).astype(np.float32)
    embs = prototypes[topic_of] + jitter
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    topic_label = [
        f"{chr(ord('A') + t // 2)}{'+' if t % 2 == 0 else '-'}" for t in topic_of
    ]
    ids = [f"n{i}" for i in range(n_nodes)]
    texts = [f"belief {i} [topic {topic_label[i]}]" for i in range(n_nodes)]
    chunks = np.array_split(np.arange(n_nodes), n_batches)
    return [
        Batch(
            ids=[ids[i] for i in chunk],
            embs=embs[chunk],
            texts=[texts[i] for i in chunk],
        )
        for chunk in chunks
        if len(chunk) > 0
    ]


class CosineJudge:
    """Ground-truth judge for clustered synthetic data: returns the cosine
    similarity between a pair of beliefs by looking up their embeddings."""

    def __init__(self, text_to_emb: dict[str, np.ndarray]) -> None:
        self._emb = text_to_emb

    async def score(self, a: str, b: str) -> float:
        ea = self._emb.get(a)
        eb = self._emb.get(b)
        if ea is None or eb is None:
            return 0.0
        return float(np.clip(float(np.dot(ea, eb)), -1.0, 1.0))


def make_cosine_judge(batches: list[Batch]) -> CosineJudge:
    """Build a ``CosineJudge`` from batches, keyed by text. Texts must be
    unique across all batches (``make_synthetic_batches`` guarantees this)."""
    text_to_emb: dict[str, np.ndarray] = {}
    for b in batches:
        for t, e in zip(b.texts, b.embs):
            text_to_emb[t] = np.asarray(e, dtype=np.float32)
    return CosineJudge(text_to_emb)


def history_to_dataframe(step_history):
    """Flatten ``list[StepStats]`` into a pandas DataFrame.

    Per-agent dicts (``sigma``, ``meta_rewards``, ``per_agent_loss``) are
    kept as object columns so downstream plots can explode them.
    """
    import pandas as pd

    return pd.DataFrame([asdict(s) for s in step_history])


def plot_sigma(step_history, ax=None):
    """Plot each agent's σ deviation from the uniform baseline.

    σ lives in a small band around 1/N unless the meta-mixture moves a
    lot, so plotting raw σ on a [0, 1] axis makes movement invisible.
    This centers on uniform=0, padded to the actual data range.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    sigma_df = pd.DataFrame([s.sigma for s in step_history]).fillna(0.0)
    n = max(len(sigma_df.columns), 1)
    uniform = 1.0 / n
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    for col in sigma_df.columns:
        ax.plot(
            sigma_df.index,
            sigma_df[col].values - uniform,
            label=col,
            linewidth=1.8,
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    pad = max(0.01, 1.15 * (sigma_df.values - uniform).__abs__().max())
    ax.set_ylim(-pad, pad)
    ax.set_title(f"σ − uniform ({uniform:.3f}) over batches")
    ax.set_xlabel("batch")
    ax.set_ylabel("σ − uniform")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_coverage(step_history, ax=None):
    """Fraction of each batch's scorable pairs that landed in each bucket."""
    import matplotlib.pyplot as plt

    scorable = np.array([max(s.scorable, 1) for s in step_history], dtype=float)
    cached = np.array([s.cached for s in step_history]) / scorable
    imputed = np.array([s.imputed for s in step_history]) / scorable
    judged = np.array([s.judged for s in step_history]) / scorable
    skipped = np.array([s.skipped for s in step_history]) / scorable
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    x = range(len(step_history))
    ax.stackplot(
        x,
        [cached, imputed, judged, skipped],
        labels=["cached", "imputed", "judged", "skipped"],
        alpha=0.8,
    )
    ax.set_title("Pair coverage per batch")
    ax.set_xlabel("batch")
    ax.set_ylabel("fraction of scorable")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_field_vs_revealed(step_history, ax=None):
    """Scatter of graph's pre-batch field prediction vs the judge's revealed
    score, colored by batch index. The diagonal is perfect prediction."""
    import matplotlib.pyplot as plt

    xs: list[float] = []
    ys: list[float] = []
    batch_ix: list[int] = []
    for i, s in enumerate(step_history):
        for field_val, revealed in s.field_revealed:
            xs.append(field_val)
            ys.append(revealed)
            batch_ix.append(i)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    if xs:
        sc = ax.scatter(xs, ys, c=batch_ix, cmap="viridis", s=14, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="batch")
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("field prediction (pre-batch)")
    ax.set_ylabel("revealed score (judge)")
    ax.set_title("Field vs revealed")
    return ax


def animate_field_vs_revealed(step_history, interval: int = 500):
    """Interactive field-vs-revealed scatter with a per-batch slider and
    play button. Past batches fade to grey; the current batch is highlighted.
    Returns an ``IPython.display.HTML`` to render inline."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation

    points = [
        (i, f, r) for i, s in enumerate(step_history) for f, r in s.field_revealed
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("field prediction (pre-batch)")
    ax.set_ylabel("revealed score (judge)")

    past = ax.scatter([], [], s=10, alpha=0.22, c="grey", label="past batches")
    current = ax.scatter(
        [],
        [],
        s=34,
        alpha=0.95,
        c="#1f77b4",
        edgecolor="white",
        linewidth=0.6,
        label="current batch",
    )
    title = ax.set_title("")
    ax.legend(loc="lower right", fontsize=8)

    def update(i: int):
        past_pts = [(f, r) for idx, f, r in points if idx < i]
        cur_pts = [(f, r) for idx, f, r in points if idx == i]
        past.set_offsets(np.array(past_pts) if past_pts else np.empty((0, 2)))
        current.set_offsets(np.array(cur_pts) if cur_pts else np.empty((0, 2)))
        title.set_text(
            f"Field vs revealed — batch {i + 1}/{len(step_history)}  "
            f"({len(cur_pts)} judged)"
        )
        return past, current, title

    anim = FuncAnimation(
        fig, update, frames=len(step_history), interval=interval, blit=False
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_belief_graph(graph, node_texts=None, *, ax=None, seed: int = 0, title=None):
    """Spring-layout view of the final belief graph.

    Green edges = coherent (+w), red = contradictory (−w), width by |w|.
    Pass ``node_texts`` to enable hover labels on the nodes (jupyter only).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    for nid in graph.get_nodes():
        G.add_node(nid)
    seen: set[tuple[str, str]] = set()
    for nid in graph.get_nodes():
        for nb, w in graph.get_neighbors(nid):
            key = tuple(sorted([nid, nb]))
            if key in seen:
                continue
            seen.add(key)
            G.add_edge(nid, nb, weight=float(w))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    n = max(len(G), 1)
    pos = nx.spring_layout(G, seed=seed, k=1.5 / np.sqrt(n))
    nx.draw_networkx_nodes(G, pos, node_size=42, node_color="#2c3e50", ax=ax)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]
    if pos_edges:
        widths = [0.6 + 2.4 * abs(G[u][v]["weight"]) for u, v in pos_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=pos_edges, edge_color="#27ae60",
            width=widths, alpha=0.55, ax=ax,
        )
    if neg_edges:
        widths = [0.6 + 2.4 * abs(G[u][v]["weight"]) for u, v in neg_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=neg_edges, edge_color="#c0392b",
            width=widths, alpha=0.7, ax=ax,
        )
    ax.set_axis_off()
    ax.set_title(
        title
        or f"belief graph — {len(G)} nodes · {len(pos_edges)}+ / {len(neg_edges)}− edges"
    )
    return ax


def top_edges(graph, node_texts: dict[str, str], n: int = 5):
    """Return ``(top_coherent, top_contradictory)`` — each is a list of
    ``(text_a, text_b, weight)`` triples ordered by |weight| descending."""
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, float]] = []
    for nid in graph.get_nodes():
        for nb, w in graph.get_neighbors(nid):
            key = tuple(sorted([nid, nb]))
            if key in seen:
                continue
            seen.add(key)
            edges.append((nid, nb, float(w)))

    def fmt(e):
        a, b, w = e
        return (node_texts.get(a, a), node_texts.get(b, b), w)

    coh = sorted([e for e in edges if e[2] > 0], key=lambda e: -e[2])[:n]
    dis = sorted([e for e in edges if e[2] < 0], key=lambda e: e[2])[:n]
    return [fmt(e) for e in coh], [fmt(e) for e in dis]


def plot_meta_rewards(step_history, ax=None):
    """Per-agent surprisal credit over batches."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame([s.meta_rewards for s in step_history]).fillna(0.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    for col in df.columns:
        ax.plot(df.index, df[col].values, label=col, linewidth=1.5)
    ax.set_title("Meta-reward per agent (surprisal credit)")
    ax.set_xlabel("batch")
    ax.set_ylabel("surprisal")
    ax.legend(loc="upper right", fontsize=8)
    return ax
