#!/usr/bin/env python3
"""
Typed GNN Probe — Numeric Precision + Fact vs Belief
=====================================================
Experiment based on GNN-LM (Shi et al., ICLR 2022):
  https://arxiv.org/abs/2110.08743

Instead of retrieving similar text passages, we build a belief graph where
nodes are raw text beliefs and edges connect semantically similar ones.
A Typed GCN enforces asymmetric message-passing:
  - Fact nodes  (document-grounded, distillation_gen=0): send freely, ignore belief neighbors
  - Belief nodes (system-generated, distillation_gen>0): receive from everyone, update freely

The model learns two things jointly:
  1. Numeric precision  — pull beliefs with the same number together, push different apart
  2. Fact vs belief     — classify whether a node is externally grounded or inferred

Numeric magnitudes are extracted from raw text with regex at training time only
(a scaffold, not a stored feature) so the model learns to read precision from text.

Architecture:
    Raw text beliefs  (+  is_fact flag from distillation_gen)
        ↓  frozen LLM mean-pool → initial node features
    TypedGCN  [trainable]  — asymmetric message-passing by node type
        ↓  node embeddings
    Projection  [trainable]
        ↓  prefix token (position 0)
    Frozen LLM  — gradient flows through frozen weights back to prefix
        ↓  contextualized prefix output
    NumericAdapter  [trainable]  — fuses LLM output + sinusoidal(regex_magnitude)
        ↓  precision-aware embedding
    ┌── InfoNCE loss   (numeric contrastive)
    └── BCE loss       (fact vs belief classification via FactHead)

Usage:
    uv run python scripts/experiments/typed_gnn_probe.py
    uv run python scripts/experiments/typed_gnn_probe.py --epochs 200
    uv run python scripts/experiments/typed_gnn_probe.py --visualize
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ─────────────────────────────────────────────────────────────────────

LLM_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM         = 384
GCN_HIDDEN      = 256
NUM_FREQS       = 16
GRAPH_THRESHOLD = 0.35
TEMPERATURE     = 0.07
N_NEGATIVES     = 5
LR              = 3e-4
CLASS_WEIGHT    = 0.5   # weight of BCE loss relative to InfoNCE
DEFAULT_EPOCHS  = 150

# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class Belief:
    text: str
    entity: str
    attribute: str
    is_fact: bool          # True = document-grounded (distillation_gen=0)
    source: str            # "paper" for facts, "reasoning" for beliefs
    distillation_gen: int  # 0 = fact, >0 = inferred


# Synthetic dataset: facts (direct from AIAYN paper) vs beliefs (system-inferred).
# The model sees ONLY the raw text — magnitudes are extracted by regex at train time.
DATASET: list[Belief] = [
    # ── Facts (document-grounded) ───────────────────────────────────────────
    Belief("The model uses d_model = 512 dimensions.",               "model",          "dimensions",    is_fact=True,  source="paper",     distillation_gen=0),
    Belief("The model uses h = 8 parallel attention heads.",         "attention",      "num_heads",     is_fact=True,  source="paper",     distillation_gen=0),
    Belief("The encoder is composed of N = 6 identical layers.",     "encoder",        "num_layers",    is_fact=True,  source="paper",     distillation_gen=0),
    Belief("The decoder is composed of N = 6 identical layers.",     "decoder",        "num_layers",    is_fact=True,  source="paper",     distillation_gen=0),
    Belief("Each attention head has d_k = 64 dimensions.",           "attention_head", "key_dim",       is_fact=True,  source="paper",     distillation_gen=0),
    Belief("Dropout probability P_drop = 0.1 is applied.",          "dropout",        "probability",   is_fact=True,  source="paper",     distillation_gen=0),
    Belief("Label smoothing of epsilon_ls = 0.1 was used.",         "label_smoothing","epsilon",       is_fact=True,  source="paper",     distillation_gen=0),
    Belief("BLEU score of 28.4 on WMT 2014 English-German.",        "translation_ende","bleu",         is_fact=True,  source="paper",     distillation_gen=0),
    Belief("BLEU score of 41.0 on WMT 2014 English-French.",        "translation_enfr","bleu",         is_fact=True,  source="paper",     distillation_gen=0),
    Belief("Shared vocabulary of 37000 BPE tokens.",                 "vocabulary",     "size",          is_fact=True,  source="paper",     distillation_gen=0),
    Belief("The feed-forward sublayer has d_ff = 2048.",             "feed_forward",   "inner_dim",     is_fact=True,  source="paper",     distillation_gen=0),
    Belief("Training used 8 NVIDIA P100 GPUs.",                      "training",       "num_gpus",      is_fact=True,  source="paper",     distillation_gen=0),

    # ── Beliefs (system-generated / inferred) ───────────────────────────────
    Belief("The model likely uses around 500 dimensions for efficiency.",          "model",          "dimensions",  is_fact=False, source="reasoning", distillation_gen=1),
    Belief("With 8 heads, each head probably has about 60 dimensions.",            "attention_head", "key_dim",     is_fact=False, source="reasoning", distillation_gen=1),
    Belief("The architecture seems to use roughly 6 encoder layers.",              "encoder",        "num_layers",  is_fact=False, source="reasoning", distillation_gen=1),
    Belief("Performance of around 28 BLEU on German translation seems plausible.", "translation_ende","bleu",       is_fact=False, source="reasoning", distillation_gen=1),
    Belief("The dropout of 0.1 is standard for this type of architecture.",        "dropout",        "probability", is_fact=False, source="reasoning", distillation_gen=1),
    Belief("Training likely requires a vocabulary of around 40000 tokens.",        "vocabulary",     "size",        is_fact=False, source="reasoning", distillation_gen=1),
    Belief("A feed-forward size of around 2000 would make sense here.",            "feed_forward",   "inner_dim",   is_fact=False, source="reasoning", distillation_gen=1),
    Belief("The big model probably uses higher dropout, perhaps 0.3.",             "dropout_big",    "probability", is_fact=False, source="reasoning", distillation_gen=2),
    Belief("With 512 model dimensions and 8 heads, each head has 64 dimensions.", "attention_head", "key_dim",     is_fact=False, source="reasoning", distillation_gen=1),
    Belief("English-French translation should score above 40 BLEU.",               "translation_enfr","bleu",       is_fact=False, source="reasoning", distillation_gen=2),
]

# Precision eval pairs — (label_a, label_b, idx_a, idx_b, expected)
EVAL_PAIRS = [
    ("fact 512 dims",    "belief ~500 dims",   0,  12, "FAR"),    # precision: 512 ≠ ~500
    ("fact 8 heads",     "belief ~60 key_dim", 1,  13, "FAR"),    # precision: 8 ≠ 60
    ("fact enc N=6",     "belief ~6 layers",   2,  14, "CLOSE"),  # same magnitude
    ("fact BLEU 28.4",   "belief ~28 BLEU",    7,  15, "FAR"),    # precision: 28.4 ≠ 28
    ("fact drop 0.1",    "belief drop 0.1",    5,  16, "CLOSE"),  # same magnitude
    ("fact 37k vocab",   "belief ~40k vocab",  9,  17, "FAR"),    # precision: 37000 ≠ 40000
    ("enc N=6",          "dec N=6",            2,  3,  "CLOSE"),  # same magnitude, diff entity
    ("label_smooth 0.1", "dropout 0.1",        5,  6,  "CLOSE"),  # same magnitude, diff concept
]

# ── Regex magnitude extraction (training scaffold only) ────────────────────────

_NUM_RE = re.compile(r"\b(\d{1,6}(?:\.\d+)?)\b")

def extract_magnitude(text: str) -> float | None:
    """Pull the most prominent number from raw belief text."""
    hits = [float(m) for m in _NUM_RE.findall(text) if float(m) > 0]
    if not hits:
        return None
    # Heuristic: prefer the largest value that isn't an obvious year (>2100)
    filtered = [h for h in hits if h < 200_000]
    return max(filtered) if filtered else max(hits)

# ── Typed GCN ──────────────────────────────────────────────────────────────────

class TypedGCN(nn.Module):
    """
    Asymmetric message-passing conditioned on node type.

    Fact nodes  : send freely; do NOT update from belief-node messages.
    Belief nodes: send and receive from all neighbors.

    Separate weight matrices for fact-source vs belief-source messages give the
    network different "voices" for grounded vs inferred information.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_fact   = nn.Linear(in_dim, hidden_dim, bias=False)  # fact → msg
        self.W_belief = nn.Linear(in_dim, hidden_dim, bias=False)  # belief → msg
        self.W_self   = nn.Linear(in_dim, out_dim, bias=False)     # self-loop
        self.W_out    = nn.Linear(hidden_dim, out_dim, bias=False)
        self.norm     = nn.LayerNorm(out_dim)
        for w in [self.W_fact, self.W_belief, self.W_self, self.W_out]:
            nn.init.xavier_uniform_(w.weight)

    def forward(
        self,
        x: torch.Tensor,        # (N, in_dim)
        adj: torch.Tensor,      # (N, N) row-normalised
        is_fact: torch.Tensor,  # (N,)  bool
    ) -> torch.Tensor:
        fact_f = is_fact.float().unsqueeze(-1)     # (N, 1)

        # Per-node messages: fact nodes use W_fact, belief nodes use W_belief
        msgs = fact_f * self.W_fact(x) + (1 - fact_f) * self.W_belief(x)  # (N, H)

        # Build typed adjacency: block edges where dst=fact AND src=belief
        # i.e. zero out adj[i, j] when is_fact[i] and NOT is_fact[j]
        fact_dst    = is_fact.float().unsqueeze(1)      # (N, 1)
        belief_src  = (~is_fact).float().unsqueeze(0)   # (1, N)
        block_mask  = fact_dst * belief_src             # (N, N) — 1 where blocked
        adj_typed   = adj * (1.0 - block_mask)          # (N, N)

        agg  = adj_typed @ msgs                         # (N, H)
        out  = self.W_out(agg) + self.W_self(x)        # (N, out_dim)
        return self.norm(F.relu(out))

    def edge_weights(
        self,
        adj: torch.Tensor,
        is_fact: torch.Tensor,
    ) -> torch.Tensor:
        """Return the typed adjacency used in forward (for visualisation)."""
        fact_dst   = is_fact.float().unsqueeze(1)
        belief_src = (~is_fact).float().unsqueeze(0)
        block_mask = fact_dst * belief_src
        return adj * (1.0 - block_mask)

# ── Supporting modules ─────────────────────────────────────────────────────────

class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.linear(x)))


class NumericAdapter(nn.Module):
    """Fuses frozen-LLM output with sinusoidal magnitude encoding."""

    def __init__(self, llm_dim: int, num_freqs: int = NUM_FREQS) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        enc_dim = num_freqs * 2
        self.fc1  = nn.Linear(llm_dim + enc_dim, llm_dim)
        self.fc2  = nn.Linear(llm_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def _sinusoidal_encode(self, magnitude: float, device: torch.device) -> torch.Tensor:
        log_m  = math.log(abs(magnitude) + 1e-8)
        freqs  = torch.exp(torch.linspace(-2.0, 4.0, self.num_freqs, device=device))
        args   = log_m * freqs
        return torch.cat([torch.sin(args), torch.cos(args)])

    def forward(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        enc = self._sinusoidal_encode(magnitude, x.device).unsqueeze(0).expand(x.shape[0], -1)
        h   = F.gelu(self.fc1(torch.cat([x, enc], dim=-1)))
        out = self.norm(self.fc2(h))
        return out.squeeze(0) if squeeze else out


class FactHead(nn.Module):
    """Binary classifier: is this node a document-grounded fact?"""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)   # (N,) logits

# ── Graph construction ─────────────────────────────────────────────────────────

def build_adj(x0: torch.Tensor, threshold: float = GRAPH_THRESHOLD) -> torch.Tensor:
    with torch.no_grad():
        sim = F.normalize(x0, dim=-1) @ F.normalize(x0, dim=-1).t()
    A   = (sim >= threshold).float()
    A   = (A + torch.eye(A.shape[0], device=A.device)).clamp(max=1.0)
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / deg

# ── LLM helpers ────────────────────────────────────────────────────────────────

def _word_embeddings(llm: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    emb = getattr(llm, "embeddings", None)
    if emb is not None and hasattr(emb, "word_embeddings"):
        with torch.no_grad():
            return emb.word_embeddings(input_ids)
    with torch.no_grad():
        return llm(input_ids=input_ids).last_hidden_state.detach()


def precompute_text_embs(beliefs, llm, tokenizer, device) -> list[torch.Tensor]:
    cached = []
    for b in beliefs:
        enc      = tokenizer(b.text, return_tensors="pt", truncation=True, max_length=64).to(device)
        word_emb = _word_embeddings(llm, enc["input_ids"])   # (1, seq+2, D)
        cached.append(word_emb.squeeze(0)[1:])               # drop CLS, keep text + SEP
    return cached


def batch_prefix_forward(llm, proj_embs, cached_text_embs, device):
    """Prefix-inject GCN embeddings into frozen LLM. Gradient flows through."""
    N, D   = proj_embs.shape
    max_len = max(t.shape[0] for t in cached_text_embs)

    padding        = torch.zeros(N, max_len, D, device=device)
    attention_mask = torch.zeros(N, 1 + max_len, device=device)
    attention_mask[:, 0] = 1.0

    for i, te in enumerate(cached_text_embs):
        sl = te.shape[0]
        padding[i, :sl]         = te.to(device)
        attention_mask[i, 1:1+sl] = 1.0

    inputs_embeds = torch.cat([proj_embs.unsqueeze(1), padding], dim=1)  # (N, 1+max_len, D)
    out           = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0, :]   # (N, D)

# ── Losses ─────────────────────────────────────────────────────────────────────

def infonce_loss(anchor, positive, negatives, temperature=TEMPERATURE):
    all_t  = torch.cat([positive.unsqueeze(0), negatives], dim=0)
    a_norm = F.normalize(anchor.unsqueeze(0), dim=-1)
    t_norm = F.normalize(all_t, dim=-1)
    logits = (a_norm @ t_norm.t()).squeeze(0) / temperature
    label  = torch.zeros(1, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits.unsqueeze(0), label)

# ── Pair sampling ──────────────────────────────────────────────────────────────

def find_positive(anchor_idx: int, magnitudes: list[float | None]) -> int | None:
    m = magnitudes[anchor_idx]
    if m is None:
        return None
    log_m = math.log(m + 1e-8)
    best_idx, best_diff = None, float("inf")
    for i, m2 in enumerate(magnitudes):
        if i == anchor_idx or m2 is None:
            continue
        d = abs(log_m - math.log(m2 + 1e-8))
        if d < best_diff:
            best_diff, best_idx = d, i
    return best_idx


def sample_negatives(anchor_idx, pos_idx, N, k):
    pool   = [i for i in range(N) if i not in (anchor_idx, pos_idx)]
    chosen = np.random.default_rng().choice(pool, size=min(k, len(pool)), replace=False)
    return chosen.tolist()

# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_all_embeddings(beliefs, gcn, projection, adapter, fact_head, llm,
                       cached_text_embs, adj, x0, is_fact, magnitudes, device):
    h           = gcn(x0, adj, is_fact)
    proj        = projection(h)
    prefix_outs = batch_prefix_forward(llm, proj, cached_text_embs, device)
    embs = torch.stack([
        adapter(prefix_outs[i], magnitudes[i] or 1.0) for i in range(len(beliefs))
    ])
    logits = fact_head(embs)
    return embs, torch.sigmoid(logits)


def report_pairs(beliefs, embs, stage):
    pairs = [p for p in EVAL_PAIRS if p[2] < len(beliefs) and p[3] < len(beliefs)]
    print(f"\n  {'─'*67}")
    print(f"  Precision pair similarities — {stage}")
    print(f"  {'─'*67}")
    print(f"  {'Pair':<38} {'Sim':>6}  {'Expected':>8}  {'Pass':>4}")
    print(f"  {'─'*38} {'─'*6}  {'─'*8}  {'─'*4}")
    results = []
    for la, lb, i, j, expected in pairs:
        sim    = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
        passed = (expected == "CLOSE" and sim > 0.5) or (expected == "FAR" and sim < 0.5)
        results.append(passed)
        print(f"  {la+' ↔ '+lb:<38} {sim:>6.3f}  {expected:>8}  {'✓' if passed else '✗':>4}")
    print(f"\n  Passed: {sum(results)}/{len(results)}")
    return results


def report_classification(beliefs, fact_probs, stage):
    correct = sum(
        (b.is_fact and p > 0.5) or (not b.is_fact and p <= 0.5)
        for b, p in zip(beliefs, fact_probs.tolist())
    )
    print(f"\n  Fact/belief classification ({stage}): {correct}/{len(beliefs)} correct")
    for b, p in zip(beliefs, fact_probs.tolist()):
        pred   = "FACT" if p > 0.5 else "BELIEF"
        truth  = "FACT" if b.is_fact else "BELIEF"
        mark   = "✓" if pred == truth else "✗"
        label  = b.text[:55]
        print(f"  {mark} [{truth:<6}→{pred:<6}  p={p:.2f}]  {label}")

# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize(beliefs, gcn, adj, x0, is_fact, magnitudes,
              pre_embs, post_embs, loss_num, loss_cls, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError:
        print("matplotlib / networkx not installed. Skipping visualisation.")
        return

    try:
        from sklearn.manifold import TSNE
        has_tsne = True
    except ImportError:
        has_tsne = False

    out_dir.mkdir(parents=True, exist_ok=True)
    N      = len(beliefs)
    labels = [f"{b.entity}\n{magnitudes[i] or '?'}" for i, b in enumerate(beliefs)]
    colors = ["steelblue" if b.is_fact else "coral" for b in beliefs]
    log_mags = np.array([math.log(m + 1e-8) if m else 0.0 for m in magnitudes])

    # ── Figure 1: Belief graph + typed message-passing ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Raw adjacency (symmetric, untyped)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, label=labels[i], is_fact=beliefs[i].is_fact)
    raw_adj_np = adj.numpy()
    for i in range(N):
        for j in range(N):
            if i != j and raw_adj_np[i, j] > 0:
                G.add_edge(j, i, weight=float(raw_adj_np[i, j]))

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Left: untyped graph
    ax = axes[0]
    fact_nodes   = [n for n in G if beliefs[n].is_fact]
    belief_nodes = [n for n in G if not beliefs[n].is_fact]
    sizes        = [max(200, 100 * math.log(magnitudes[n] + 2)) if magnitudes[n] is not None else 200 for n in G]

    nx.draw_networkx_nodes(G, pos, nodelist=fact_nodes,   node_color="steelblue", node_size=[sizes[n] for n in fact_nodes],  ax=ax, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=belief_nodes, node_color="coral",     node_size=[sizes[n] for n in belief_nodes], ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax, arrows=True, arrowsize=10, edge_color="gray")
    nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in G}, font_size=5, ax=ax)

    ax.set_title("Belief graph (all edges)", fontsize=11)
    ax.legend(handles=[
        mpatches.Patch(color="steelblue", label="Fact (doc-grounded)"),
        mpatches.Patch(color="coral",     label="Belief (inferred)"),
    ], loc="upper left", fontsize=8)
    ax.axis("off")

    # Right: typed adjacency — show blocked edges as red dashed
    ax2 = axes[1]
    with torch.no_grad():
        typed_adj = gcn.edge_weights(adj, is_fact).numpy()

    active_edges  = [(j, i) for i in range(N) for j in range(N) if i != j and typed_adj[i, j] > 0]
    blocked_edges = [(j, i) for i in range(N) for j in range(N) if i != j and raw_adj_np[i, j] > 0 and typed_adj[i, j] == 0]

    nx.draw_networkx_nodes(G, pos, nodelist=fact_nodes,   node_color="steelblue", node_size=[sizes[n] for n in fact_nodes],  ax=ax2, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=belief_nodes, node_color="coral",     node_size=[sizes[n] for n in belief_nodes], ax=ax2, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=active_edges,  edge_color="steelblue", ax=ax2, alpha=0.5, arrows=True, arrowsize=10)
    nx.draw_networkx_edges(G, pos, edgelist=blocked_edges, edge_color="red",       ax=ax2, alpha=0.6, arrows=True, arrowsize=10, style="dashed")
    nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in G}, font_size=5, ax=ax2)

    ax2.set_title("Typed GCN message-passing\n(blue=active, red dashed=blocked belief→fact)", fontsize=11)
    ax2.legend(handles=[
        mpatches.Patch(color="steelblue", label="Active edge"),
        mpatches.Patch(color="red",       label="Blocked (belief src → fact dst)"),
    ], loc="upper left", fontsize=8)
    ax2.axis("off")

    fig.suptitle("Belief Graph: Typed Message-Passing Mechanism", fontsize=13, fontweight="bold")
    p1 = out_dir / "01_belief_graph.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Saved {p1}")

    # ── Figure 2: t-SNE embeddings (before / after) ─────────────────────────
    if has_tsne and pre_embs is not None and post_embs is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        perp = min(5, N - 1)

        for row, (embs_t, title) in enumerate([(pre_embs, "Before training"), (post_embs, "After training")]):
            embs_np = embs_t.numpy()
            coords  = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(embs_np)

            # Col 0: coloured by fact/belief
            ax = axes[row][0]
            for i, (x, y) in enumerate(coords):
                c   = "steelblue" if beliefs[i].is_fact else "coral"
                ax.scatter(x, y, c=c, s=80, zorder=2)
                ax.annotate(labels[i], (x, y), fontsize=5.5, ha="center", va="bottom")
            ax.set_title(f"{title} — coloured by type", fontsize=10)
            ax.legend(handles=[
                mpatches.Patch(color="steelblue", label="Fact"),
                mpatches.Patch(color="coral",     label="Belief"),
            ], fontsize=8)

            # Col 1: coloured by log(magnitude)
            ax = axes[row][1]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=log_mags, cmap="plasma", s=80, zorder=2)
            for i, (x, y) in enumerate(coords):
                ax.annotate(labels[i], (x, y), fontsize=5.5, ha="center", va="bottom")
            plt.colorbar(sc, ax=ax, label="log(magnitude)")
            ax.set_title(f"{title} — coloured by log(magnitude)", fontsize=10)

        fig.suptitle("Embedding Space: Before vs After Training", fontsize=13, fontweight="bold")
        p2 = out_dir / "02_embeddings_tsne.png"
        plt.tight_layout()
        plt.savefig(p2, dpi=150)
        plt.close()
        print(f"  Saved {p2}")

    # ── Figure 3: Loss curves ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(loss_num,  color="steelblue", label="InfoNCE (numeric)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Numeric Contrastive Loss"); axes[0].legend()
    axes[1].plot(loss_cls,  color="coral",      label="BCE (fact/belief)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Fact vs Belief Classification Loss"); axes[1].legend()
    fig.suptitle("Training Losses", fontsize=13, fontweight="bold")
    p3 = out_dir / "03_losses.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"  Saved {p3}")

    # ── Figure 4: Magnitude similarity heatmap before/after ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, embs_t, title in [(axes[0], pre_embs, "Before"), (axes[1], post_embs, "After")]:
        embs_np = F.normalize(embs_t, dim=-1).numpy()
        sim_mat = embs_np @ embs_np.T
        im = ax.imshow(sim_mat, cmap="RdBu", vmin=-1, vmax=1)
        short = [b.entity[:12] for b in beliefs]
        ax.set_xticks(range(N)); ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticks(range(N)); ax.set_yticklabels(short, fontsize=6)
        ax.set_title(f"Cosine similarity — {title} training", fontsize=10)
        plt.colorbar(im, ax=ax)
    fig.suptitle("Pairwise Embedding Similarity", fontsize=13, fontweight="bold")
    p4 = out_dir / "04_similarity_heatmap.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=150)
    plt.close()
    print(f"  Saved {p4}")

# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(beliefs, epochs, do_visualize):
    print(f"\nTyped GNN Probe")
    print(f"  beliefs : {len(beliefs)}  ({sum(b.is_fact for b in beliefs)} facts, {sum(not b.is_fact for b in beliefs)} beliefs)")
    print(f"  model   : {LLM_NAME}")
    print(f"  epochs  : {epochs}")

    device = torch.device("cpu")

    # Regex magnitude extraction (scaffold — model never sees these directly)
    magnitudes: list[float | None] = [extract_magnitude(b.text) for b in beliefs]
    print("\n  Extracted magnitudes from raw text:")
    for b, m in zip(beliefs, magnitudes):
        tag = "FACT  " if b.is_fact else "BELIEF"
        print(f"    [{tag}] {m!s:>10}  {b.text[:65]}")

    # Load frozen LLM
    print("\nLoading frozen LLM …")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed.")
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm       = AutoModel.from_pretrained(LLM_NAME).to(device)
    for p in llm.parameters():
        p.requires_grad = False

    # Initial node embeddings (frozen LLM mean-pool)
    print("Computing initial node embeddings …")
    x0_list = []
    for b in beliefs:
        enc = tokenizer(b.text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = llm(**enc)
        x0_list.append(out.last_hidden_state.mean(dim=1).squeeze(0))
    x0 = torch.stack(x0_list).detach()  # (N, EMB_DIM)

    cached_text_embs = precompute_text_embs(beliefs, llm, tokenizer, device)
    is_fact          = torch.tensor([b.is_fact for b in beliefs])

    adj = build_adj(x0, GRAPH_THRESHOLD)
    n_edges = int((adj > 0).sum().item()) - len(beliefs)
    print(f"Graph: {len(beliefs)} nodes, {n_edges} edges (threshold={GRAPH_THRESHOLD})")

    # Trainable modules
    gcn        = TypedGCN(EMB_DIM, GCN_HIDDEN, EMB_DIM).to(device)
    projection = Projection(EMB_DIM, EMB_DIM).to(device)
    adapter    = NumericAdapter(EMB_DIM, NUM_FREQS).to(device)
    fact_head  = FactHead(EMB_DIM).to(device)
    optimizer  = torch.optim.Adam(
        list(gcn.parameters()) + list(projection.parameters()) +
        list(adapter.parameters()) + list(fact_head.parameters()),
        lr=LR,
    )

    N = len(beliefs)

    # Baseline
    print("\n=== BEFORE TRAINING ===")
    pre_embs, pre_probs = get_all_embeddings(
        beliefs, gcn, projection, adapter, fact_head, llm,
        cached_text_embs, adj, x0, is_fact, magnitudes, device
    )
    pre_pair_results = report_pairs(beliefs, pre_embs, "before")
    report_classification(beliefs, pre_probs, "before")

    # Training
    print(f"\nTraining …")
    loss_num_hist: list[float] = []
    loss_cls_hist: list[float] = []
    fact_labels = is_fact.float().to(device)

    for epoch in range(epochs):
        h           = gcn(x0, adj, is_fact)
        proj        = projection(h)
        prefix_outs = batch_prefix_forward(llm, proj, cached_text_embs, device)

        all_embs = torch.stack([
            adapter(prefix_outs[i], magnitudes[i] or 1.0) for i in range(N)
        ])  # (N, EMB_DIM)

        # ── Loss 1: numeric InfoNCE ──────────────────────────────────────────
        numeric_loss = torch.zeros(1, device=device)
        n_contrib    = 0
        for i in range(N):
            if magnitudes[i] is None:
                continue
            pos_idx = find_positive(i, magnitudes)
            if pos_idx is None:
                continue
            neg_indices  = sample_negatives(i, pos_idx, N, N_NEGATIVES)
            numeric_loss = numeric_loss + infonce_loss(all_embs[i], all_embs[pos_idx], all_embs[neg_indices])
            n_contrib   += 1
        if n_contrib > 0:
            numeric_loss = numeric_loss / n_contrib

        # ── Loss 2: fact vs belief BCE ───────────────────────────────────────
        logits   = fact_head(all_embs)                 # (N,)
        cls_loss = F.binary_cross_entropy_with_logits(logits, fact_labels)

        loss = numeric_loss + CLASS_WEIGHT * cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_num_hist.append(numeric_loss.item())
        loss_cls_hist.append(cls_loss.item())

        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1:4d}/{epochs}  numeric={numeric_loss.item():.4f}  cls={cls_loss.item():.4f}")

    # After training
    print("\n=== AFTER TRAINING ===")
    post_embs, post_probs = get_all_embeddings(
        beliefs, gcn, projection, adapter, fact_head, llm,
        cached_text_embs, adj, x0, is_fact, magnitudes, device
    )
    post_pair_results = report_pairs(beliefs, post_embs, "after")
    report_classification(beliefs, post_probs, "after")

    # Summary
    print(f"\n{'─'*67}")
    print(f"  Summary")
    print(f"{'─'*67}")
    print(f"  Numeric precision pairs  before: {sum(pre_pair_results)}/{len(pre_pair_results)}  after: {sum(post_pair_results)}/{len(post_pair_results)}")
    delta = sum(post_pair_results) - sum(pre_pair_results)
    print(f"  Delta: {'+' if delta >= 0 else ''}{delta}")
    if sum(post_pair_results) > sum(pre_pair_results):
        print("\n  ✓ Model improved numeric precision over baseline")
    else:
        print("\n  ~ No improvement — try more epochs or lower threshold")

    # Visualise
    if do_visualize:
        print("\nGenerating visualisations …")
        out_dir = Path("scripts/experiments/typed_gnn_results")
        visualize(
            beliefs, gcn, adj, x0, is_fact, magnitudes,
            pre_embs.detach(), post_embs.detach(),
            loss_num_hist, loss_cls_hist, out_dir,
        )
        print(f"  Plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs",    type=int,  default=DEFAULT_EPOCHS)
    parser.add_argument("--visualize", action="store_true", help="Save all visualisation plots")
    args = parser.parse_args()
    run_experiment(DATASET, args.epochs, args.visualize)


if __name__ == "__main__":
    main()
