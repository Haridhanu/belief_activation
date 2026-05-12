#!/usr/bin/env python3
"""
Numeric Precision Probe
=======================
Can a dummy GCN + frozen LLM + numeric adapter learn to distinguish
beliefs that differ only in numeric precision?

Architecture:
    Belief nodes (text + magnitude)
        ↓  frozen LLM word-embeddings → initial node features
    2-layer GCN  [trainable]
        ↓  node embeddings
    Projection  [trainable]
        ↓  prefix token (position 0)
    Frozen LLM  — gradient still flows through frozen weights back to prefix
        ↓  contextualized prefix output
    NumericAdapter  [trainable]  — fuses LLM output with sinusoidal(magnitude)
        ↓  precision-aware embedding
    InfoNCE contrastive loss

Usage:
    python scripts/experiments/numeric_precision_probe.py
    python scripts/experiments/numeric_precision_probe.py --pdf paper.pdf
    python scripts/experiments/numeric_precision_probe.py --epochs 200 --plot
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ─────────────────────────────────────────────────────────────────────

LLM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
GCN_HIDDEN = 256
NUM_FREQS = 16          # sinusoidal frequencies for magnitude encoding
GRAPH_THRESHOLD = 0.3   # cosine-sim threshold for graph edges
TEMPERATURE = 0.07      # InfoNCE temperature
N_NEGATIVES = 6         # negatives per anchor in contrastive loss
LR = 3e-4
DEFAULT_EPOCHS = 150

# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class NumericBelief:
    text: str
    entity: str
    attribute: str
    magnitude: float
    unit: str


# Numeric facts from "Attention Is All You Need" (Vaswani et al., 2017).
# Used when no PDF is supplied.  Includes several pairs that test precision:
#   • dropout=0.1 vs label_smoothing=0.1  — same magnitude, different concept
#   • dropout=0.1 vs dropout_big=0.3      — same concept, different magnitude
#   • BLEU 28.4 vs BLEU 41.0             — same concept, close but different
#   • encoder N=6 vs decoder N=6          — same magnitude, different entity
AIAYN_BELIEFS: list[NumericBelief] = [
    NumericBelief("The model uses d_model = 512 dimensions.",           "model",           "dimensions",   512.0,    "dims"),
    NumericBelief("The feed-forward sublayer has d_ff = 2048.",         "feed_forward",    "inner_dim",    2048.0,   "dims"),
    NumericBelief("The model uses h = 8 parallel attention heads.",     "attention",       "num_heads",    8.0,      "heads"),
    NumericBelief("Each attention head has d_k = 64 dimensions.",       "attention_head",  "key_dim",      64.0,     "dims"),
    NumericBelief("The encoder is composed of N = 6 identical layers.", "encoder",         "num_layers",   6.0,      "layers"),
    NumericBelief("The decoder is composed of N = 6 identical layers.", "decoder",         "num_layers",   6.0,      "layers"),
    NumericBelief("Dropout probability P_drop = 0.1 is applied.",       "dropout",         "probability",  0.1,      ""),
    NumericBelief("Label smoothing of epsilon_ls = 0.1 was used.",      "label_smoothing", "epsilon",      0.1,      ""),
    NumericBelief("Warmup_steps = 4000 steps for learning rate.",       "training",        "warmup_steps", 4000.0,   "steps"),
    NumericBelief("Training ran for 100000 steps total.",               "training",        "total_steps",  100000.0, "steps"),
    NumericBelief("Beam search uses beam size 4.",                       "beam_search",     "beam_size",    4.0,      ""),
    NumericBelief("Length penalty alpha = 0.6 is applied.",             "beam_search",     "length_penalty", 0.6,   ""),
    NumericBelief("BLEU score of 28.4 on WMT 2014 English-German.",    "translation_ende","bleu",         28.4,     "BLEU"),
    NumericBelief("BLEU score of 41.0 on WMT 2014 English-French.",    "translation_enfr","bleu",         41.0,     "BLEU"),
    NumericBelief("Shared vocabulary of 37000 BPE tokens.",             "vocabulary",      "size",         37000.0,  "tokens"),
    NumericBelief("Training used 8 NVIDIA P100 GPUs.",                  "training",        "num_gpus",     8.0,      "GPUs"),
    NumericBelief("Big model dropout rate is 0.3.",                     "dropout_big",     "probability",  0.3,      ""),
    NumericBelief("Learning rate schedule uses factor = 2.",            "optimizer",       "lr_factor",    2.0,      ""),
    NumericBelief("Base model training time is 12 hours.",              "training_base",   "hours",        12.0,     "hours"),
    NumericBelief("Big model training time is 3.5 days.",               "training_big",    "days",         3.5,      "days"),
]

# Index pairs we care about for precision evaluation
# (label_a, label_b, idx_a, idx_b, expected_relation)
EVAL_PAIRS = [
    ("dropout 0.1",    "label_smooth 0.1",  6,  7,  "CLOSE"),  # same magnitude
    ("dropout 0.1",    "dropout_big 0.3",   6,  16, "FAR"),    # same attribute, different value
    ("BLEU 28.4",      "BLEU 41.0",         12, 13, "FAR"),    # same attribute, close values
    ("encoder N=6",    "decoder N=6",        4,  5,  "CLOSE"),  # same magnitude
    ("8 heads",        "8 GPUs",             2,  15, "CLOSE"),  # same magnitude, different concept
    ("512 dims",       "64 key_dim",         0,  3,  "FAR"),    # clearly different
]

# ── PDF extraction (optional) ──────────────────────────────────────────────────

def extract_beliefs_from_pdf(pdf_path: str) -> list[NumericBelief]:
    try:
        import pypdf
    except ImportError:
        print("pypdf not installed (pip install pypdf). Falling back to built-in AIAYN facts.")
        return []

    reader = pypdf.PdfReader(pdf_path)
    text = " ".join(page.extract_text() or "" for page in reader.pages)

    beliefs: list[NumericBelief] = []
    # Split into sentences and look for numeric expressions
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for m in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*(?:\s*[=:]\s*|\s+of\s+|\s+is\s+|\s+are\s+))"
            r"([\d,]+(?:\.\d+)?)"
            r"(?:\s+([A-Za-z]+))?",
            sentence,
        ):
            try:
                magnitude = float(m.group(2).replace(",", ""))
            except ValueError:
                continue
            if magnitude <= 0:
                continue
            entity = re.sub(r"\s*[=:]\s*$", "", m.group(1)).strip()
            beliefs.append(
                NumericBelief(
                    text=sentence.strip()[:200],
                    entity=entity,
                    attribute="value",
                    magnitude=magnitude,
                    unit=m.group(3) or "",
                )
            )
    # Deduplicate by (magnitude, entity) and cap
    seen: set[tuple[str, float]] = set()
    deduped: list[NumericBelief] = []
    for b in beliefs:
        key = (b.entity, b.magnitude)
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    return deduped[:30]

# ── Neural modules ─────────────────────────────────────────────────────────────

class SimpleGCN(nn.Module):
    """Two-layer GCN from scratch — no torch_geometric needed."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.W1(adj @ x))
        return self.W2(adj @ h)


class Projection(nn.Module):
    """Projects GCN output into the frozen LLM's input space."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.linear(x)))


class NumericAdapter(nn.Module):
    """
    Fuses frozen-LLM output with sinusoidal magnitude encoding.

    The sinusoidal encoding gives the network a multi-resolution view of
    the numeric value: low frequencies capture order-of-magnitude, high
    frequencies capture fine precision differences.
    """

    def __init__(self, llm_dim: int, num_freqs: int = NUM_FREQS) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        enc_dim = num_freqs * 2
        self.fc1 = nn.Linear(llm_dim + enc_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def _sinusoidal_encode(self, magnitude: float, device: torch.device) -> torch.Tensor:
        log_m = math.log(abs(magnitude) + 1e-8)
        # Frequencies span 0.01–1e4 in log-space to cover the AIAYN range
        freqs = torch.exp(torch.linspace(-2.0, 4.0, self.num_freqs, device=device))
        args = log_m * freqs
        return torch.cat([torch.sin(args), torch.cos(args)])  # (2*num_freqs,)

    def forward(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        """x: (B, llm_dim) or (llm_dim,)"""
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        enc = self._sinusoidal_encode(magnitude, x.device).unsqueeze(0).expand(x.shape[0], -1)
        h = F.gelu(self.fc1(torch.cat([x, enc], dim=-1)))
        out = self.norm(self.fc2(h))
        return out.squeeze(0) if squeeze else out

# ── Graph construction ─────────────────────────────────────────────────────────

def build_normalized_adj(x0: torch.Tensor, threshold: float = GRAPH_THRESHOLD) -> torch.Tensor:
    """Row-normalised adjacency A_hat = D^{-1}(A + I) built from cosine similarity."""
    with torch.no_grad():
        sim = F.normalize(x0, dim=-1) @ F.normalize(x0, dim=-1).t()
    A = (sim >= threshold).float()
    A = (A + torch.eye(A.shape[0], device=A.device)).clamp(max=1.0)
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / deg

# ── LLM helpers ────────────────────────────────────────────────────────────────

def _word_embeddings(llm: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract raw word embeddings (no positional encoding) from LLM."""
    emb_layer = getattr(llm, "embeddings", None)
    if emb_layer is not None and hasattr(emb_layer, "word_embeddings"):
        with torch.no_grad():
            return emb_layer.word_embeddings(input_ids)  # (1, seq, D)
    # Fallback: run the full model and take last hidden state (no grad, detached)
    with torch.no_grad():
        return llm(input_ids=input_ids).last_hidden_state.detach()


def precompute_text_embs(
    beliefs: list[NumericBelief],
    llm: nn.Module,
    tokenizer,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    For each belief, cache word embeddings of the text tokens (CLS stripped).
    These are fixed throughout training — only the GCN prefix changes.
    """
    cached = []
    for b in beliefs:
        enc = tokenizer(
            b.text, return_tensors="pt", truncation=True, max_length=64, padding=False
        ).to(device)
        word_embs = _word_embeddings(llm, enc["input_ids"])  # (1, seq+2, D)
        # Drop CLS (index 0), keep text tokens + SEP
        cached.append(word_embs.squeeze(0)[1:])  # (seq+1, D)
    return cached


def batch_prefix_forward(
    llm: nn.Module,
    proj_embs: torch.Tensor,          # (N, D)  — differentiable GCN outputs
    cached_text_embs: list[torch.Tensor],  # list of (seq_i, D)
    device: torch.device,
) -> torch.Tensor:
    """
    Injects proj_embs as prefix tokens at position 0, pads text embeddings,
    runs the frozen LLM, and returns the output at each prefix position.

    Gradient flows: loss → adapter → output[:,0,:] → frozen LLM ops → proj_embs
    The frozen LLM weights don't update (requires_grad=False), but the
    computation graph through them remains intact, so dL/d(proj_embs) is valid.
    """
    N, D = proj_embs.shape
    max_len = max(t.shape[0] for t in cached_text_embs)

    # Build padding tensor from fixed (no-grad) text embeddings
    padding = torch.zeros(N, max_len, D, device=device)
    attention_mask = torch.zeros(N, 1 + max_len, device=device)
    attention_mask[:, 0] = 1.0  # prefix always attended

    for i, text_emb in enumerate(cached_text_embs):
        seq_len = text_emb.shape[0]
        padding[i, :seq_len] = text_emb.to(device)
        attention_mask[i, 1 : 1 + seq_len] = 1.0

    # torch.cat preserves the gradient connection through proj_embs
    inputs_embeds = torch.cat([proj_embs.unsqueeze(1), padding], dim=1)  # (N, 1+max_len, D)

    out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0, :]  # (N, D) — prefix positions only

# ── Loss ───────────────────────────────────────────────────────────────────────

def infonce_loss(
    anchor: torch.Tensor,    # (D,)
    positive: torch.Tensor,  # (D,)
    negatives: torch.Tensor, # (K, D)
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    all_targets = torch.cat([positive.unsqueeze(0), negatives], dim=0)  # (1+K, D)
    a_norm = F.normalize(anchor.unsqueeze(0), dim=-1)          # (1, D)
    t_norm = F.normalize(all_targets, dim=-1)                   # (1+K, D)
    logits = (a_norm @ t_norm.t()).squeeze(0) / temperature     # (1+K,)
    label = torch.zeros(1, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits.unsqueeze(0), label)

# ── Pair sampling ──────────────────────────────────────────────────────────────

def find_positive(anchor_idx: int, magnitudes: list[float]) -> int:
    """Return index of belief with closest log-magnitude (excluding self)."""
    log_m = math.log(magnitudes[anchor_idx] + 1e-8)
    best_idx, best_diff = anchor_idx, float("inf")
    for i, m in enumerate(magnitudes):
        if i == anchor_idx:
            continue
        d = abs(log_m - math.log(m + 1e-8))
        if d < best_diff:
            best_diff, best_idx = d, i
    return best_idx


def sample_negatives(anchor_idx: int, pos_idx: int, N: int, k: int) -> list[int]:
    pool = [i for i in range(N) if i not in (anchor_idx, pos_idx)]
    chosen = np.random.default_rng().choice(pool, size=min(k, len(pool)), replace=False)
    return chosen.tolist()

# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_all_embeddings(
    beliefs: list[NumericBelief],
    gcn: SimpleGCN,
    projection: Projection,
    adapter: NumericAdapter,
    llm: nn.Module,
    cached_text_embs: list[torch.Tensor],
    adj: torch.Tensor,
    x0: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    h = gcn(x0, adj)
    proj = projection(h)
    prefix_outs = batch_prefix_forward(llm, proj, cached_text_embs, device)
    return torch.stack(
        [adapter(prefix_outs[i], beliefs[i].magnitude) for i in range(len(beliefs))]
    )


def report_pairs(
    beliefs: list[NumericBelief],
    embs: torch.Tensor,
    stage: str,
) -> list[bool]:
    pairs = [p for p in EVAL_PAIRS if p[2] < len(beliefs) and p[3] < len(beliefs)]
    print(f"\n  {'─'*63}")
    print(f"  Precision pair cosine similarities — {stage}")
    print(f"  {'─'*63}")
    print(f"  {'Pair':<36} {'Sim':>6}  {'Expected':>8}  {'Pass':>4}")
    print(f"  {'─'*36} {'─'*6}  {'─'*8}  {'─'*4}")
    results = []
    for la, lb, i, j, expected in pairs:
        sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
        passed = (expected == "CLOSE" and sim > 0.5) or (expected == "FAR" and sim < 0.5)
        results.append(passed)
        mark = "✓" if passed else "✗"
        pair = f"{la} ↔ {lb}"
        print(f"  {pair:<36} {sim:>6.3f}  {expected:>8}  {mark:>4}")
    print(f"\n  Passed: {sum(results)}/{len(results)}")
    return results

# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(
    beliefs: list[NumericBelief],
    epochs: int,
    plot: bool,
) -> None:
    print(f"\nNumeric Precision Probe")
    print(f"  beliefs : {len(beliefs)}")
    print(f"  model   : {LLM_NAME}")
    print(f"  epochs  : {epochs}")

    device = torch.device("cpu")

    # ── Load frozen LLM ───────────────────────────────────────────────────────
    print("\nLoading frozen LLM …")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed. Run: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModel.from_pretrained(LLM_NAME).to(device)
    for p in llm.parameters():
        p.requires_grad = False  # freeze weights; gradient still flows through

    # ── Initial node features (frozen LLM mean-pool) ──────────────────────────
    print("Computing initial node embeddings …")
    x0_list: list[torch.Tensor] = []
    for b in beliefs:
        enc = tokenizer(b.text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = llm(**enc)
        x0_list.append(out.last_hidden_state.mean(dim=1).squeeze(0))
    x0 = torch.stack(x0_list).detach()  # (N, EMB_DIM) — fixed throughout

    # ── Precompute text word-embeddings for prefix injection ───────────────────
    cached_text_embs = precompute_text_embs(beliefs, llm, tokenizer, device)

    # ── Build graph ────────────────────────────────────────────────────────────
    adj = build_normalized_adj(x0, GRAPH_THRESHOLD)
    n_edges = int((adj > 0).sum().item()) - len(beliefs)  # exclude self-loops
    print(f"Graph: {len(beliefs)} nodes, {n_edges} edges (threshold={GRAPH_THRESHOLD})")

    # ── Trainable modules ──────────────────────────────────────────────────────
    gcn = SimpleGCN(EMB_DIM, GCN_HIDDEN, EMB_DIM).to(device)
    projection = Projection(EMB_DIM, EMB_DIM).to(device)
    adapter = NumericAdapter(EMB_DIM, NUM_FREQS).to(device)

    optimizer = torch.optim.Adam(
        list(gcn.parameters()) + list(projection.parameters()) + list(adapter.parameters()),
        lr=LR,
    )

    magnitudes = [b.magnitude for b in beliefs]
    N = len(beliefs)

    # ── Baseline (before training) ─────────────────────────────────────────────
    print("\n=== BEFORE TRAINING (baseline) ===")
    pre_embs = get_all_embeddings(beliefs, gcn, projection, adapter, llm, cached_text_embs, adj, x0, device)
    pre_results = report_pairs(beliefs, pre_embs, "before training")

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs …")
    loss_history: list[float] = []

    for epoch in range(epochs):
        # Single GCN forward for all nodes
        h = gcn(x0, adj)           # (N, EMB_DIM)
        proj = projection(h)       # (N, EMB_DIM)

        # Batch prefix injection through frozen LLM — gradient flows to proj
        prefix_outs = batch_prefix_forward(llm, proj, cached_text_embs, device)  # (N, EMB_DIM)

        # Adapter outputs for all nodes
        all_embs = torch.stack(
            [adapter(prefix_outs[i], magnitudes[i]) for i in range(N)]
        )  # (N, EMB_DIM)

        # Accumulate InfoNCE loss over all anchors
        total_loss = torch.zeros(1, device=device)
        for i in range(N):
            pos_idx = find_positive(i, magnitudes)
            neg_indices = sample_negatives(i, pos_idx, N, N_NEGATIVES)
            total_loss = total_loss + infonce_loss(
                all_embs[i],
                all_embs[pos_idx],
                all_embs[neg_indices],
            )

        loss = total_loss / N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1:4d}/{epochs}  loss {loss.item():.4f}")

    # ── After training ─────────────────────────────────────────────────────────
    print("\n=== AFTER TRAINING ===")
    post_embs = get_all_embeddings(beliefs, gcn, projection, adapter, llm, cached_text_embs, adj, x0, device)
    post_results = report_pairs(beliefs, post_embs, "after training")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*63}")
    print(f"  Summary")
    print(f"{'─'*63}")
    pre_pass = sum(pre_results)
    post_pass = sum(post_results)
    total = len(pre_results)
    print(f"  Before : {pre_pass}/{total} pairs correct")
    print(f"  After  : {post_pass}/{total} pairs correct")
    delta = post_pass - pre_pass
    sign = "+" if delta >= 0 else ""
    print(f"  Delta  : {sign}{delta}")
    if post_pass > pre_pass:
        print("\n  ✓ Model learned numeric precision (improvement over baseline)")
    elif post_pass == total:
        print("\n  ✓ Perfect precision discrimination achieved")
    else:
        print("\n  ✗ No improvement — check gradient flow or increase epochs")

    # ── Optional plot ──────────────────────────────────────────────────────────
    if plot:
        _plot(beliefs, pre_embs, post_embs, loss_history)


def _plot(
    beliefs: list[NumericBelief],
    pre_embs: torch.Tensor,
    post_embs: torch.Tensor,
    loss_history: list[float],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("\nmatplotlib/scikit-learn not installed. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # t-SNE colored by log(magnitude)
    log_mags = np.array([math.log(b.magnitude + 1e-8) for b in beliefs])
    labels = [f"{b.entity}\n{b.magnitude}" for b in beliefs]

    for ax, embs, title in [
        (axes[0], pre_embs, "Before training"),
        (axes[1], post_embs, "After training"),
    ]:
        coords = TSNE(n_components=2, perplexity=min(5, len(beliefs) - 1), random_state=42).fit_transform(
            embs.numpy()
        )
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=log_mags, cmap="plasma", s=80)
        for i, (x, y) in enumerate(coords):
            ax.annotate(labels[i], (x, y), fontsize=6, ha="center", va="bottom")
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label="log(magnitude)")

    # Loss curve
    axes[2].plot(loss_history)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("InfoNCE loss")
    axes[2].set_title("Training loss")

    out = Path("scripts/experiments/numeric_precision_probe_results.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved to {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF (optional; uses built-in AIAYN facts if omitted)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--plot", action="store_true", help="Save t-SNE + loss plot")
    args = parser.parse_args()

    if args.pdf:
        beliefs = extract_beliefs_from_pdf(args.pdf)
        if not beliefs:
            print("No numeric beliefs extracted from PDF. Falling back to built-in AIAYN facts.")
            beliefs = AIAYN_BELIEFS
        else:
            print(f"Extracted {len(beliefs)} numeric beliefs from {args.pdf}")
    else:
        beliefs = AIAYN_BELIEFS
        print(f"Using {len(beliefs)} built-in numeric facts from 'Attention Is All You Need'")

    run_experiment(beliefs, args.epochs, args.plot)


if __name__ == "__main__":
    main()
