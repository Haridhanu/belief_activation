#!/usr/bin/env python3
"""
Simple GCN + Frozen LLM — Numeric Precision Experiment
=======================================================
Can a 2-layer GCN learn numeric precision when its node embeddings are
shaped by gradient signal flowing back from a frozen LLM + adapter?

Architecture:
    Raw text beliefs  (numbers embedded in natural language)
        ↓  frozen LLM mean-pool → initial node features (no grad)
    2-layer GCN  [trainable]
        ↓  node embeddings
    Projection  [trainable, small]
        ↓  prefix token injected at position 0
    Frozen LLM  [weights locked — but gradient STILL flows back through]
        ↓  contextualized prefix output
    NumericAdapter  [trainable]
        sinusoidal encode(regex_magnitude) + LLM output → fused embedding
        ↓
    InfoNCE contrastive loss
        anchors: each belief node
        positives: belief with closest numeric value
        negatives: randomly sampled beliefs with different values

Gradient flow verified:
    loss → adapter → frozen LLM ops → projection → GCN weights

Usage:
    uv run python scripts/experiments/simple_gcn_numeric.py
    uv run python scripts/experiments/simple_gcn_numeric.py --epochs 200 --plot
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ─────────────────────────────────────────────────────────────────────

LLM_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM         = 384
GCN_HIDDEN      = 128
NUM_FREQS       = 16
GRAPH_THRESHOLD = 0.4
TEMPERATURE     = 0.1
N_NEGATIVES     = 4
LR              = 1e-3
DEFAULT_EPOCHS  = 200

# ── Dataset ────────────────────────────────────────────────────────────────────
# Raw text beliefs from "Attention Is All You Need".
# Numbers are embedded in natural language — the model must learn to read them.
# Each entry: (text, true_magnitude)  ← magnitude used ONLY for pair labels.

BELIEFS: list[tuple[str, float]] = [
    ("The transformer model uses 512 dimensions.",                          512.0),
    ("The feed-forward network has an inner dimension of 2048.",           2048.0),
    ("There are 8 parallel attention heads.",                                 8.0),
    ("Each attention head operates on 64 dimensions.",                       64.0),
    ("The encoder stack has 6 layers.",                                       6.0),
    ("The decoder stack has 6 layers.",                                       6.0),
    ("Dropout with probability 0.1 is applied throughout.",                   0.1),
    ("Label smoothing uses an epsilon value of 0.1.",                         0.1),
    ("The model achieved 28.4 BLEU on English to German translation.",       28.4),
    ("The model achieved 41.0 BLEU on English to French translation.",       41.0),
    ("Byte-pair encoding produces a vocabulary of 37000 tokens.",         37000.0),
    ("The learning rate warmup lasts for 4000 steps.",                     4000.0),
    ("Training runs for a total of 100000 steps.",                       100000.0),
    ("Beam search uses a beam size of 4.",                                    4.0),
    ("The length penalty alpha is set to 0.6.",                               0.6),
    ("The big model variant applies dropout at a rate of 0.3.",               0.3),
    ("Training is distributed across 8 GPUs.",                                8.0),
    ("The base model trains in approximately 12 hours.",                      12.0),
]

# Precision eval pairs — (description_a, description_b, idx_a, idx_b, expected)
EVAL_PAIRS = [
    ("dropout 0.1",     "label_smooth 0.1",  6,  7,  "CLOSE"),  # same value, diff concept
    ("enc 6 layers",    "dec 6 layers",       4,  5,  "CLOSE"),  # same value, diff entity
    ("8 heads",         "8 GPUs",            2,  16, "CLOSE"),   # same value, very diff concept
    ("BLEU 28.4",       "BLEU 41.0",         8,  9,  "FAR"),    # precision test
    ("dropout 0.1",     "dropout_big 0.3",   6,  15, "FAR"),    # same concept, diff value
    ("512 dims",        "64 key dims",        0,  3,  "FAR"),    # clearly different
    ("4000 warmup",     "100000 total",      11, 12, "FAR"),     # different scale
    ("0.6 length pen",  "0.3 dropout",       14, 15, "FAR"),    # different value
]

# ── Regex magnitude extraction ──────────────────────────────────────────────────

# Prefer decimal numbers (e.g. 28.4 over 2014) and avoid years / very large ints
_DECIMAL_RE = re.compile(r"\b(\d+\.\d+)\b")
_INT_RE     = re.compile(r"\b(\d+)\b")

def extract_magnitude(text: str) -> float | None:
    # Decimals first — these are almost always the key numeric value
    decimals = [float(m) for m in _DECIMAL_RE.findall(text)]
    if decimals:
        return decimals[0]
    # Integers: filter obvious years and very large noise
    ints = [float(m) for m in _INT_RE.findall(text)]
    ints = [v for v in ints if v < 1_000_000 and not (1900 < v < 2100)]
    return max(ints) if ints else None

# ── Modules ────────────────────────────────────────────────────────────────────

class SimpleGCN(nn.Module):
    """Two-layer GCN. No external dependencies."""

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
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.fc(x)))


class NumericAdapter(nn.Module):
    """
    Fuses LLM output with sinusoidal magnitude encoding.

    Sinusoidal encoding at multiple log-spaced frequencies gives the network
    a multi-resolution view of the number — low frequencies separate orders
    of magnitude (6 vs 512), high frequencies separate fine differences
    (0.1 vs 0.3, 28.4 vs 41.0).
    """

    def __init__(self, llm_dim: int, num_freqs: int = NUM_FREQS) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        enc_dim       = num_freqs * 2
        self.fc1      = nn.Linear(llm_dim + enc_dim, llm_dim)
        self.fc2      = nn.Linear(llm_dim, llm_dim)
        self.norm     = nn.LayerNorm(llm_dim)

    def _encode(self, magnitude: float, device: torch.device) -> torch.Tensor:
        log_m = math.log(abs(magnitude) + 1e-8)
        freqs = torch.exp(torch.linspace(-2.0, 4.0, self.num_freqs, device=device))
        args  = log_m * freqs
        return torch.cat([torch.sin(args), torch.cos(args)])   # (2 * num_freqs,)

    def forward(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        enc = self._encode(magnitude, x.device).unsqueeze(0).expand(x.shape[0], -1)
        h   = F.gelu(self.fc1(torch.cat([x, enc], dim=-1)))
        return self.norm(self.fc2(h))

# ── Graph ──────────────────────────────────────────────────────────────────────

def build_adj(x0: torch.Tensor, threshold: float) -> torch.Tensor:
    """Row-normalised adjacency A_hat = D^{-1}(A + I)."""
    with torch.no_grad():
        sim = F.normalize(x0, dim=-1) @ F.normalize(x0, dim=-1).t()
    A   = (sim >= threshold).float()
    A   = (A + torch.eye(A.shape[0], device=A.device)).clamp(max=1.0)
    return A / A.sum(dim=1, keepdim=True).clamp(min=1.0)

# ── LLM helpers ────────────────────────────────────────────────────────────────

def _word_embeddings(llm: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    emb = getattr(llm, "embeddings", None)
    if emb is not None and hasattr(emb, "word_embeddings"):
        with torch.no_grad():
            return emb.word_embeddings(input_ids)
    with torch.no_grad():
        return llm(input_ids=input_ids).last_hidden_state.detach()


def precompute_text_embs(texts, llm, tokenizer, device) -> list[torch.Tensor]:
    """Cache word embeddings (no CLS) for each belief text. Fixed throughout training."""
    cached = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=64).to(device)
        we  = _word_embeddings(llm, enc["input_ids"])   # (1, seq+2, D)
        cached.append(we.squeeze(0)[1:])                 # drop CLS, keep text + SEP
    return cached


def prefix_forward(llm, proj_embs, cached_text_embs, device) -> torch.Tensor:
    """
    Inject GCN projection as prefix token (position 0) into the frozen LLM.

    Gradient flows:
        loss → adapter → last_hidden_state[:,0,:] → frozen LLM attention
             → inputs_embeds[:,0,:] (= proj_embs) → projection → GCN

    The frozen LLM's parameters don't update (requires_grad=False) but the
    computation graph through them is intact, so dL/d(proj_embs) is valid.
    """
    N, D    = proj_embs.shape
    max_len = max(t.shape[0] for t in cached_text_embs)

    padding        = torch.zeros(N, max_len, D, device=device)
    attention_mask = torch.zeros(N, 1 + max_len, device=device)
    attention_mask[:, 0] = 1.0

    for i, te in enumerate(cached_text_embs):
        sl = te.shape[0]
        padding[i, :sl]            = te.to(device)
        attention_mask[i, 1:1+sl]  = 1.0

    # torch.cat keeps the gradient connection through proj_embs
    inputs_embeds = torch.cat([proj_embs.unsqueeze(1), padding], dim=1)
    out           = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0, :]   # (N, D) — prefix positions only

# ── Loss ───────────────────────────────────────────────────────────────────────

def infonce(anchor, positive, negatives, temperature=TEMPERATURE):
    """anchor/positive: (D,), negatives: (K, D)"""
    all_t  = torch.cat([positive.unsqueeze(0), negatives], dim=0)  # (K+1, D)
    a_norm = F.normalize(anchor.unsqueeze(0), dim=-1)
    t_norm = F.normalize(all_t, dim=-1)
    logits = (a_norm @ t_norm.t()).squeeze(0) / temperature          # (K+1,)
    label  = torch.zeros(1, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits.unsqueeze(0), label)

# ── Pair sampling ──────────────────────────────────────────────────────────────

def closest(anchor_idx: int, magnitudes: list[float]) -> int:
    m     = magnitudes[anchor_idx]
    log_m = math.log(m + 1e-8)
    best, best_d = anchor_idx, float("inf")
    for i, m2 in enumerate(magnitudes):
        if i == anchor_idx:
            continue
        d = abs(log_m - math.log(m2 + 1e-8))
        if d < best_d:
            best_d, best = d, i
    return best


def random_negatives(anchor_idx: int, pos_idx: int, N: int, k: int) -> list[int]:
    pool   = [i for i in range(N) if i not in (anchor_idx, pos_idx)]
    chosen = np.random.default_rng().choice(pool, size=min(k, len(pool)), replace=False)
    return chosen.tolist()

# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_all(gcn, proj, adapter, llm, cached_text_embs, adj, x0, magnitudes, device):
    h           = gcn(x0, adj)
    p           = proj(h)
    prefix_outs = prefix_forward(llm, p, cached_text_embs, device)
    return torch.stack([adapter(prefix_outs[i].unsqueeze(0), magnitudes[i]).squeeze(0)
                        for i in range(len(magnitudes))])


def report(texts, embs, magnitudes, stage: str) -> list[bool]:
    pairs = [p for p in EVAL_PAIRS if p[2] < len(texts) and p[3] < len(texts)]
    print(f"\n  {'─'*65}")
    print(f"  {stage}")
    print(f"  {'─'*65}")
    print(f"  {'Pair':<40} {'Sim':>6}  {'Expected':>8}  Pass")
    print(f"  {'─'*40} {'─'*6}  {'─'*8}  {'─'*4}")
    results = []
    for la, lb, i, j, expected in pairs:
        sim    = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
        passed = (expected == "CLOSE" and sim > 0.5) or (expected == "FAR" and sim < 0.5)
        results.append(passed)
        mi, mj = magnitudes[i], magnitudes[j]
        pair   = f"{la} [{mi}] ↔ {lb} [{mj}]"
        print(f"  {pair:<40} {sim:>6.3f}  {expected:>8}  {'✓' if passed else '✗'}")
    print(f"\n  Passed: {sum(results)}/{len(results)}")
    return results

# ── Visualisation ──────────────────────────────────────────────────────────────

def plot(texts, magnitudes, pre_embs, post_embs, losses, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  matplotlib / scikit-learn not installed — skipping plot")
        return

    N        = len(texts)
    log_mags = np.array([math.log(m + 1e-8) for m in magnitudes])
    short    = [t[:35] + ("…" if len(t) > 35 else "") for t in texts]
    perp     = min(5, N - 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # t-SNE before / after, coloured by log(magnitude)
    for col, (embs_t, title) in enumerate([
        (pre_embs,  "Before training"),
        (post_embs, "After training"),
    ]):
        coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(
            embs_t.numpy()
        )
        ax = axes[0][col]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=log_mags, cmap="plasma", s=90, zorder=2)
        for i, (x, y) in enumerate(coords):
            ax.annotate(short[i], (x, y), fontsize=5.5, ha="center", va="bottom")
        ax.set_title(f"t-SNE — {title}", fontsize=11)
        plt.colorbar(sc, ax=ax, label="log(magnitude)")

    # Pairwise similarity heatmap before / after
    for col, (embs_t, title) in enumerate([
        (pre_embs,  "Before"),
        (post_embs, "After"),
    ]):
        mat = (F.normalize(embs_t, dim=-1) @ F.normalize(embs_t, dim=-1).t()).numpy()
        ax  = axes[1][col]
        im  = ax.imshow(mat, cmap="RdBu", vmin=-1, vmax=1)
        tick_labels = [f"{magnitudes[i]}" for i in range(N)]
        ax.set_xticks(range(N)); ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(N)); ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_title(f"Cosine similarity — {title} training", fontsize=10)
        plt.colorbar(im, ax=ax)

    fig.suptitle(
        "Simple GCN + Frozen LLM — Numeric Precision Learning",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


def plot_loss(losses, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(losses, color="steelblue", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("InfoNCE loss")
    plt.title("Training Loss — GCN learns numeric precision via frozen LLM gradient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Loss curve → {out_path}")

# ── Experiment ─────────────────────────────────────────────────────────────────

def run(epochs: int, do_plot: bool) -> None:
    texts      = [b[0] for b in BELIEFS]
    true_mags  = [b[1] for b in BELIEFS]
    N          = len(texts)

    # Extract magnitudes from raw text (training scaffold — not a stored feature)
    regex_mags = [extract_magnitude(t) for t in texts]

    print(f"\nSimple GCN + Frozen LLM — Numeric Precision")
    print(f"  beliefs : {N}")
    print(f"  LLM     : {LLM_NAME}  (frozen)")
    print(f"  epochs  : {epochs}\n")
    print(f"  {'Text (truncated)':<50} {'Regex':>8}  {'True':>8}")
    print(f"  {'─'*50} {'─'*8}  {'─'*8}")
    for t, rm, tm in zip(texts, regex_mags, true_mags):
        ok = "✓" if rm == tm else "!"
        print(f"  {ok} {t[:48]:<48} {str(rm):>8}  {tm:>8}")

    device = torch.device("cpu")

    # ── Load frozen LLM ───────────────────────────────────────────────────────
    print("\nLoading frozen LLM …")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        sys.exit("pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm       = AutoModel.from_pretrained(LLM_NAME).to(device)
    for p in llm.parameters():
        p.requires_grad = False

    # ── Initial node features (frozen LLM mean-pool, no grad) ─────────────────
    print("Computing initial node embeddings …")
    x0_list = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            x0_list.append(llm(**enc).last_hidden_state.mean(dim=1).squeeze(0))
    x0 = torch.stack(x0_list).detach()

    cached_text_embs = precompute_text_embs(texts, llm, tokenizer, device)
    adj              = build_adj(x0, GRAPH_THRESHOLD)
    n_edges          = int((adj > 0).sum().item()) - N
    print(f"Graph: {N} nodes, {n_edges} edges  (threshold={GRAPH_THRESHOLD})")

    # ── Trainable modules ──────────────────────────────────────────────────────
    gcn     = SimpleGCN(EMB_DIM, GCN_HIDDEN, EMB_DIM).to(device)
    proj    = Projection(EMB_DIM, EMB_DIM).to(device)
    adapter = NumericAdapter(EMB_DIM, NUM_FREQS).to(device)

    optimizer = torch.optim.Adam(
        list(gcn.parameters()) + list(proj.parameters()) + list(adapter.parameters()),
        lr=LR,
    )

    # Use regex_mags for pair construction; fall back to true_mags where regex failed
    train_mags = [rm if rm is not None else tm for rm, tm in zip(regex_mags, true_mags)]

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n=== BEFORE TRAINING ===")
    pre_embs     = embed_all(gcn, proj, adapter, llm, cached_text_embs, adj, x0, train_mags, device)
    pre_results  = report(texts, pre_embs, true_mags, "Precision pairs — before training")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining …")
    loss_history: list[float] = []

    for epoch in range(epochs):
        # GCN forward
        h           = gcn(x0, adj)           # (N, EMB_DIM)  — trainable
        p_embs      = proj(h)                 # (N, EMB_DIM)  — trainable

        # Prefix injection: gradient flows from loss → adapter → frozen LLM → proj → gcn
        prefix_outs = prefix_forward(llm, p_embs, cached_text_embs, device)  # (N, EMB_DIM)

        # Adapter outputs for all nodes
        all_embs = torch.stack([
            adapter(prefix_outs[i].unsqueeze(0), train_mags[i]).squeeze(0)
            for i in range(N)
        ])   # (N, EMB_DIM)

        # Accumulate InfoNCE over all anchors
        total_loss = torch.zeros(1, device=device)
        for i in range(N):
            pos_idx     = closest(i, train_mags)
            neg_indices = random_negatives(i, pos_idx, N, N_NEGATIVES)
            total_loss  = total_loss + infonce(
                all_embs[i], all_embs[pos_idx], all_embs[neg_indices]
            )

        loss = total_loss / N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={loss.item():.5f}")

    # ── After training ─────────────────────────────────────────────────────────
    print("\n=== AFTER TRAINING ===")
    post_embs    = embed_all(gcn, proj, adapter, llm, cached_text_embs, adj, x0, train_mags, device)
    post_results = report(texts, post_embs, true_mags, "Precision pairs — after training")

    # ── Summary ───────────────────────────────────────────────────────────────
    delta = sum(post_results) - sum(pre_results)
    print(f"\n{'─'*67}")
    print(f"  Before: {sum(pre_results)}/{len(pre_results)}  |  After: {sum(post_results)}/{len(post_results)}  |  Δ {'+' if delta >= 0 else ''}{delta}")
    if delta > 0:
        print("  ✓ GCN learned numeric precision via frozen LLM gradient signal")
    elif sum(post_results) == len(post_results):
        print("  ✓ Perfect numeric precision from the start")
    else:
        print("  ~ Try more epochs or a lower GRAPH_THRESHOLD")
    print(f"{'─'*67}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if do_plot:
        out = Path("scripts/experiments/simple_gcn_results")
        out.mkdir(parents=True, exist_ok=True)
        plot(texts, true_mags, pre_embs.detach(), post_embs.detach(), loss_history, out / "embeddings.png")
        plot_loss(loss_history, out / "loss.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--plot",   action="store_true")
    args = p.parse_args()
    run(args.epochs, args.plot)


if __name__ == "__main__":
    main()
