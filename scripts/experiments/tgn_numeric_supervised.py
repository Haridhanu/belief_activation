#!/usr/bin/env python3
"""
TGN + Frozen LLM — Numeric Supervised Training (Dummy Experiment)
=================================================================
Alternating loop per step:

    A) PSRO-sim   tgn.train_step(fake_events)  →  link_head + GRU + msg_encoder
                  via psro_optimizer  (does NOT touch mem_to_emb)

    B) Numeric    detached_memory → mem_to_emb → Projection → frozen LLM prefix
                  → NumericHead → MSE(pred_log_mag, true_log_mag)
                  via numeric_optimizer  (only touches mem_to_emb + head layers)

Gradient flow verified: ∂numeric_loss/∂mem_to_emb.weight ≠ 0

Corpus: 15 hardcoded paper-like passages (no PDF download needed).
Heuristic extractor: decimal-first regex, year filter on integers.

Usage:
    uv run python scripts/experiments/tgn_numeric_supervised.py
    uv run python scripts/experiments/tgn_numeric_supervised.py --steps 80 --plot
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from multi_agent.tgn import TGNModule

# ── Config ─────────────────────────────────────────────────────────────
LLM_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM     = 384   # matches MiniLM output; also tgn.emb_dim
MEM_DIM     = 64    # tgn.memory_dim
TIME_DIM    = 16
PSRO_LR     = 1e-3
NUMERIC_LR  = 1e-4
N_EVENTS    = 20    # fake PSRO edges per step
LOG_EVERY   = 10

# ── Dummy corpus ─────────────────────────────────────────────────────────
# Simulates heuristic extraction from ArXiv PDFs.
# (text, true_magnitude) — true_magnitude is ground truth for the loss.
PASSAGES: list[tuple[str, float]] = [
    ("ResNet-50 achieves 76.1% top-1 accuracy on ImageNet.",            76.1),
    ("The model trains with a batch size of 256 on 8 GPUs.",           256.0),
    ("Learning rate is 0.001 with cosine annealing schedule.",          0.001),
    ("Attention uses 512 hidden dimensions per head.",                  512.0),
    ("Dropout of probability 0.1 is applied throughout.",                0.1),
    ("The model achieves F1 of 0.923 on SQuAD 2.0.",                   0.923),
    ("Training runs for 300 epochs until convergence.",                300.0),
    ("Vocabulary size is 32000 tokens after byte-pair encoding.",     32000.0),
    ("BLEU score of 34.6 on WMT newstest2014 English-German.",          34.6),
    ("Weight decay of 0.0001 is used as L2 regularization.",          0.0001),
    ("The transformer has 6 encoder and 6 decoder layers.",               6.0),
    ("Beam search uses a width of 4 during decoding.",                    4.0),
    ("Perplexity of 57.3 is achieved on Penn Treebank test split.",     57.3),
    ("We use 16 attention heads in each multi-head attention block.",    16.0),
    ("Top-5 accuracy on CIFAR-100 reaches 91.8 with our augmentation.", 91.8),
]

# ── Heuristic magnitude extractor (same logic as simple_gcn_numeric) ───
_DECIMAL_RE = re.compile(r"\b(\d+\.\d+)\b")
_INT_RE     = re.compile(r"\b(\d+)\b")

def extract_magnitude(text: str) -> float | None:
    decimals = [float(m) for m in _DECIMAL_RE.findall(text)]
    if decimals:
        return decimals[0]
    ints = [float(m) for m in _INT_RE.findall(text)]
    ints = [v for v in ints if v < 1_000_000 and not (1900 < v < 2100)]
    return max(ints) if ints else None

# ── Trainable modules (all outside TGN, owned by numeric_optimizer) ────

class Projection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc   = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.fc(x)))


class NumericHead(nn.Module):
    """MLP that maps frozen-LLM prefix output → predicted log(magnitude)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (N,) or scalar

# ── Frozen LLM helpers ──────────────────────────────────────────────────

def _word_embeddings(llm: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    emb = getattr(llm, "embeddings", None)
    if emb is not None and hasattr(emb, "word_embeddings"):
        with torch.no_grad():
            return emb.word_embeddings(input_ids)
    with torch.no_grad():
        return llm(input_ids=input_ids).last_hidden_state.detach()


def precompute_text_embs(texts, llm, tokenizer, device) -> list[torch.Tensor]:
    cached = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=64).to(device)
        we  = _word_embeddings(llm, enc["input_ids"])
        cached.append(we.squeeze(0)[1:])   # drop CLS, keep text + SEP
    return cached


def prefix_forward(llm, proj_embs, cached_text_embs, device) -> torch.Tensor:
    """
    Inject proj_embs[i] as the prefix token (position 0) for passage i.

    Gradient path:
        loss → head → prefix_out[:,0,:] → frozen LLM attention
             → inputs_embeds[:,0,:] = proj_embs → Projection → mem_to_emb
    """
    N, D    = proj_embs.shape
    max_len = max(t.shape[0] for t in cached_text_embs)

    padding        = torch.zeros(N, max_len, D, device=device)
    attention_mask = torch.zeros(N, 1 + max_len, device=device)
    attention_mask[:, 0] = 1.0

    for i, te in enumerate(cached_text_embs):
        sl = te.shape[0]
        padding[i, :sl]           = te.to(device)
        attention_mask[i, 1:1+sl] = 1.0

    inputs_embeds = torch.cat([proj_embs.unsqueeze(1), padding], dim=1)
    out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0, :]   # (N, D) — prefix position

# ── PSRO simulation ─────────────────────────────────────────────────────

def make_psro_events(
    magnitudes: list[float], n_events: int, step: int
) -> list[tuple[str, str, float, float, float, float]]:
    """Fake judged-pair events for tgn.train_step.
    Similar log-magnitude → coherent (+), very different → dissonant (-)."""
    N   = len(magnitudes)
    rng = np.random.default_rng(step)
    events = []
    for i in range(n_events):
        src, dst = rng.choice(N, size=2, replace=False)
        gap   = abs(math.log(magnitudes[src] + 1e-8) - math.log(magnitudes[dst] + 1e-8))
        sign  = 1.0 if gap < 1.0 else -1.0
        y     = sign * max(0.0, 1.0 - gap / 5.0)
        events.append((f"p{src}", f"p{dst}", sign, float(step * N + i), 0.5, y))
    return events

# ── Numeric loss helper ─────────────────────────────────────────────────

def numeric_forward(tgn, proj, head, llm, cached_text_embs, node_ids, mags, device):
    """One numeric supervised forward pass. Returns (loss, prefix_out)."""
    memories   = tgn.memory.get_batch(node_ids).to(device)   # detached
    mem_embs   = tgn.mem_to_emb(memories)                    # grad through weights
    proj_embs  = proj(mem_embs)
    prefix_out = prefix_forward(llm, proj_embs, cached_text_embs, device)

    log_targets = torch.tensor(
        [math.log(m + 1e-8) for m in mags], dtype=torch.float32, device=device
    )
    pred_logs = head(prefix_out)   # (N,)
    loss = F.mse_loss(pred_logs, log_targets)
    return loss, pred_logs.detach()

# ── Main ────────────────────────────────────────────────────────────────

def run(n_steps: int, do_plot: bool) -> None:
    texts     = [p[0] for p in PASSAGES]
    true_mags = [p[1] for p in PASSAGES]
    N         = len(PASSAGES)

    regex_mags = [extract_magnitude(t) for t in texts]
    train_mags = [rm if rm is not None else tm for rm, tm in zip(regex_mags, true_mags)]

    print("\nTGN + Frozen LLM — Numeric Supervised (Dummy)")
    print(f"  passages: {N}  |  steps: {n_steps}  |  PSRO_LR={PSRO_LR}  NUMERIC_LR={NUMERIC_LR}\n")

    print(f"  {'Text (truncated)':<55} {'Extracted':>10}  {'True':>8}")
    print(f"  {'─'*55} {'─'*10}  {'─'*8}")
    for t, rm, tm in zip(texts, regex_mags, true_mags):
        flag = "✓" if rm == tm else "!"
        print(f"  {flag} {t[:54]:<54} {str(rm):>10}  {tm:>8}")

    device = torch.device("cpu")

    print("\nLoading frozen LLM …")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm       = AutoModel.from_pretrained(LLM_NAME).to(device)
    for p in llm.parameters():
        p.requires_grad = False

    cached_text_embs = precompute_text_embs(texts, llm, tokenizer, device)

    # TGN: emb_dim=384 so mem_to_emb bridges (64 → 384)
    tgn  = TGNModule(emb_dim=EMB_DIM, memory_dim=MEM_DIM, time_dim=TIME_DIM).to(device)
    proj = Projection(EMB_DIM).to(device)
    head = NumericHead(EMB_DIM).to(device)

    # PSRO optimizer: link_head + GRU + msg_encoder (NOT mem_to_emb)
    psro_optimizer = torch.optim.Adam(
        [p for n, p in tgn.named_parameters() if "mem_to_emb" not in n],
        lr=PSRO_LR,
    )
    # Numeric optimizer: mem_to_emb + proj + head
    numeric_optimizer = torch.optim.Adam(
        list(tgn.mem_to_emb.parameters())
        + list(proj.parameters())
        + list(head.parameters()),
        lr=NUMERIC_LR,
    )

    node_ids = [f"p{i}" for i in range(N)]

    # ── Gradient verification ────────────────────────────────────────────
    # Seed memories with one PSRO batch first — mem_to_emb grad is zero
    # when all memories are zeros (∂(W·0)/∂W = 0).
    print("\nSeeding TGN memories with one PSRO batch for gradient check …")
    seed_events = make_psro_events(train_mags, N_EVENTS, step=9999)
    psro_optimizer.zero_grad()
    seed_loss = tgn.train_step(seed_events)
    if seed_loss.requires_grad:
        seed_loss.backward()
        psro_optimizer.step()
    tgn.detach_all_memory()

    print("Verifying gradient flow to tgn.mem_to_emb …")
    loss0, _ = numeric_forward(tgn, proj, head, llm, cached_text_embs, node_ids, train_mags, device)
    numeric_optimizer.zero_grad()
    loss0.backward()
    grad_norm = tgn.mem_to_emb.weight.grad.norm().item()
    print(f"  ∂loss/∂mem_to_emb.weight  norm = {grad_norm:.6f}  "
          f"{'✓ gradient flows' if grad_norm > 1e-8 else '✗ ZERO — broken!'}")
    numeric_optimizer.step()

    # Baseline MSE (before training)
    with torch.no_grad():
        mse_before, _ = numeric_forward(
            tgn, proj, head, llm, cached_text_embs, node_ids, train_mags, device
        )
    print(f"  Baseline MSE (log-magnitude): {mse_before.item():.4f}\n")

    # ── Alternating training loop ────────────────────────────────────────
    print(f"  {'Step':>5}  {'PSRO link-loss':>16}  {'Numeric MSE':>12}")
    print(f"  {'─'*5}  {'─'*16}  {'─'*12}")

    loss_history: list[float] = []

    for step in range(n_steps):
        # A) PSRO-sim step
        events = make_psro_events(train_mags, N_EVENTS, step)
        psro_optimizer.zero_grad()
        link_loss = tgn.train_step(events)
        if link_loss.requires_grad:
            link_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in tgn.named_parameters() if "mem_to_emb" not in n],
                max_norm=1.0,
            )
            psro_optimizer.step()
        tgn.detach_all_memory()

        # B) Numeric supervised step
        num_loss, _ = numeric_forward(
            tgn, proj, head, llm, cached_text_embs, node_ids, train_mags, device
        )
        numeric_optimizer.zero_grad()
        num_loss.backward()
        numeric_optimizer.step()

        loss_history.append(num_loss.item())

        if (step + 1) % LOG_EVERY == 0:
            print(f"  {step+1:>5}  {link_loss.item():>16.5f}  {num_loss.item():>12.5f}")

    # ── Results ─────────────────────────────────────────────────────────
    with torch.no_grad():
        mse_after, pred_logs = numeric_forward(
            tgn, proj, head, llm, cached_text_embs, node_ids, train_mags, device
        )
    delta = mse_before.item() - mse_after.item()

    print(f"\n{'─'*60}")
    print(f"  MSE before : {mse_before.item():.4f}")
    print(f"  MSE after  : {mse_after.item():.4f}  "
          f"({'↓ improved' if delta > 0 else '↑ worse'}  Δ={abs(delta):.4f})")

    print(f"\n  {'Passage (truncated)':<50}  {'Pred':>8}  {'True':>8}  {'Err%':>6}")
    print(f"  {'─'*50}  {'─'*8}  {'─'*8}  {'─'*6}")
    for i, (text, tm) in enumerate(zip(texts, true_mags)):
        pred_val = math.exp(pred_logs[i].item())
        err_pct  = abs(pred_val - tm) / (tm + 1e-8) * 100.0
        print(f"  {text[:50]:<50}  {pred_val:>8.4f}  {tm:>8.4f}  {err_pct:>5.1f}%")
    print(f"{'─'*60}")

    if do_plot:
        _plot(loss_history)


def _plot(losses: list[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot")
        return
    out = Path("scripts/experiments/tgn_supervised_results")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color="steelblue", lw=1.5)
    plt.xlabel("Step")
    plt.ylabel("Numeric MSE (log-magnitude)")
    plt.title("TGN + Frozen LLM — Numeric Supervised\n(Alternating PSRO-sim + numeric step)")
    plt.tight_layout()
    path = out / "tgn_numeric_loss.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Loss curve → {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--plot",  action="store_true")
    args = p.parse_args()
    run(args.steps, args.plot)


if __name__ == "__main__":
    main()
