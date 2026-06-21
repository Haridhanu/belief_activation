"""Generator for notebooks/belief_activation_tgn.ipynb.

Run:  uv run python scripts/_gen_ba_tgn_notebook.py
Builds a clean, fully-runnable notebook that exercises the belief-activation
PSRO loop with the TGN integration on vs off, on the **SOC 2 / STATE-Bench
task-1** example (real belief text, real sentence-transformer embeddings,
deterministic offline cosine judge).
"""

from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(
    """
# Belief activation + TGN — SOC 2 walkthrough

A self-contained, **offline** harness for the belief-activation component in
`src/multi_agent/`, focused on the **Temporal Graph Network (TGN) integration**,
run on the **SOC 2 / STATE-Bench task-1** example (the Mariana / SOC 2
compliance conversation used in
`dyssonance-backend/notebooks/enrich_async_walkthrough.ipynb`, Step 0).

Real belief text → real sentence-transformer embeddings → the PSRO loop with a
deterministic **cosine judge** (no Redis, no Gemini, no GPU). We:

1. run the loop with `use_tgn=False` (Bayesian graph baseline) and `use_tgn=True`
   (TGN replaces the graph's link prediction),
2. compare per-step diagnostics — signed AUC of link prediction, TGN loss,
   meta-loop surprisal, imputation rate, and the strategy mixture σ,
3. visualise the belief graph and route the real SOC 2 query
   *"What work should be assigned to Mariana for completing SOC 2 compliance?"*
   through each trained role.

> **Judge caveat.** The offline `CosineJudge` returns embedding cosine — a
> *semantic-similarity* proxy, not entailment. Sentence-transformer vectors of
> contradictory statements still have positive cosine, so the **contradiction**
> arm gets little signal here. For true coherence/contradiction use a real NLI /
> `LocalLLMJudge` (see `src/multi_agent/judge.py`). The TGN diagnostics below
> are still meaningful — they measure recovery of the cosine signal.
"""
)

md("## Setup")

code(
    """
import sys, pathlib
ROOT = pathlib.Path.cwd()
if (ROOT / "notebooks").is_dir() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multi_agent.config import MultiAgentConfig
from multi_agent.runner import Trainer
from multi_agent.benchmarks import Batch
from multi_agent.utils.notebook import (
    make_cosine_judge,
    plot_belief_graph,
    plot_sigma,
    top_edges,
)

print("imports OK")
"""
)

md(
    """
## SOC 2 beliefs (STATE-Bench task 1)

Twelve belief statements from three people (Jennifer, Harsha, Carlos) about SOC 2
readiness and who owns it. There's real structure: several beliefs reinforce
*"Mariana owns SOC 2 / incident response"*, a few are in tension (*"don't mark
controls complete until reviewed"* vs *"customer expects readiness confirmed"*),
and one is off-topic (the baby shower) — a natural neutral.

We stream them as four chronological batches so the loop runs several steps and
the TGN memory evolves over edge arrival order.
"""
)

code(
    """
SOC2_BELIEFS = [
    "[Jennifer] The acting Head of Security is Mariana.",
    "[Jennifer] The Head of Security is responsible for being on call for security incidents.",
    "[Jennifer] Please help me draft the security policy knowing Mariana is the Head of Security.",
    "[Jennifer] I am planning a baby shower for Mariana next Thursday.",
    "[Harsha] We need Mariana's sign-off on the SOC 2 controls before the enterprise launch.",
    "[Harsha] The SOC 2 launch checklist still has Security Owner Review marked as pending.",
    "[Harsha] Mariana is the only person currently mapped as accountable for the incident response policy.",
    "[Carlos] The enterprise customer expects SOC 2 readiness to be confirmed before we move forward.",
    "[Carlos] We told the customer we are targeting the end of the month for the compliance package.",
    "[Harsha] We should not mark the SOC 2 controls as complete until Security has reviewed the escalation path.",
    "[Carlos] I do not see a backup security approver listed in the launch plan.",
    "[Carlos] The launch plan expects Mariana will attend the final SOC 2 readiness review.",
]
QUERY = "What work should be assigned to Mariana for completing SOC 2 compliance?"
N_BATCHES = 4

print(f"{len(SOC2_BELIEFS)} beliefs, streamed as {N_BATCHES} batches")
print(f"query: {QUERY!r}")
"""
)

md(
    """
## Embed with a sentence-transformer

`all-MiniLM-L6-v2` (384-dim, normalized). Downloaded once from the HF Hub, then
cached. Set `EMB_MODEL` to a 768-dim model to match production embedding width.
"""
)

code(
    """
from sentence_transformers import SentenceTransformer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_st = SentenceTransformer(EMB_MODEL)

def embed(texts):
    return np.asarray(
        _st.encode(list(texts), normalize_embeddings=True), dtype=np.float32
    )

emb_all = embed(SOC2_BELIEFS)
query_emb = embed([QUERY])[0]
EMB_DIM = emb_all.shape[1]
print(f"embedded {emb_all.shape[0]} beliefs, dim={EMB_DIM}")

# Build chronological batches of Batch(ids, embs, texts)
chunks = np.array_split(np.arange(len(SOC2_BELIEFS)), N_BATCHES)
batches = [
    Batch(
        ids=[f"b{i}" for i in idx],
        embs=emb_all[idx],
        texts=[SOC2_BELIEFS[i] for i in idx],
    )
    for idx in chunks if len(idx) > 0
]
for bi, b in enumerate(batches):
    print(f"  batch {bi}: {len(b.ids)} beliefs")
"""
)

md(
    """
## Run helper

We drive the `Trainer` directly (it auto-wires a `TGNModule` when
`config.use_tgn=True`) so we keep the trained policies for query routing.
A non-zero `judge_budget_per_batch` is required or the judge never fires.
"""
)

code(
    """
JUDGE_BUDGET = 40     # pairs sent to the (free, offline) cosine judge per batch
NUM_AGENTS = 2        # coherence + contradiction (cosine baseline added automatically)
K = 4                 # top-k proposals per query
SEED = 0

def make_config(**overrides):
    base = dict(emb_dim=EMB_DIM, device="cpu", num_agents=NUM_AGENTS, k=K,
                judge_budget_per_batch=JUDGE_BUDGET, seed=SEED)
    base.update(overrides)   # let callers override any field, incl. judge_budget
    return MultiAgentConfig(**base)

def run(tag, judge=None, **overrides):
    judge = judge if judge is not None else make_cosine_judge(batches)
    trainer = Trainer(make_config(**overrides), judge)
    for b in batches:
        trainer.step(b)
    h = trainer.history
    sigma = dict(trainer.loop.sigma)
    print(f"[{tag}] steps={len(h)} nodes={len(trainer.graph.get_nodes())} "
          f"sigma={ {k: round(v,3) for k,v in sigma.items()} }")
    return {"tag": tag, "trainer": trainer, "history": h, "sigma": sigma}

def step_frame(res):
    h = res["history"]
    return pd.DataFrame({
        "step":           [s.step for s in h],
        "judged":         [s.judged for s in h],
        "imputed":        [s.imputed for s in h],
        "imputation_rate":[round(s.imputation_rate, 3) for s in h],
        "auc_signed":     [s.auc_signed for s in h],
        "tgn_loss":       [round(s.tgn_loss, 4) for s in h],
        "mean_surprisal": [round(s.mean_surprisal, 4) for s in h],
        "repr_divergence":[round(s.repr_divergence, 4) for s in h],
        "winner":         [s.winner_id for s in h],
    })

def print_top_edges(res, n=6):
    g, texts = res["trainer"].graph, res["trainer"].node_texts
    coh, dis = top_edges(g, texts, n=n)
    print(f"Top coherent edges ({res['tag']}):")
    for a, b, w in coh:
        print(f"  {w:+.3f}  {a}  ~  {b}")
    print(f"Top contradictory edges ({res['tag']}):")
    for a, b, w in (dis or [["(none)", "", 0.0]]):
        print(f"  {w:+.3f}  {a}  ~  {b}" if a != "(none)" else "  (none)")
"""
)

md("## Baseline — `use_tgn=False` (Bayesian graph link prediction)")

code(
    """
res_base = run("baseline", use_tgn=False)
step_frame(res_base)
"""
)

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_belief_graph(res_base["trainer"].graph, res_base["trainer"].node_texts,
                  ax=axes[0], title="Baseline belief graph")
plot_sigma(res_base["history"], ax=axes[1])
axes[1].set_title("Baseline σ trajectory")
plt.tight_layout(); plt.show()

print_top_edges(res_base)
"""
)

md("## TGN — `use_tgn=True` (TGN replaces link prediction)")

code(
    """
res_tgn = run("tgn", use_tgn=True)
step_frame(res_tgn)
"""
)

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_belief_graph(res_tgn["trainer"].graph, res_tgn["trainer"].node_texts,
                  ax=axes[0], title="TGN belief graph")
plot_sigma(res_tgn["history"], ax=axes[1])
axes[1].set_title("TGN σ trajectory")
plt.tight_layout(); plt.show()

print_top_edges(res_tgn)
"""
)

md(
    """
## A/B comparison

- **`auc_signed`** — does link prediction recover the signed cosine ground truth?
  (0.5 = chance, 1.0 = perfect).
- **`tgn_loss`** — non-zero under TGN (link head training); always 0 in baseline.
- **`mean_surprisal`** — drives the meta-loop's σ update (`|y_judged − field|`).
  It collapses to ~0 under TGN here because the cosine cold-start fallback equals
  the cosine judge (see the analysis section) — an artifact of the offline judge.
- **`imputation_rate`** — fraction of scorable pairs resolved without a judge call.
"""
)

code(
    """
def agg(res):
    h = res["history"]
    aucs = [s.auc_signed for s in h if s.auc_signed is not None]
    return {
        "mean_auc_signed":  round(float(np.mean(aucs)), 3) if aucs else None,
        "final_auc_signed": aucs[-1] if aucs else None,
        "total_tgn_loss":   round(sum(s.tgn_loss for s in h), 4),
        "mean_surprisal":   round(float(np.mean([s.mean_surprisal for s in h])), 4),
        "mean_imputation":  round(float(np.mean([s.imputation_rate for s in h])), 3),
        "total_judged":     sum(s.judged for s in h),
        "total_imputed":    sum(s.imputed for s in h),
        **{f"sigma[{k}]": round(v, 3) for k, v in res["sigma"].items()},
    }

comparison = pd.DataFrame({res["tag"]: agg(res) for res in (res_base, res_tgn)})
comparison
"""
)

code(
    """
metrics = ["auc_signed", "tgn_loss", "mean_surprisal", "imputation_rate"]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, m in zip(axes.ravel(), metrics):
    for res, style in ((res_base, "o-"), (res_tgn, "s--")):
        df = step_frame(res)
        ax.plot(df["step"], df[m], style, label=res["tag"])
    ax.set_title(m); ax.set_xlabel("step"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""
)

md(
    """
## Real judge — Gemini 2.5 Flash (Vertex AI)

The cosine judge can't express contradiction and coincides with the TGN
cold-start field, which flatlines `mean_surprisal` and leaves `auc_signed`
undefined. Swap in **`LLMJudge` with `gemini-2.5-flash`** — a real
entailment/contradiction signal — and the diagnostics come alive:
`mean_surprisal > 0`, σ adapts, and negative labels make `auc_signed` computable.

**Auth.** No `GOOGLE_API_KEY` here, so this uses **Vertex AI via ADC**. If you
see `RefreshError: Reauthentication is needed`, refresh ADC by typing this in the
Claude Code prompt:

```
! gcloud auth application-default login
```

Scores are cached on disk (`~/.cache/dyssonance/llm_judge`), so re-runs are cheap.
The cell below degrades gracefully — if creds are missing/stale it prints the hint
and skips, so the rest of the notebook still runs.
"""
)

code(
    """
import os
# Route google-genai through Vertex AI using ADC (no API key in this env).
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "harsha-dev-481004")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from multi_agent.judge import LLMJudge

JUDGE_BUDGET_LLM = 20   # keep Gemini calls modest; disk cache makes re-runs free
GEMINI_MODEL = "gemini-2.5-flash"

res_base_llm = res_tgn_llm = None
try:
    # one shared judge so its disk cache is reused across both runs
    llm_judge = LLMJudge(model=GEMINI_MODEL, max_concurrency=4)
    res_base_llm = run("baseline+gemini", judge=llm_judge,
                       use_tgn=False, judge_budget_per_batch=JUDGE_BUDGET_LLM)
    res_tgn_llm = run("tgn+gemini", judge=llm_judge,
                      use_tgn=True, judge_budget_per_batch=JUDGE_BUDGET_LLM)
    print("\\nGemini judge stats:", llm_judge.stats)
except Exception as e:
    msg = str(e)
    print(f"[skipped Gemini judge run] {type(e).__name__}: {msg[:200]}")
    if "Reauthentication" in msg or "credentials" in msg.lower() or "auth" in msg.lower():
        print("-> ADC stale. In the Claude Code prompt run:")
        print("   ! gcloud auth application-default login")
"""
)

code(
    """
# Compare cosine-judge vs Gemini-judge (only if the Gemini run succeeded)
if res_tgn_llm is not None:
    rows = [res_base, res_tgn, res_base_llm, res_tgn_llm]
    table = pd.DataFrame({r["tag"]: agg(r) for r in rows})
    display(table)
    print("\\nStep diagnostics — tgn+gemini:")
    display(step_frame(res_tgn_llm))
else:
    print("Gemini run unavailable — see the hint above. "
          "Cosine-judge results stand; auc_signed/surprisal stay degenerate "
          "by the cosine↔fallback identity.")
"""
)

md(
    """
## Inference — route the SOC 2 query through each role

`Trainer.rank(query_emb)` scores every belief against the query through each
trained policy. The **coherence** arm should surface the beliefs that actually
define Mariana's SOC 2 work (sign-off, owner review pending, readiness review).
"""
)

code(
    """
def show_ranking(res, top=5):
    print(f"\\n=== {res['tag']} — top beliefs per role ===")
    ranking = res["trainer"].rank(query_emb, k=top)
    texts = res["trainer"].node_texts
    for agent_id, items in ranking.items():
        print(f"\\n[{agent_id}]")
        for nid, score in items:
            print(f"  {score:+.3f}  {texts.get(nid, nid)}")

show_ranking(res_base)
show_ranking(res_tgn)
"""
)

md(
    """
## PyG batching — link-prediction throughput

The TGN runs on **PyTorch Geometric**: an authoritative `edge_index` data index
plus a `TransformerConv` neighbourhood aggregator. The win is *batched* link
prediction — `graph.impute_batch(pairs)` runs **one** whole-graph conv pass +
**one** `link_head` forward over all pairs, instead of a forward per pair (the
old `_impute_after_judge` hot path). Below: per-pair vs batched on a trained
graph, with a parity check (identical gating ⇒ identical scores).
"""
)

code(
    """
import time, itertools
from multi_agent.runner import Trainer as _Trainer
from multi_agent.utils.notebook import make_synthetic_batches as _msb, make_cosine_judge as _mcj

# Train a larger synthetic graph so the throughput gap is visible.
_b = _msb(n_nodes=300, n_batches=15, n_topic_pairs=4, emb_dim=128, seed=1)
_tr = _Trainer(make_config(emb_dim=128, use_tgn=True, judge_budget_per_batch=60, k=8), _mcj(_b))
for _bb in _b:
    _tr.step(_bb)
_g = _tr.graph
_nodes = _g.get_nodes()
_pairs = list(itertools.islice(((a, b) for a in _nodes for b in _nodes if a != b), 5000))

_t0 = time.perf_counter(); _per = [_g.impute(a, b) for a, b in _pairs]; _t_per = time.perf_counter() - _t0
_t0 = time.perf_counter(); _bat = _g.impute_batch(_pairs);            _t_bat = time.perf_counter() - _t0

_mismatch = sum(
    1 for i, (a, b) in enumerate(_pairs)
    if (_per[i] is None) != (_bat[(a, b)] is None)
    or (_per[i] is not None and _bat[(a, b)] is not None and abs(_per[i] - _bat[(a, b)]) > 1e-5)
)
print(f"{len(_pairs)} pairs over {len(_nodes)} nodes, committed edges={len(_tr.tgn._committed_edges)}")
print(f"per-pair impute : {_t_per*1000:8.1f} ms")
print(f"batched impute  : {_t_bat*1000:8.1f} ms   speedup x{_t_per/_t_bat:.1f}")
print(f"result mismatches: {_mismatch} / {len(_pairs)}  (0 ⇒ batching is exact)")
"""
)

md(
    """
## Observations & analysis (verified against the code)

What this run shows on the SOC 2 example:

- **TGN trains and forms sensible edges.** `tgn_loss > 0` every step (the link
  head + GRU updater + message encoder are learning) and the top *coherent*
  edges are exactly the SOC-2 ownership cluster (Mariana = Head of Security,
  sign-off ↔ owner-review-pending, readiness review).

- **σ is frozen at uniform (0.333) under TGN; `mean_surprisal ≡ 0`.** Root cause
  (confirmed in `graph.py`): with `tgn_cold_start="raw_fallback"`, cold pairs
  take `Graph._tgn_raw_fallback_score`, which returns the **raw embedding
  cosine**. The offline `CosineJudge` *also* returns embedding cosine as the
  truth `y`. So for cold pairs `field(q,c) == y` exactly → surprisal
  `|y − field| ≡ 0` → the meta-reward (`psro._meta_reward`) is all-zero → the
  multiplicative-weights σ update never moves. This is an **identity artifact of
  pairing the cosine judge with the cosine cold-start fallback**, not a meta-loop
  ordering bug (the code already captures `field` *before* the post-judge extend,
  see `psro.py` line ~478). A real judge (next bullet) breaks the identity and
  the meta-loop comes alive. The Bayesian baseline avoids the identity because
  its 2-hop prior ≠ raw cosine, so its σ does drift.

- **`auc_signed` is undefined (None/NaN) here** because the cosine judge yields
  **no contradiction (negative) labels** — every cosine is ≥ 0, so `auc_n_neg=0`
  and the signed AUC can't be computed. The contradiction arm earns ~no reward.

- **`repr_divergence` grows under TGN** (≈0.44 → 0.61): the projected memory
  representations drift away from raw embeddings over steps. On this tiny example
  the TGN-backed query ranking is *worse* than the plain cosine baseline for the
  SOC-2 query (see the inference cell) — expected with so few events and a judge
  that can't supervise contradiction.

**To make the TGN diagnostics meaningful**, the *Real judge — Gemini 2.5 Flash*
section above swaps the cosine judge for `LLMJudge(model="gemini-2.5-flash")`,
which expresses entailment *and* contradiction. That (a) breaks the cosine↔field
identity so `mean_surprisal` and σ become informative, and (b) produces negative
labels so `auc_signed` is defined and TGN-vs-Bayesian link prediction can be
compared. Other knobs to explore: `tgn_cold_start` (`raw_fallback` vs `pure`),
`tgn_predict_threshold`, `tgn_rep_align_weight`, `tgn_memory_dim`,
`judge_budget_per_batch`.

Scratchpad below — example sweep over `tgn_predict_threshold` (note: with the
cosine judge, σ and AUC stay degenerate by the identity above — this sweep is a
template for when you plug in a real judge).
"""
)

code(
    """
for thr in (0.1, 0.2, 0.4):
    r = run(f"tgn thr={thr}", use_tgn=True, tgn_predict_threshold=thr)
    h = r["history"]
    aucs = [s.auc_signed for s in h if s.auc_signed is not None]
    print(f"  -> mean_auc={np.mean(aucs):.3f} "
          f"mean_surprisal={np.mean([s.mean_surprisal for s in h]):.4f} "
          f"total_imputed={sum(s.imputed for s in h)}")
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "belief_activation_tgn.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
