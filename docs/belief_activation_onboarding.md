# Belief Activation → Inference: Onboarding

How the belief-activation system works, end to end, from a belief graph to an
`/infer` answer — and where the TGN fits. Read top to bottom.

> **Repos.** The `multi_agent` package lives in this repo (`src/multi_agent/…`)
> and is vendored into the backend at
> `dyssonance-backend/research_deployment/belief_activation/`. The serving path
> (`/infer`, activation orchestration) lives in **dyssonance-backend**
> (`perseverate_api/…`, `research_deployment/pi_infer/…`). File:line refs below
> name the repo when it isn't this one.

---

## 0. One-sentence mental model

> **Belief activation is a *training* step that learns a `query → which beliefs
> matter` policy (the Router); `/infer` is a separate *inference* step that uses
> that Router only to pick the *seed* beliefs reasoning starts from.** No Router
> snapshot ⇒ `/infer` silently falls back to the vector index.

```
seeds → sync(embed+novelty) → ingest → coherence → PI hypotheses → edges
        → ★ BELIEF ACTIVATION (train) ★ → canonicalization      [enrich]
                       │ publishes
                       ▼
                 Router snapshot (Redis, HMAC-signed)
                       │ consumed by
                       ▼
     /infer:  seed selection (Router=priority 1) → beam search → select → synthesize → cited_beliefs
```

---

## 1. The belief graph

`graph.py` (this repo). Nodes = beliefs (each an embedding); edges are **signed**:

- `+w` **coherence** (agree / entail), `−w` **contradiction** (disagree).

The signed structure is what makes this more than vector search. Each node also
carries a **representation** the policy reads — `_z` (Bayesian mode) or projected
TGN memory (TGN mode).

---

## 2. The activation loop (PSRO)

Training = a population of policy "agents" learning to score `(query, candidate)`
belief pairs. `Trainer.step` → `PSROLoop.step` (`psro.py:437`).

**Agents & roles** (`agent.py`, `utils/helpers.py:role_sign`):

| Role | Sign | Trains on |
|---|---|---|
| `coherence` | +1 | `+y` (judge says "agree") |
| `contradiction` | −1 | `−y` (judge says "disagree") |
| `semantic` (`CosineActor`) | 0 | nothing — fixed cosine baseline |

**Per batch** (new beliefs = queries; pool = existing nodes + batch):

1. **Propose** — each agent scores its pairs, samples top-k (softmax).
2. **Judge** — resolve each pair via **cache → `graph.impute` → LLM/static judge**
   until `judge_budget_per_batch` is hit; the rest are skipped.
3. **Reward** — **only judge-revealed** pairs feed `_reward`
   (`reward_a = mean(sign_a · y)`). *Never* train on the graph's own imputations.
4. **Backward** — REINFORCE with a leave-out-self baseline; one Adam step/agent.
5. **Meta-reward** — `σ` (mixture over strategies) moves by multiplicative weights
   on **surprisal** `= |y − graph.field(q,c)|` (how much the judge surprised the
   graph). `_meta_reward`.
6. **Extend** — judged edges wired in; skipped pairs **re-imputed** in batch
   (`_impute_after_judge:599` → `graph.impute_batch`, the batched hot path).

**Zero-sum heart:** the same `y` rewards coherence and punishes contradiction;
they compete over which signed evidence the judge surfaces, and `σ` credits the
more *informative* (higher-surprisal) strategy.

---

## 3. Link prediction: **Bayesian posterior** vs **TGN** (the substrate)

`graph.impute` (step 2) and `graph.field` (step 5) are computed one of two ways.

### Bayesian baseline (default, `_tgn is None`) — `graph.py:_prior(294)`
Closed-form Gaussian posterior over edge `y_qc`, each 2-hop path `q→k→c` a noisy
observation `y_kc` weighted by `w_qk`:

```
total_precision = Σ_k w_qk²/σ_obs²  + 1/σ_prior²
posterior_mean  = (Σ_k w_qk·y_kc/σ_obs²) / total_precision
```

- `impute` returns the mean **only if `data_precision ≥ confidence_floor (0.25)`**,
  else `None` → **defer to the judge** (calibrated abstention).
- `field` = same mean, always defined.

### TGN substrate (`use_tgn=True` → *replaces* the Bayesian machinery) — `tgn.py`
Each node has a learned **memory** evolved over the **order** signed edges arrive
(edge counter, not wall-clock). PyTorch Geometric backend:

- `TimeEncoder` → `TemporalMessageEncoder` → `MemoryUpdater` (GRUCell) update
  memory per event; `TemporalNeighborhoodAggregator` = PyG `TransformerConv` over
  an authoritative `edge_index`; `link_head` (MLP) → signed score in `[-1,1]`.
- `train_step` (inside PSRO step): predict from **pre-event** memory, MSE vs `y`,
  then update memory. `impute`/`field` delegate to `predict_link` / **batched
  `predict_links`**. Cold nodes: `tgn_cold_start="raw_fallback"` (raw cosine) vs
  `"pure"`.

### The distinction

| | Bayesian posterior | TGN |
|---|---|---|
| Learned from judge? | No (closed-form) | **Yes**, per session |
| Evidence | local 2-hop signed paths | per-node memory over **all** its events + neighbourhood attention |
| Uncertainty / abstention | **Yes** (`None` below precision floor) | **No** native; `|score|<threshold` proxy |
| Order / time aware | No | **Yes** |
| Cost / knobs | ~free, none | GRU+MLP+conv; `memory_dim`, `predict_threshold`, `cold_start`, `tgn_lr`, `rep_align_weight` |
| Interpretability | high | low |
| Sparse graphs | abstains (no path) | can still predict |

### Why TGN *could* help
1. Higher-order structure (a node's accumulated role, beyond 2 hops).
2. Order/retraction dynamics (later contradictions shift memory).
3. Nonlinear coherence the Gaussian mean can't fit.
4. Predicts where Bayesian abstains → saves judge budget on sparse graphs.
5. Scales — and after the PyG rewrite it's **fast** (batched `impute_batch`,
   ~32× on the impute path, numerically identical to per-pair).

### Issues / where TGN currently does **not** help (be honest with candidates)
1. **Lost calibrated abstention** — TGN over-commits instead of deferring to the
   judge ⇒ judge budget spent less intelligently.
2. **Cold-start degeneracy** — `"pure"`: `mem_to_emb(0)=bias` ⇒ all cold nodes
   collapse to one vector. `raw_fallback` mitigates but cold pairs = raw cosine.
3. **Meta-loop starvation (measured)** — with a cosine judge, `field`(raw_fallback)
   = raw cosine = the judge's `y` ⇒ `surprisal ≡ 0` ⇒ `σ` frozen. A real judge
   (Gemini) breaks the identity (`surprisal 0 → 0.40` in the notebook).
4. **Quality benefit UNPROVEN** — on the SOC2 demo, `auc_signed` was undefined
   (no contradiction labels) and TGN query-ranking was *worse* than cosine on a
   tiny graph. We made TGN fast; we have **not** shown it improves answers — that
   is the open **AMABench A/B**.
5. **Representation drift** — TGN reps drift from raw embedding space
   (`repr_divergence` ~0.44→0.61); `tgn_rep_align_weight` counters it, not enough
   on short runs.

**Takeaway:** TGN = powerful, scalable, order-aware, now fast. Bayesian =
interpretable, calibrated, zero-cost, strong baseline. *Whether TGN improves
answer quality is still open.*

---

## 4. Training → Router snapshot

`Trainer.to_snapshot(session_id)` (`runner.py`) → two artifacts:

1. **`RouterSnapshot`** (JSON): `multi_agent_config`, **`graph_z`** (belief
   representations frozen at end of training — TGN-projected when TGN is on),
   **`sigma`**, `roles`, `bid_to_text`. (+ full graph/history for *resume*.)
2. **weights blob** (torch): population `state_dict` (+ Adam momentum; **+ TGN
   weights & per-node memory** when attached).

Published via `SnapshotStore.save` (`snapshot.py`) to **Redis**, **HMAC-signed**
with `BELIEF_ACTIVATION_SNAPSHOT_HMAC_KEY` — an **integrity signature**, re-verified
on load (`RouterVerifyFailed` ⇒ treated as "no Router"). Backend orchestration:
`run_activation` (backend `belief_activation.py:699`), reached from enrich Step 5
via `_run_belief_activation` (backend `worker.py:3542`) → `_run_activation_inline`.

**The Router** (`router.py`, read-only) reconstructs from the snapshot
(`from_redis:174` → `from_snapshot:127`) and exposes:

- **`rank(query_emb, k)`** (`router.py:94`): each agent scores the query vs the
  frozen candidate beliefs; scores **fused weighted by `sigma`**
  (`fusion="sigma"`); returns **top-k `(belief_id, score)`** = the seeds.

> The snapshot freezes **both** the policy *and* the belief representations, so the
> Router scores a *new* query against a *fixed* candidate space — no graph/TGN
> recompute at query time.

---

## 5. `/infer` end-to-end

`run_inference` (backend `worker.py:4188`) → `_run_inference` (`worker.py:4026`) →
`pi_infer.infer` (backend `research_deployment/pi_infer/pi_infer/__init__.py:312`).

1. Build `PiInferConfig` (tier, `token_budget`, `use_knapsack`, …); optional
   conversation **query rewrite**.
2. **Load Router** (`_load_activation_router:4106`).
3. **`pi_infer.infer`**:
   1. embed query.
   2. **Seed selection cascade** (`seed_selection.py:find_seed_nodes:316`) — *the
      only place belief activation touches inference*:
      - **(1) Router** — if `enable_router_seeds and router`: `router.rank(...)` →
        resolve `bid→text→graph node` → top-k. Success short-circuits.
      - **(2) Vector index** — else if `index`: `index.search_vector`.
        ⚠️ **If the index is present but returns 0 hits, return empty — do NOT
        fall through to MMR.**
      - **(3) MMR** — reachable **only when `index is None`** (offline). Dead on
        Cloud Run.
   3. **Beam search** from seeds (`query_search.py`): path-integral over signed
      edges (query relevance + interior coherence).
   4. **Selection within token budget**: knapsack (`use_knapsack=1`) or path-walk.
   5. **Synthesis + dissonance typing** → conflicts, dissonance score.
4. Return `cited_beliefs` (+ conflicts, `dissonance_score`, stats, timing).

### The boundary that matters
Belief activation's **entire leverage on `/infer` is the seed set** — *where* beam
search starts. Beam search, selection, and synthesis are identical regardless of
seed source.

```
better seeds → better beams → better cited_beliefs
```

So "does TGN improve quality?" reduces to "does a TGN-trained Router pick better
*seeds*?" — measurable on **AMABench** (it grounds answers on `/infer`'s
`cited_beliefs`). That A/B is the open task (currently blocked on the Qwen/Llama
grounding-model credentials).

---

## File:line map

**This repo (`src/multi_agent/`):**
- `psro.py:437` `PSROLoop.step` · `:599` `_impute_after_judge`
- `graph.py:294` `_prior` · `:320` `impute` · `:353` `field` · `impute_batch`
- `tgn.py` `TGNModule` (`predict_link`, `predict_links`, `train_step`)
- `runner.py` `Trainer.step`, `run`, `to_snapshot`, `rank`
- `router.py:94` `rank` · `:174` `from_redis` · `:127` `from_snapshot`
- `snapshot.py` `RouterSnapshot`, `SnapshotStore`
- `config.py` `MultiAgentConfig` (`use_tgn`, `tgn_*`)
- `agent.py` `AttentionAgent`, `AgentPopulation` · `judge.py` judges

**dyssonance-backend:**
- `perseverate_api/worker.py:3542` `_run_belief_activation` · `:3666`
  `_run_activation_inline` · `:4026` `_run_inference` · `:4106`
  `_load_activation_router` · `:4158` `_build_infer_config` · `:4188` `run_inference`
- `perseverate_api/belief_activation.py:699` `run_activation` · `:553` config build
- `research_deployment/pi_infer/pi_infer/__init__.py:312` `infer`
- `…/pi_infer/seed_selection.py:316` `find_seed_nodes` (cascade) · `:123` router ·
  `:177` index · `:245` MMR · `config.py` `PiInferConfig`

## Glossary

- **σ (sigma)** — learned mixture over strategies (coherence/contradiction/
  semantic); fuses per-agent scores at rank time.
- **surprisal** — `|y_judged − field|`; drives the σ meta-update.
- **impute vs field** — `impute` may abstain (`None` → judge); `field` is always
  defined (meta-loop needs a number).
- **Router** — the trained activation policy + frozen belief reps, exposed as
  `query → top-k seed beliefs`.
- **seed** — the belief node(s) `/infer`'s beam search starts from.

## Hands-on
- `notebooks/belief_activation_tgn.ipynb` — runnable SOC2 walkthrough: baseline vs
  TGN, PyG batching benchmark (~32×), Gemini-judge A/B (surprisal revival).
- Programmatic: `Trainer(MultiAgentConfig(...), judge)` then `.step(batch)`; toggle
  `use_tgn=True`.
