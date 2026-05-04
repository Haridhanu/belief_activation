# multi-agent

Correlated equilibrium over a belief graph via masked multi-agent attention. A small population of policies plays a per-batch tournament to propose edges (coherent / contradictory) over a streaming corpus of beliefs; an LLM judge reveals truth on a budgeted subset, and the graph imputes the rest. Over batches the meta-mixture σ reweights which strategy gets credit, the trained agents take one policy-gradient step each, and the graph itself learns to predict the judge.

## The loop

`Trainer.step(batch)` runs one PSRO step (`src/multi_agent/psro.py:321`). For each batch:

1. **Propose** — every agent scores `(query, candidate)` pairs over the current pool (existing graph nodes + the new batch) and samples top-k without replacement via softmax.
2. **Judge** — pairs are first looked up in a per-run score cache, then attempted via `Graph.impute`, then routed to the LLM judge until `judge_budget_per_batch` is hit. Anything past budget is skipped.
3. **Reward** — judged pairs only (never imputed ones) feed `_reward`.
4. **Backward** — actor parameters update.
5. **Meta-reward + mixture update** — the σ over strategies moves.
6. **Graph extend** — judged edges are wired in. Skipped pairs are then re-imputed because q now has a neighborhood.

Steps 1-5 are the **inner loop** (per arm). Steps 5-6 are the **outer loop** (over arms). One pass through `step` does both.

## Two zero-sum strategies

Each trainable agent has a role with a sign (`utils/helpers.py:role_sign`):

| Role | Sign | Reward |
|---|---|---|
| `coherence` | +1 | `+y` from the judge — entailment / agreement |
| `contradiction` | −1 | `−y` from the judge — disagreement |
| `semantic` | 0 | none — fixed cosine baseline |

Reward per query for agent *a* = `mean( sign_a * y_judged )` over *a*'s judged proposals (`psro.py:_reward`). Imputed pairs are excluded — the policy never trains on the graph's own predictions, only on judge-revealed truth.

This is zero-sum on the *judge signal*: the same `y` rewards the coherence arm and punishes the contradiction arm. They compete over which signed evidence the judge surfaces.

## Actor training

Each agent (`agent.py:AttentionAgent`) has a random feature mask + a small attention head (Q/K projections) and a residual MLP. Forward pass: `attn_logits + RESIDUAL_WEIGHT * residual_logits` over candidates; softmax over the result; multinomial top-k.

Update is REINFORCE with a **leave-out-self baseline** (`psro.py:_backward`):

```
advantage[q, a] = reward[q, a] − mean_{a' ∈ trainable} reward[q, a']
loss_a          = −Σ_q advantage[q, a] · log π_a(actions | q) / B
```

`grad_norm_clip = 0.5`, Adam(lr=`config.learning_rate`, default 5e-3), one optimizer cached per agent. The `CosineActor` semantic baseline has no parameters and gets a `None` optimizer (no-op).

The **outer loop** updates the mixture σ over strategies via multiplicative weights on a *surprisal* signal (`psro.py:_meta_reward`):

```
surprisal(q,c) = |y_judged(q,c) − graph.field(q,c)|
```

Surprisal for each judged pair is split across all agents that proposed it, then normalized to its max-abs and rolled into the meta-weights as `w_a *= exp(meta_lr · r_a)`. σ is `(1−ε)·w/Σw + ε/N` with `meta_eps=0.05` so no arm collapses to zero.

## Link prediction / imputation

The graph (`graph.py`) is the core inference engine. After each batch's judged edges are wired, every node's signed-attention representation `_z` is refreshed (`_update_representations`): a node moves toward the weighted, sign-flipped average of its neighbors' representations.

`Graph._prior(q, c)` — closed-form Gaussian posterior over the unobserved edge `y_qc`, treating each two-hop path `q → k → c` as one noisy observation `y_kc` weighted by `w_qk`:

```
total_precision = Σ_k w_qk² / σ_obs² + 1 / σ_prior²
posterior_mean  = ( Σ_k w_qk · y_kc / σ_obs² ) / total_precision
```

Two methods consume this:

- `Graph.impute(q, c)` — returns the clamped posterior mean **only if** `data_precision ≥ confidence_floor (=0.25)`. Otherwise `None` and the pair is deferred to the judge. This is what saves judge calls during step 2 of the loop.
- `Graph.field(q, c)` — same prior mean but always defined. Used by the meta-loop to compute surprisal *before* the judged edge is added (otherwise `field` would just regurgitate `y` and surprisal would always be zero).

Coherence/contradiction asymmetry comes for free: the prior aggregates **signed** weights, so a node strongly contradicting a coherent neighborhood gets `mu < 0` and vice versa.

## Temporal memory (TGN)

`src/multi_agent/tgn.py` adds an optional Temporal Graph Network module that gives each node a learned memory vector evolved over the *order* in which signed edges arrived. It's standalone (only depends on `torch`) and the `Graph` integration is fully gated — when no `TGNModule` is attached, `Graph` behaviour is bit-identical to the baseline.

**Components.** `TimeEncoder` (deterministic sinusoidal Δt → time_dim), `TemporalMessageEncoder` (`(src_mem, dst_mem, sign, time_enc, |w|) → memory_dim`), `NodeMemory` (per-node detached buffer, plain Python — no autograd state across steps), `MemoryUpdater` (GRUCell), `TemporalNeighborhoodAggregator` (multi-head attention over neighbor memories). `TGNModule` wires them together and adds `mem_to_emb` (memory_dim → emb_dim, used for blending) and `link_proj` (concat memories → sigmoid score).

**Wiring into Graph.** Two hooks, both no-ops when `graph._tgn is None`:

1. **`extend()`** — for each new signed edge `(a, b, w)` that survives dedup, calls `tgn.update(a, b, sign=±1/0, timestamp=_edge_count, edge_weight=w)`. The timestamp is a monotonic edge counter, not wall-clock — TGN cares about *order*, not real time.
2. **`_update_representations()`** — after the existing signed-attention update produces `updates`, the projected memory of each touched node is blended in:

   ```
   z_new = normalize( (1 − tgn_blend) · z_signed + tgn_blend · normalize(mem_to_emb(memory_v)) )
   ```

   `tgn_blend` defaults to `0.3`. The projection runs under `torch.no_grad()` so TGN gradients never flow into the policy graph.

**Config.** Four new fields on `MultiAgentConfig`, all default to safe no-ops:

| Field | Default | Effect |
|---|---|---|
| `use_tgn` | `False` | Caller flag — tells the runner whether to construct a `TGNModule` and attach it |
| `tgn_memory_dim` | `128` | Per-node memory vector size |
| `tgn_time_dim` | `32` | Time encoding dimensionality |
| `tgn_n_attn_heads` | `4` | Attention heads in the aggregator |

`use_tgn=True` in config but no `tgn` attached to `graph` is fine — the gates are checked on `graph._tgn`, not the config.

**Persistence.** `TGNModule.state_dict()` / `load_state_dict()` round-trip both the `nn.Module` parameters and the per-node memory dictionary, so checkpoints survive `torch.save` / `torch.load`. Only nodes with non-default memory are serialized.

**Lifecycle.** Memory persists across `extend()` calls within one `Graph`. Call `tgn.reset()` between independent questions/sessions.

**Usage.**

```python
from multi_agent.graph import Graph
from multi_agent.tgn import TGNModule

tgn = TGNModule(emb_dim=768)            # memory_dim/time_dim default to 128/32
graph = Graph(emb_dim=768)
graph._tgn = tgn                         # attaches the hooks
graph.extend(node_ids, embs, edges)      # extend() now updates tgn memory
                                         # and _z is blended with projected memory
```

## CPU / GPU

Two independent device knobs:

**Policy / agents** — `MultiAgentConfig.device` (default `"cpu"`). Single string. `AgentPopulation.__init__` does `agent.to(self.device)`. Embeddings are cast on the fly via `population.embeddings_to_device(np_array)`. Set `device="cuda"` or `"mps"` to move all policy parameters and per-step tensors there. The graph itself is numpy on CPU regardless — its scale is small (one graph per question in the FinanceBench notebook) and it doesn't benefit from GPU.

**NLI judge** — `NLIJudge(device=...)` (default: autodetect, `cuda → mps → cpu`). Half precision on accelerator, fp32 on CPU. Independent of `config.device` so you can train policies on CPU while the judge runs on GPU. See `judge.py:68-82`.

This split exists because the policy is tiny (a few projections + MLP, microseconds per pass) but the NLI judge is a 180 MB DeBERTa pulling its weight. On a Mac you typically want `config.device="cpu"` and let the judge auto-select MPS; on a Linux/CUDA box, `config.device="cuda"` is fine but rarely a speedup at FinanceBench scale.

## Quickstart

```bash
uv sync --group notebook                            # install + dev/notebook deps
uv run pytest                                       # 54 tests
uv run python scripts/fetch_financebench.py        # pre-pull the dataset (optional; load_financebench() does it lazily)
uv run jupyter lab demo.ipynb                       # the FinanceBench notebook
```

The notebook picks one prose-heavy FinanceBench question, extracts atomic claims with Gemini (cached on disk), trains the loop, draws the resulting graph, lists the strongest pairs, and routes the original question through each trained role for inference.

For a programmatic loop:

```python
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import NLIJudge
from multi_agent.runner import Trainer, JsonlEdgeLogger

trainer = Trainer(MultiAgentConfig(...), NLIJudge())
with JsonlEdgeLogger("runs/edges.jsonl") as sink:
    for batch in stream():           # caller-controlled, can be infinite
        sink.write(trainer.step(batch))
```
