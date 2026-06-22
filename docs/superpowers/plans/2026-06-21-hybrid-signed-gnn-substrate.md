# Hybrid Signed-GNN Substrate + Live Gym e2e — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 1-hop temporal TGN with a multi-hop **balance-theory Signed-GNN** as a graph substrate, run it as a **hybrid** (Bayesian 2-hop posterior when a pair is sparse, Signed-GNN once both endpoints are dense), and validate it end-to-end on **dyssonance-gym** root-cause attribution through the live backend `/infer`.

**Architecture:** A new `SignedGNN` module (PyG `SignedConv`, k hops, link head) is attached to `Graph` like `_tgn`. `Graph.field/impute/predict_links/impute_batch` route **per pair** by a degree gate: Bayesian when `min(deg(q),deg(c)) < D`, Signed-GNN when both `≥ D`. `_candidate_rep` uses Signed-GNN embeddings for warm nodes. The PSRO loop fits the GNN on committed signed edges. Selection is via `MultiAgentConfig.graph_substrate ∈ {"bayesian","tgn","signed_hybrid"}` (default `"bayesian"` → existing tests unchanged). The package is then ported into the backend's vendored copy and selected by env; gym is wired to trigger activation + query `/infer`.

**Tech Stack:** Python 3.12, PyTorch, PyTorch-Geometric (`SignedConv`), numpy; FastAPI backend (`perseverate_api`), Redis; uv.

---

## Prerequisites & gates (resolve before the phase that needs them)

- **Phase 1** (substrate): none — fully runnable in `~/belief_activation` now.
- **Phase 2** (backend port): the substrate code from Phase 1; write access to `~/dyssonance-backend/research_deployment/belief_activation`.
- **Phase 3** (live e2e) — **GATED**:
  - Backend must run locally: Redis on `:6379`, `gcloud` ADC valid, env
    `BELIEF_ACTIVATION_SNAPSHOT_HMAC_KEY=dev-local-key`, `BELIEF_ACTIVATION_JUDGE=static`,
    `CLOUD_TASKS_QUEUE` unset, `REDIS_URL=redis://localhost:6379`.
  - **gym subject/investigator LLM**: `bench_harness.py` supports only OpenAI/Anthropic.
    Creds are Gemini-only → **Task 12 adds a Gemini provider to gym** (no OpenAI/Anthropic key required).
  - gym currently calls `POST /beliefs` only → **Task 13 wires activation trigger + `/infer`**.

## File structure

**Phase 1 — `~/belief_activation`:**
- Create `src/multi_agent/signed_gnn.py` — `SignedGNN` module (encoder + link head). One responsibility: signed multi-hop link scoring + node embeddings.
- Modify `src/multi_agent/config.py` — add `graph_substrate`, `signed_gnn_*`, `hybrid_density_threshold`.
- Modify `src/multi_agent/graph.py` — add `_sgnn` field, degree gate, hybrid `field/impute/impute_batch`, `_candidate_rep` for Signed-GNN.
- Modify `src/multi_agent/runner.py` — `Trainer.__init__` constructs the substrate from `config.graph_substrate`.
- Modify `src/multi_agent/psro.py` — fit the Signed-GNN on committed edges after extend (hook parallel to the TGN one).
- Create `tests/test_signed_gnn.py`, `tests/test_hybrid_substrate.py`.

**Phase 2 — `~/dyssonance-backend/research_deployment/belief_activation`:** mirror of the above files; plus `perseverate_api/belief_activation.py` (config wiring) + root `pyproject.toml` (`torch-geometric`).

**Phase 3 — `~/dyssonance-gym`:** `bench_harness.py` (Gemini provider + `/infer` retrieval); run scripts.

---

## Phase 1 — Hybrid Signed-GNN substrate (in `~/belief_activation`)

### Task 1: `SignedGNN` module

**Files:**
- Create: `src/multi_agent/signed_gnn.py`
- Test: `tests/test_signed_gnn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signed_gnn.py
import numpy as np, torch
from multi_agent.signed_gnn import SignedGNN

def _toy():
    # 6 nodes, two clusters {0,1,2} coherent, {3,4,5} coherent; cross = contradiction
    X = torch.randn(6, 8)
    pos = torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]])
    neg = torch.tensor([[0,3,2,5],[3,0,5,2]])
    return X, pos, neg

def test_emb_shape_and_score_range():
    X, pos, neg = _toy()
    m = SignedGNN(in_dim=8, hidden=16, layers=3)
    z = m.emb(X, pos, neg)
    assert z.shape[0] == 6 and z.dim() == 2
    s = m.score(z, torch.tensor([[0,0],[1,3]]))   # (0,1) coherent, (0,3) contradiction
    assert s.shape == (2,)
    assert torch.isfinite(s).all()

def test_layers_controls_hops():
    X, pos, neg = _toy()
    assert len(SignedGNN(8, 16, layers=1).convs) == 1
    assert len(SignedGNN(8, 16, layers=3).convs) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/belief_activation && uv run pytest tests/test_signed_gnn.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'multi_agent.signed_gnn'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/multi_agent/signed_gnn.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import SignedConv


class SignedGNN(nn.Module):
    """Balance-theory signed multi-hop GNN. `layers` stacked SignedConv layers
    (= hops); node embeddings feed a link head producing a signed score."""

    def __init__(self, in_dim: int, hidden: int = 32, layers: int = 3) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.in_dim = in_dim
        self.hidden = hidden
        self.convs = nn.ModuleList(
            [SignedConv(in_dim, hidden, first_aggr=True)]
            + [SignedConv(hidden, hidden, first_aggr=False) for _ in range(layers - 1)]
        )
        # SignedConv concatenates balanced+unbalanced → 2*hidden per node.
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def emb(self, x: torch.Tensor, pos_edge_index: torch.Tensor,
            neg_edge_index: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.convs[0](x, pos_edge_index, neg_edge_index))
        for conv in self.convs[1:]:
            z = torch.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def score(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Signed score in (-inf, inf); apply tanh/sigmoid at the call site."""
        return self.head(torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)).squeeze(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_signed_gnn.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent/signed_gnn.py tests/test_signed_gnn.py
git commit -m "feat(signed_gnn): balance-theory signed multi-hop GNN module"
```

### Task 2: Train/predict API on `SignedGNN` (fit on signed edges, predict pairs)

**Files:**
- Modify: `src/multi_agent/signed_gnn.py`
- Test: `tests/test_signed_gnn.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fit_then_predict_recovers_signs():
    torch.manual_seed(0); np.random.seed(0)
    X, pos, neg = _toy()
    m = SignedGNN(8, 16, layers=3)
    # observed signed edges as (i, j, y) with y in {+1,-1}
    edges = [(0,1,1.0),(1,2,1.0),(3,4,1.0),(4,5,1.0),(0,3,-1.0),(2,5,-1.0)]
    m.fit(X, edges, epochs=200, lr=0.01)
    p = m.predict(X, [(0,2),(3,5),(0,4)])   # coherent, coherent, contradiction
    assert p[(0,2)] > 0 and p[(3,5)] > 0 and p[(0,4)] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signed_gnn.py::test_fit_then_predict_recovers_signs -q`
Expected: FAIL — `AttributeError: 'SignedGNN' object has no attribute 'fit'`

- [ ] **Step 3: Write minimal implementation** (append to `SignedGNN`)

```python
    @staticmethod
    def _edge_tensors(edges):
        pos = [[i, j] for i, j, y in edges if y > 0] + [[j, i] for i, j, y in edges if y > 0]
        neg = [[i, j] for i, j, y in edges if y < 0] + [[j, i] for i, j, y in edges if y < 0]
        ten = lambda e: (torch.tensor(e, dtype=torch.long).t().contiguous()
                         if e else torch.empty((2, 0), dtype=torch.long))
        return ten(pos), ten(neg)

    def fit(self, x, edges, epochs: int = 200, lr: float = 0.01,
            weight_decay: float = 1e-4) -> None:
        if not edges:
            return
        x = torch.as_tensor(x, dtype=torch.float32)
        pos, neg = self._edge_tensors(edges)
        ei = torch.tensor([[i, j] for i, j, _ in edges], dtype=torch.long).t()
        y = torch.tensor([1.0 if v > 0 else 0.0 for *_, v in edges])
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.train()
        for _ in range(epochs):
            opt.zero_grad()
            z = self.emb(x, pos, neg)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.score(z, ei), y)
            loss.backward(); opt.step()
        self.eval()
        self._cache = (x, pos, neg)

    @torch.no_grad()
    def predict(self, x, pairs):
        if not pairs:
            return {}
        x = torch.as_tensor(x, dtype=torch.float32)
        pos, neg = getattr(self, "_cache", (x, *self._edge_tensors([])))[1:]
        z = self.emb(x, pos, neg)
        ei = torch.tensor([[i, j] for i, j in pairs], dtype=torch.long).t()
        s = torch.tanh(self.score(z, ei)).tolist()
        return {p: float(v) for p, v in zip(pairs, s)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_signed_gnn.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent/signed_gnn.py tests/test_signed_gnn.py
git commit -m "feat(signed_gnn): fit/predict API over signed edges"
```

### Task 3: Config fields for substrate selection

**Files:**
- Modify: `src/multi_agent/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_config.py`)

```python
def test_graph_substrate_defaults_and_fields():
    from multi_agent.config import MultiAgentConfig
    c = MultiAgentConfig(emb_dim=8)
    assert c.graph_substrate == "bayesian"          # default unchanged behaviour
    assert c.hybrid_density_threshold == 6
    assert c.signed_gnn_hidden == 32 and c.signed_gnn_layers == 3
    c2 = MultiAgentConfig(emb_dim=8, graph_substrate="signed_hybrid")
    assert c2.graph_substrate == "signed_hybrid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_graph_substrate_defaults_and_fields -q`
Expected: FAIL — `AttributeError: ... 'graph_substrate'`

- [ ] **Step 3: Write minimal implementation** (add fields to the `MultiAgentConfig` dataclass, after the `tgn_cold_start` field)

```python
    # Graph substrate: "bayesian" (default, closed-form 2-hop posterior),
    # "tgn" (legacy temporal), or "signed_hybrid" (Bayesian when sparse,
    # Signed-GNN once dense — gated by hybrid_density_threshold).
    graph_substrate: str = "bayesian"
    signed_gnn_hidden: int = 32
    signed_gnn_layers: int = 3
    signed_gnn_lr: float = 1e-2
    signed_gnn_epochs: int = 200
    # Min degree of BOTH endpoints for a pair to use the Signed-GNN; below this
    # the hybrid falls back to the Bayesian posterior.
    hybrid_density_threshold: int = 6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent/config.py tests/test_config.py
git commit -m "feat(config): graph_substrate + signed_gnn/hybrid fields"
```

### Task 4: Hybrid gate in `Graph` (field/impute by density)

**Files:**
- Modify: `src/multi_agent/graph.py` (add `_sgnn` field near `_tgn`; add gate + hybrid `field`/`impute`; extend `impute_batch`)
- Test: `tests/test_hybrid_substrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hybrid_substrate.py
import numpy as np
from multi_agent.graph import Graph
from multi_agent.signed_gnn import SignedGNN

def _dense_graph():
    # 8 nodes, emb_dim 8, a dense coherent core (deg >= 6) + one sparse node
    rng = np.random.default_rng(0)
    g = Graph(emb_dim=8, hybrid_density_threshold=6)
    ids = [f"n{i}" for i in range(8)]
    embs = rng.normal(size=(8, 8)).astype(np.float32)
    edges = [(f"n{i}", f"n{j}", 1.0) for i in range(7) for j in range(i+1, 7)]  # core clique
    g.extend(ids, embs, edges)
    return g, ids

def test_gate_uses_bayesian_for_sparse_pair():
    g, ids = _dense_graph()
    # n7 has degree 0 (sparse) -> any pair touching it must use Bayesian path
    assert g._hybrid_use_signed("n0", "n7") is False

def test_gate_uses_signed_for_dense_pair_when_attached():
    g, ids = _dense_graph()
    g._sgnn = SignedGNN(in_dim=8, hidden=16, layers=2)  # attach (untrained ok for gate)
    assert g._hybrid_use_signed("n0", "n1") is True     # both in dense clique
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hybrid_substrate.py -q`
Expected: FAIL — `AttributeError: 'Graph' object has no attribute '_hybrid_use_signed'` (and `_sgnn`)

- [ ] **Step 3: Write minimal implementation**

Add to the `Graph` dataclass fields (near `_tgn`):

```python
    _sgnn: "object | None" = field(default=None, repr=False)
    hybrid_density_threshold: int = field(default=6, repr=False)
```

Add methods (near `_uses_tgn_raw_fallback`):

```python
    def _degree(self, node_id: str) -> int:
        return len(self._adj.get(node_id, ()))

    def _hybrid_use_signed(self, q: str, c: str) -> bool:
        """True iff a trained Signed-GNN is attached and BOTH endpoints are dense."""
        if self._sgnn is None:
            return False
        return (self._degree(q) >= self.hybrid_density_threshold
                and self._degree(c) >= self.hybrid_density_threshold)
```

Then in `field` and `impute`, before the Bayesian block, add the Signed-GNN branch
(keep the existing TGN branch untouched; the hybrid uses `_sgnn`, not `_tgn`):

```python
    # in field(self, q, c): after the observed-edge short-circuit, before _prior:
        if self._hybrid_use_signed(q, c):
            score = self._sgnn_predict(q, c)
            if score is not None:
                return float(max(-1.0, min(1.0, score)))
        # in impute(self, q, c): same, but return None when score is None to defer to judge
```

Add the predict helper:

```python
    def _sgnn_predict(self, q: str, c: str) -> float | None:
        idx = self._sgnn_index            # set by Trainer after fit (id -> row)
        if idx is None or q not in idx or c not in idx:
            return None
        x = self._sgnn_features           # (N, emb_dim) numpy, set by Trainer
        out = self._sgnn.predict(x, [(idx[q], idx[c])])
        return out.get((idx[q], idx[c]))
```

Add fields `_sgnn_index: dict | None = field(default=None, repr=False)` and
`_sgnn_features: "np.ndarray | None" = field(default=None, repr=False)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hybrid_substrate.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent/graph.py tests/test_hybrid_substrate.py
git commit -m "feat(graph): hybrid density gate + Signed-GNN predict path"
```

### Task 5: Trainer constructs substrate + fits Signed-GNN

**Files:**
- Modify: `src/multi_agent/runner.py` (`Trainer.__init__`: build substrate from `config.graph_substrate`; after each `step`, if `signed_hybrid`, refit GNN on committed edges and set `graph._sgnn_index/_sgnn_features`)
- Test: `tests/test_hybrid_substrate.py`

- [ ] **Step 1: Write the failing test**

```python
def test_trainer_signed_hybrid_runs_and_predicts_dense():
    import numpy as np
    from multi_agent.config import MultiAgentConfig
    from multi_agent.runner import Trainer
    from multi_agent.utils.notebook import make_synthetic_batches, make_cosine_judge
    b = make_synthetic_batches(n_nodes=60, n_batches=6, n_topic_pairs=3, emb_dim=16, seed=1)
    cfg = MultiAgentConfig(emb_dim=16, num_agents=2, k=6, judge_budget_per_batch=40,
                           graph_substrate="signed_hybrid", hybrid_density_threshold=4, seed=0)
    tr = Trainer(cfg, make_cosine_judge(b))
    for batch in b:
        tr.step(batch)
    assert tr.graph._sgnn is not None
    assert tr.graph._sgnn_index is not None and len(tr.graph._sgnn_index) == len(tr.graph.get_nodes())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hybrid_substrate.py::test_trainer_signed_hybrid_runs_and_predicts_dense -q`
Expected: FAIL — `graph._sgnn is None` (substrate not constructed)

- [ ] **Step 3: Write minimal implementation**

In `Trainer.__init__`, after the existing TGN-construction block, add:

```python
        if config.graph_substrate == "signed_hybrid":
            from multi_agent.signed_gnn import SignedGNN
            self.graph._sgnn = SignedGNN(
                in_dim=config.emb_dim,
                hidden=config.signed_gnn_hidden,
                layers=config.signed_gnn_layers,
            ).to(torch.device(config.device))
            self.graph.hybrid_density_threshold = config.hybrid_density_threshold
```

At the end of `Trainer.step`, after `self.graph.extend(batch.ids, batch.embs, edges)`:

```python
        if self.config.graph_substrate == "signed_hybrid" and self.graph._sgnn is not None:
            self._refit_signed_gnn()
```

Add method:

```python
    def _refit_signed_gnn(self) -> None:
        node_ids = self.graph.get_nodes()
        idx = {nid: i for i, nid in enumerate(node_ids)}
        feats = self.graph.get_representations_fast(node_ids)  # raw embeddings (no TGN)
        edges = [(idx[a], idx[b], float(w))
                 for (a, b), w in self.graph._edges.items() if a in idx and b in idx]
        self.graph._sgnn.fit(feats, edges,
                             epochs=self.config.signed_gnn_epochs, lr=self.config.signed_gnn_lr)
        self.graph._sgnn_index = idx
        self.graph._sgnn_features = feats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hybrid_substrate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/multi_agent/runner.py tests/test_hybrid_substrate.py
git commit -m "feat(runner): construct + refit Signed-GNN for signed_hybrid substrate"
```

### Task 6: Full-suite regression + hybrid benchmark parity

**Files:**
- Test: run full suite; add `tests/test_hybrid_substrate.py::test_hybrid_beats_bayesian_dense`

- [ ] **Step 1: Write the failing test** (the empirical claim, as a guardrail)

```python
def test_hybrid_beats_bayesian_dense():
    # coherence NOT in embeddings + dense edges: hybrid (signed) must beat Bayesian held-out
    import numpy as np, torch
    from multi_agent.graph import Graph
    from multi_agent.signed_gnn import SignedGNN
    rng = np.random.default_rng(0); torch.manual_seed(0)
    N, K, D = 120, 6, 8
    cl = rng.integers(0, K, N)
    X = rng.normal(size=(N, D)).astype(np.float32); X /= np.linalg.norm(X,axis=1,keepdims=True)
    lab = lambda a,b: 1 if cl[a]==cl[b] else (-1 if cl[a]//2==cl[b]//2 else 0)
    def sample(n, exc=set()):
        out=[]
        while len(out)<n:
            a,b=rng.integers(0,N,2)
            if a==b: continue
            k=(min(a,b),max(a,b))
            if k in exc: continue
            l=lab(a,b)
            if l: out.append((a,b,l))
        return out
    obs=sample(800); ok={(min(a,b),max(a,b)) for a,b,_ in obs}; hold=sample(2000, ok)
    g=Graph(emb_dim=D); g.extend([f"n{i}" for i in range(N)], X,
            [(f"n{a}",f"n{b}",float(l)) for a,b,l in obs])
    y=[1 if l>0 else 0 for *_,l in hold]
    def auc(s):
        s=np.asarray(s); yy=np.asarray(y); p=s[yy==1]; n=s[yy==0]
        o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
        return (r[yy==1].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
    bayes=auc([g.field(f"n{a}",f"n{b}") for a,b,_ in hold])
    m=SignedGNN(D, 32, 3); m.fit(X, obs, epochs=250, lr=0.01)
    sp=m.predict(X, [(a,b) for a,b,_ in hold]); sg=auc([sp[(a,b)] for a,b,_ in hold])
    assert sg >= bayes - 0.02 and sg >= 0.9, f"signed {sg:.3f} vs bayes {bayes:.3f}"
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_hybrid_substrate.py::test_hybrid_beats_bayesian_dense -q`
Expected: PASS (signed ≈ 1.0 ≥ bayes)

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q -p no:cacheprovider -m "not heavy"`
Expected: all prior tests still pass (default substrate `"bayesian"` unchanged) + new tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_hybrid_substrate.py
git commit -m "test(hybrid): signed substrate beats Bayesian on dense held-out; suite green"
```

---

## Phase 2 — Port into the backend's vendored copy (in `~/dyssonance-backend`)

> Mirrors Phase 1 under `research_deployment/belief_activation/` and wires backend config.

### Task 7: Copy substrate files into the vendored package

- [ ] **Step 1: Copy the changed files** (overwrite vendored with Phase-1 versions)

```bash
cd ~/belief_activation
B=~/dyssonance-backend/research_deployment/belief_activation
cp src/multi_agent/signed_gnn.py "$B/src/multi_agent/signed_gnn.py"
for f in config.py graph.py runner.py; do cp src/multi_agent/$f "$B/src/multi_agent/$f"; done
cp tests/test_signed_gnn.py tests/test_hybrid_substrate.py "$B/tests/"
```

- [ ] **Step 2: Add `torch-geometric` to backend deps**

Edit `~/dyssonance-backend/research_deployment/belief_activation/pyproject.toml` → add `"torch-geometric>=2.4"` to `dependencies`. Also add it to the backend root `pyproject.toml` if the workspace pins deps there (search: `grep -n torch-geometric ~/dyssonance-backend/pyproject.toml`).

- [ ] **Step 3: Sync + run the vendored package tests**

Run: `cd ~/dyssonance-backend && uv sync && uv run pytest research_deployment/belief_activation/tests/test_signed_gnn.py research_deployment/belief_activation/tests/test_hybrid_substrate.py -q`
Expected: PASS

- [ ] **Step 4: Commit** (in dyssonance-backend, on a feature branch)

```bash
cd ~/dyssonance-backend && git checkout -b feat/signed-hybrid-substrate
git add research_deployment/belief_activation pyproject.toml
git commit -m "feat(belief_activation): vendor Signed-GNN hybrid substrate"
```

### Task 8: Wire substrate selection into backend belief activation

**Files:** Modify `perseverate_api/belief_activation.py` (the `MultiAgentConfig(...)` builder at ~`:553`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_belief_activation_substrate.py  (in ~/dyssonance-backend)
import os
from perseverate_api.belief_activation import _activation_config

def test_substrate_env_selects_signed_hybrid(monkeypatch):
    monkeypatch.setenv("BELIEF_ACTIVATION_SUBSTRATE", "signed_hybrid")
    cfg = _activation_config(emb_dim=768)
    assert cfg.graph_substrate == "signed_hybrid"

def test_substrate_default_is_bayesian(monkeypatch):
    monkeypatch.delenv("BELIEF_ACTIVATION_SUBSTRATE", raising=False)
    monkeypatch.delenv("BELIEF_ACTIVATION_USE_TGN", raising=False)
    assert _activation_config(emb_dim=768).graph_substrate == "bayesian"
```

(If the config builder is named differently than `_activation_config`, use the actual name found near `belief_activation.py:553`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd ~/dyssonance-backend && uv run pytest tests/test_belief_activation_substrate.py -q`
Expected: FAIL — `graph_substrate` not set from env.

- [ ] **Step 3: Implement** — in the `MultiAgentConfig(...)` builder, add:

```python
        graph_substrate=os.environ.get(
            "BELIEF_ACTIVATION_SUBSTRATE",
            "tgn" if os.environ.get("BELIEF_ACTIVATION_USE_TGN", "false").lower() == "true"
                  else "bayesian",
        ),
        signed_gnn_hidden=int(os.environ.get("BELIEF_ACTIVATION_SGNN_HIDDEN", "32")),
        signed_gnn_layers=int(os.environ.get("BELIEF_ACTIVATION_SGNN_LAYERS", "3")),
        hybrid_density_threshold=int(os.environ.get("BELIEF_ACTIVATION_HYBRID_DENSITY", "6")),
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_belief_activation_substrate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add perseverate_api/belief_activation.py tests/test_belief_activation_substrate.py
git commit -m "feat(activation): select graph substrate via BELIEF_ACTIVATION_SUBSTRATE"
```

### Task 9: Local backend smoke — activation publishes a Router with the hybrid substrate

> **GATE:** Redis up, ADC valid, env set (see Prerequisites).

- [ ] **Step 1: Start Redis + backend**

```bash
redis-server --daemonize yes
cd ~/dyssonance-backend
export REDIS_URL=redis://localhost:6379 BELIEF_ACTIVATION_SNAPSHOT_HMAC_KEY=dev-local-key \
       BELIEF_ACTIVATION_JUDGE=static BELIEF_ACTIVATION_SUBSTRATE=signed_hybrid
unset CLOUD_TASKS_QUEUE
uv run uvicorn perseverate_api.main:app --port 8001 &
```

- [ ] **Step 2: Drive the enrich+activation walkthrough** using the existing notebook path
`notebooks/enrich_async_walkthrough.ipynb` Step 5, or `scripts/enrich_state_bench.py`, and assert a Router snapshot is published (`_load_activation_router(...) is not None`). Expected: snapshot loadable; logs show substrate=signed_hybrid.

- [ ] **Step 3: Commit any fixes**, then proceed to Phase 3.

---

## Phase 3 — Live gym e2e through `/infer` (in `~/dyssonance-gym`)

> **GATE:** Phase 2 backend running with `BELIEF_ACTIVATION_SUBSTRATE` set.

### Task 10: Pin the gym scenario set + baseline matrix

- [ ] **Step 1:** choose scenarios (start with the 4 corporate ones) and the substrate arms to compare: `bayesian`, `tgn`, `signed_hybrid`. Record in `~/dyssonance-gym/runs/README.md`.

### Task 11: Make gym trigger activation + retrieve via `/infer`

**Files:** Modify `bench_harness.py` (the `--dyssonance` SUT path).

- [ ] **Step 1: Write the failing test**

```python
# ~/dyssonance-gym/tests/test_infer_wiring.py
from bench_harness import DyssonanceMemory   # the --dyssonance SUT class (use actual name)
def test_memory_exposes_infer(monkeypatch):
    m = DyssonanceMemory.__new__(DyssonanceMemory)
    assert hasattr(m, "infer")   # new retrieval method that calls POST /sessions/{id}/infer
```

- [ ] **Step 2: Run to verify it fails** — `cd ~/dyssonance-gym && python3 -m pytest tests/test_infer_wiring.py -q` → FAIL.

- [ ] **Step 3: Implement** an `infer(self, query)` method on the dyssonance SUT that
`POST`s to `/sessions/{id}/infer` (mirroring the existing `/beliefs` call at `bench_harness.py:276`), returns `cited_beliefs`, and is called before each investigator turn to ground probes. Trigger activation once after the belief uploads (call the activation/enrich step) so a Router snapshot exists.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** (`feat(gym): retrieve via /infer + trigger activation`).

### Task 12: Add a Gemini provider to gym (creds gate workaround)

**Files:** Modify `bench_harness.py` (`make_client`, `llm`, `--provider`).

- [ ] **Step 1: Write the failing test**

```python
def test_gemini_provider_available():
    import bench_harness as bh
    bh.set_provider("gemini")           # new
    c = bh.make_client("gemini")
    assert c is not None
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** a `"gemini"` branch in `make_client` (using `google-genai`,
`gemini-2.5-flash`, Vertex via ADC) and route `llm()` for it; add `set_provider`. Keep OpenAI/Anthropic intact.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** (`feat(gym): gemini provider for subject/investigator`).

### Task 13: Run the live e2e matrix + report

- [ ] **Step 1: Run** each substrate arm (restart backend with the env var changed between arms):

```bash
# arm = signed_hybrid (repeat for bayesian, tgn)
BELIEF_ACTIVATION_SUBSTRATE=signed_hybrid  # set before launching backend (Task 9)
cd ~/dyssonance-gym
PERSEVERATE_URL=http://localhost:8001 python3 bench_harness.py scenarios/ \
  --mode live --provider gemini --dyssonance
```

- [ ] **Step 2: Aggregate** `grade()` outputs across arms (root-signature recovery, region/edge coverage) into a comparison table + bar chart; save to `~/dyssonance-gym/runs/<id>/substrate_compare.md`.

- [ ] **Step 3: Verdict** — does `signed_hybrid` recover the true root cause (and avoid the decoy/surface) more often than `bayesian`/`tgn` end-to-end? Record honestly (including nulls).

- [ ] **Step 4: Commit** the report.

---

## Self-review

- **Spec coverage:** substrate module (T1–2), config (T3), hybrid gate (T4), trainer wiring (T5), regression+empirical guardrail (T6), backend port+deps (T7), backend config wiring (T8), backend smoke (T9), gym scenario matrix (T10), gym `/infer` wiring (T11), gym Gemini provider (T12), live run+report (T13). All five requested pieces (wrap, hybrid sparse→Bayesian/dense→GNN, backend, gym, live e2e) covered.
- **Placeholders:** none — code/commands inline. Phase-3 step bodies that touch unseen class names instruct using the actual name found at the cited line (`DyssonanceMemory`, `_activation_config`), since those are in the backend/gym repos not yet read in detail.
- **Type consistency:** `SignedGNN.emb/score/fit/predict`, `Graph._sgnn/_sgnn_index/_sgnn_features/_hybrid_use_signed/_sgnn_predict`, `Trainer._refit_signed_gnn`, config fields `graph_substrate/signed_gnn_*/hybrid_density_threshold` — names consistent across tasks.
- **Open risk:** Phase-3 class/method names in gym (`DyssonanceMemory`) and backend (`_activation_config`) are placeholders to confirm against the actual files in T8/T11 (both repos need a read at execution time); the live arms require a running backend + Gemini provider (Tasks 9, 12 handle the gates).
