import numpy as np
import torch

from multi_agent.graph import Graph
from multi_agent.signed_gnn import SignedGNN


def _dense_graph():
    rng = np.random.default_rng(0)
    g = Graph(emb_dim=8, hybrid_density_threshold=6)
    ids = [f"n{i}" for i in range(8)]
    embs = rng.normal(size=(8, 8)).astype(np.float32)
    # dense coherent clique over n0..n6 (degree 6 each); n7 isolated (degree 0)
    edges = [(f"n{i}", f"n{j}", 1.0) for i in range(7) for j in range(i + 1, 7)]
    g.extend(ids, embs, edges)
    return g, ids


def test_gate_uses_bayesian_for_sparse_pair():
    g, _ = _dense_graph()
    g._sgnn = SignedGNN(in_dim=8, hidden=16, layers=2)
    assert g._hybrid_use_signed("n0", "n7") is False  # n7 sparse


def test_gate_uses_signed_for_dense_pair_when_attached():
    g, _ = _dense_graph()
    g._sgnn = SignedGNN(in_dim=8, hidden=16, layers=2)
    assert g._hybrid_use_signed("n0", "n1") is True  # both in dense clique


def test_gate_off_when_no_sgnn():
    g, _ = _dense_graph()
    assert g._hybrid_use_signed("n0", "n1") is False  # no _sgnn attached


def test_trainer_signed_hybrid_runs_and_indexes():
    from multi_agent.config import MultiAgentConfig
    from multi_agent.runner import Trainer
    from multi_agent.utils.notebook import make_synthetic_batches, make_cosine_judge

    b = make_synthetic_batches(n_nodes=60, n_batches=6, n_topic_pairs=3, emb_dim=16, seed=1)
    cfg = MultiAgentConfig(
        emb_dim=16, num_agents=2, k=6, judge_budget_per_batch=40,
        graph_substrate="signed_hybrid", hybrid_density_threshold=4, seed=0,
    )
    tr = Trainer(cfg, make_cosine_judge(b))
    for batch in b:
        tr.step(batch)
    assert tr.graph._sgnn is not None
    assert tr.graph._sgnn_index is not None
    assert len(tr.graph._sgnn_index) == len(tr.graph.get_nodes())


def _auc(scores, y):
    s = np.asarray(scores); y = np.asarray(y); p = s[y == 1]; n = s[y == 0]
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def test_hybrid_beats_bayesian_dense():
    """Coherence NOT in embeddings + dense edges: the Signed-GNN must match/beat
    the Bayesian posterior on held-out signed link prediction."""
    rng = np.random.default_rng(0); torch.manual_seed(0)
    N, K, D = 120, 6, 8
    cl = rng.integers(0, K, N)
    X = rng.normal(size=(N, D)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)

    def lab(a, b):
        return 1 if cl[a] == cl[b] else (-1 if cl[a] // 2 == cl[b] // 2 else 0)

    def sample(n, exc=set()):
        out = []
        while len(out) < n:
            a, b = rng.integers(0, N, 2)
            if a == b:
                continue
            k = (min(a, b), max(a, b))
            if k in exc:
                continue
            l = lab(a, b)
            if l:
                out.append((a, b, l))
        return out

    obs = sample(800); ok = {(min(a, b), max(a, b)) for a, b, _ in obs}; hold = sample(2000, ok)
    y = [1 if l > 0 else 0 for *_, l in hold]
    g = Graph(emb_dim=D)
    g.extend([f"n{i}" for i in range(N)], X, [(f"n{a}", f"n{b}", float(l)) for a, b, l in obs])
    bayes = _auc([g.field(f"n{a}", f"n{b}") for a, b, _ in hold], y)
    m = SignedGNN(D, 32, 3); m.fit(X, obs, epochs=250, lr=0.01)
    sp = m.predict(X, [(a, b) for a, b, _ in hold])
    signed = _auc([sp[(a, b)] for a, b, _ in hold], y)
    assert signed >= 0.9, f"signed AUC {signed:.3f} too low"
    assert signed >= bayes - 0.02, f"signed {signed:.3f} < bayes {bayes:.3f}"
