import numpy as np
import torch

from multi_agent.signed_gnn import SignedGNN


def _toy():
    # 6 nodes, two coherent clusters {0,1,2} and {3,4,5}; cross-cluster = contradiction
    X = torch.randn(6, 8)
    pos = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    neg = torch.tensor([[0, 3, 2, 5], [3, 0, 5, 2]])
    return X, pos, neg


def test_emb_shape_and_score_range():
    X, pos, neg = _toy()
    m = SignedGNN(in_dim=8, hidden=16, layers=3)
    z = m.emb(X, pos, neg)
    assert z.shape[0] == 6 and z.dim() == 2
    s = m.score(z, torch.tensor([[0, 0], [1, 3]]))  # (0,1) coherent, (0,3) contradiction
    assert s.shape == (2,)
    assert torch.isfinite(s).all()


def test_layers_controls_hops():
    assert len(SignedGNN(8, 16, layers=1).convs) == 1
    assert len(SignedGNN(8, 16, layers=3).convs) == 3


def test_fit_then_predict_recovers_signs():
    # Mechanics check: after fit, the TRAINED edges are recovered with the right
    # sign. (Held-out generalization needs a real graph — see the Task-6 guardrail
    # test_hybrid_beats_bayesian_dense on 120 nodes; a 6-node toy can't generalize.)
    torch.manual_seed(0)
    np.random.seed(0)
    X, _, _ = _toy()
    m = SignedGNN(8, 16, layers=3)
    edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0), (0, 3, -1.0), (2, 5, -1.0)]
    m.fit(X, edges, epochs=300, lr=0.02)
    p = m.predict(X, [(0, 1), (3, 4), (0, 3), (2, 5)])  # trained: +, +, -, -
    assert p[(0, 1)] > 0 and p[(3, 4)] > 0 and p[(0, 3)] < 0 and p[(2, 5)] < 0
