from __future__ import annotations

from multi_agent.psro import _auc_prediction_groups, _signed_auc


def test_auc_counts_use_same_valid_pairs_as_signed_auc():
    items = [
        *[{"actual": 0.9, "predicted": p} for p in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)],
        *[{"actual": -0.9, "predicted": p} for p in (0.1, 0.2, 0.3, 0.4, 0.5)],
        {"actual": 0.8, "predicted": "bad"},
        {"actual": 0.05, "predicted": 0.5},
        {"predicted": 0.5},
    ]

    pos, neg = _auc_prediction_groups(items)

    assert len(pos) == 6
    assert len(neg) == 5
    assert _signed_auc(items) is not None
