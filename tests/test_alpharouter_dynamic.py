import numpy as np
import torch

from alpharouter.committee_experiment import (_hard_consensus, _residual_gate,
                                              _training_prior)
from alpharouter.experiment import (EXPERT_NAMES, TrainingParameters,
                                    _dynamic_utility, dynamic_label_gate,
                                    hard_gate)
from alpharouter.model import DynamicExpertRouter


def test_dynamic_labels_are_per_date_and_risk_adjusted() -> None:
    path = np.array(
        [
            [[0.02, 0.01, 0.00, -0.01], [0.02, 0.01, 0.00, -0.01]],
            [[-0.02, 0.00, 0.01, 0.02], [-0.02, 0.00, 0.01, 0.02]],
        ],
        dtype=np.float32,
    )
    experts = np.full((2, 4, 8), 1.0 / 8.0, dtype=np.float32)
    utility = _dynamic_utility(path, experts, np.zeros((2, 4)), TrainingParameters())
    assert utility.argmax(axis=1).tolist() == [0, 3]


def test_dynamic_utility_charges_turnover_on_every_horizon_day() -> None:
    path = np.zeros((1, 2, 4), dtype=np.float32)
    experts = np.full((1, 2, 4, 8), 1.0 / 8.0, dtype=np.float32)
    no_cost = _dynamic_utility(path, experts, np.zeros((1, 2, 4)), TrainingParameters())
    costly = _dynamic_utility(path, experts, np.ones((1, 2, 4)), TrainingParameters())
    assert np.all(costly < no_cost)


def test_router_emits_shared_expert_gate() -> None:
    router = DynamicExpertRouter(25, hidden=32, heads=4, layers=1, feedforward=64)
    state = torch.randn(5, 30, 25)
    logits, gate = router(state)
    assert logits.shape == (5, 4)
    assert torch.allclose(gate.sum(dim=1), torch.ones(5), atol=1e-6)


def test_switching_margin_only_changes_experts_when_evidence_is_large() -> None:
    probability = np.array([[0.60, 0.40], [0.49, 0.51], [0.20, 0.80]], dtype=np.float32)
    choice = hard_gate(probability, switching_margin=0.20).argmax(axis=1)
    assert choice.tolist() == [0, 0, 1]


def test_test_labels_cannot_be_used_as_oracle() -> None:
    panel = type("Panel", (), {"label": np.array([-1, -1], dtype=np.int64)})()
    try:
        dynamic_label_gate(panel)
    except ValueError as error:
        assert "test split" in str(error)
    else:
        raise AssertionError("test-label oracle must be rejected")


def test_expert_names_do_not_contain_market_identifiers() -> None:
    assert not any(
        token in name
        for name in EXPERT_NAMES
        for token in ("djia", "hsi", "csi", "crypto")
    )


def test_residual_gate_keeps_the_requested_dynamic_sleeve_and_simplex() -> None:
    prior = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dynamic = np.eye(4, dtype=np.float32)[np.array([1, 2, 3])]
    gate = _residual_gate(prior, dynamic, adaptive_share=0.20)
    assert np.allclose(gate.sum(axis=1), 1.0)
    assert np.allclose(gate[np.arange(3), [1, 2, 3]], 0.20)
    assert np.allclose(gate[:, 0], 0.80)


def test_training_prior_uses_only_supplied_training_panel() -> None:
    returns = np.array([[0.01, 0.00], [0.01, 0.00], [0.01, 0.00]], dtype=float)
    experts = np.repeat(
        np.array([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]]]),
        len(returns),
        axis=0,
    )
    panel = type(
        "Panel",
        (),
        {"experts": experts, "market": type("Market", (), {"next_returns": returns})()},
    )()
    prior, audit = _training_prior(panel, lookback=3)
    assert prior.argmax() == 0
    assert audit["selected_expert"] == EXPERT_NAMES[0]


def test_hard_consensus_falls_back_to_prior_on_any_disagreement() -> None:
    prior = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    probability = np.array(
        [
            [[0.1, 0.8, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]],
            [[0.1, 0.8, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]],
            [[0.1, 0.8, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]],
            [[0.1, 0.8, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]],
            [[0.1, 0.8, 0.05, 0.05], [0.8, 0.1, 0.05, 0.05]],
        ],
        dtype=np.float32,
    )
    choice = _hard_consensus(probability, 0.0, prior).argmax(axis=1)
    assert choice.tolist() == [1, 0]
