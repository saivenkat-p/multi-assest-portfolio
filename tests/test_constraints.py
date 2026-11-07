"""
Tests for optimization constraints module.
"""

import numpy as np
import pytest

from optimization.constraints import (
    get_budget_constraint,
    get_weight_bounds,
    budget_penalty,
)


def test_get_budget_constraint():
    constraint = get_budget_constraint()
    assert constraint["type"] == "eq"
    assert callable(constraint["fun"])

    # Test balanced weights sum to 1
    w_valid = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(constraint["fun"](w_valid), 0.0)

    # Test unbalanced weights
    w_invalid = np.array([0.5, 0.2])
    assert np.isclose(constraint["fun"](w_invalid), -0.3)


def test_get_weight_bounds():
    bounds = get_weight_bounds(4, lower_bound=0.0, upper_bound=1.0)
    assert len(bounds) == 4
    for b in bounds:
        assert b == (0.0, 1.0)

    with pytest.raises(ValueError):
        get_weight_bounds(0)

    with pytest.raises(ValueError):
        get_weight_bounds(3, lower_bound=0.8, upper_bound=0.2)


def test_budget_penalty():
    # Exactly budget satisfied -> penalty 0
    w_valid = np.array([0.4, 0.6])
    assert np.isclose(budget_penalty(w_valid, penalty_factor=100.0), 0.0)

    # Violation -> (1.2 - 1.0)^2 * 100 = 0.04 * 100 = 4.0
    w_excess = np.array([0.6, 0.6])
    assert np.isclose(budget_penalty(w_excess, penalty_factor=100.0), 4.0)

    with pytest.raises(ValueError):
        budget_penalty(w_valid, penalty_factor=-1.0)
