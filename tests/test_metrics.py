"""
Tests for portfolio metrics evaluation module.
"""

import numpy as np
import pytest

from optimization.portfolio_metrics import evaluate_portfolio


def test_evaluate_portfolio_basic():
    weights = np.array([0.6, 0.4])
    expected_returns = np.array([0.10, 0.05])
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.02]])

    metrics = evaluate_portfolio(weights, expected_returns, cov_matrix, risk_free_rate=0.02)

    # Expected return: 0.6 * 0.10 + 0.4 * 0.05 = 0.08
    assert np.isclose(metrics["Expected_Return"], 0.08)

    # Expected variance: w^T Sigma w
    # [0.6, 0.4] @ [[0.04, 0.01], [0.01, 0.02]] @ [0.6, 0.4]^T = 0.0224
    assert np.isclose(metrics["Variance"], 0.0224)
    assert np.isclose(metrics["Volatility"], np.sqrt(0.0224))

    expected_sharpe = (0.08 - 0.02) / np.sqrt(0.0224)
    assert np.isclose(metrics["Sharpe_Ratio"], expected_sharpe)


def test_evaluate_portfolio_validation():
    weights = np.array([0.5, 0.5])
    returns = np.array([0.1])
    cov = np.eye(2)

    with pytest.raises((ValueError, TypeError)):
        evaluate_portfolio(weights, returns, cov)
