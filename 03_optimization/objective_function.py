"""
Module: 03_optimization.objective_function

Provides the core portfolio objective functions used during classical
portfolio optimization in the ARQPO framework, including:
- Portfolio variance
- Portfolio expected return
- Mean-Variance (Markowitz) objective
- Sharpe Ratio objective
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================
# Module Constants
# ============================================================

EPSILON = 1e-8


# ============================================================
# Validation Helper
# ============================================================

def _validate_inputs(
    weights: np.ndarray,
    covariance_matrix: Optional[np.ndarray] = None,
    expected_returns: Optional[np.ndarray] = None,
) -> None:
    """Validates portfolio optimization inputs for dimensions, types, and finite values."""
    
    # Validate weights
    if not isinstance(weights, np.ndarray):
        logger.error("weights must be a NumPy array.")
        raise TypeError("weights must be a NumPy array.")
    if weights.ndim != 1 or weights.size == 0:
        logger.error("weights must be a non-empty one-dimensional array.")
        raise ValueError("weights must be a non-empty one-dimensional array.")
    if not np.isfinite(weights).all():
        logger.error("weights contain NaN or infinite values.")
        raise ValueError("weights contain NaN or infinite values.")

    # Validate expected returns
    if expected_returns is not None:
        if not isinstance(expected_returns, np.ndarray):
            logger.error("expected_returns must be a NumPy array.")
            raise TypeError("expected_returns must be a NumPy array.")
        if expected_returns.ndim != 1 or expected_returns.size == 0:
            logger.error("expected_returns must be a non-empty one-dimensional array.")
            raise ValueError("expected_returns must be a non-empty one-dimensional array.")
        if not np.isfinite(expected_returns).all():
            logger.error("expected_returns contain NaN or infinite values.")
            raise ValueError("expected_returns contain NaN or infinite values.")
        if len(weights) != len(expected_returns):
            logger.error("weights and expected_returns dimensions do not match.")
            raise ValueError("weights and expected_returns must have equal length.")

    # Validate covariance matrix
    if covariance_matrix is not None:
        if not isinstance(covariance_matrix, np.ndarray):
            logger.error("covariance_matrix must be a NumPy array.")
            raise TypeError("covariance_matrix must be a NumPy array.")
        if covariance_matrix.ndim != 2:
            logger.error("covariance_matrix must be two-dimensional.")
            raise ValueError("covariance_matrix must be two-dimensional.")
        
        rows, cols = covariance_matrix.shape
        if rows == 0 or rows != cols:
            logger.error("covariance_matrix must be a non-empty square matrix.")
            raise ValueError("covariance_matrix must be a non-empty square matrix.")
        if rows != len(weights):
            logger.error("weights dimension does not match covariance matrix.")
            raise ValueError("weights length must equal covariance matrix dimension.")
        if not np.isfinite(covariance_matrix).all():
            logger.error("covariance_matrix contains NaN or infinite values.")
            raise ValueError("covariance_matrix contains NaN or infinite values.")
        if not np.allclose(covariance_matrix, covariance_matrix.T, atol=EPSILON):
            logger.warning("Covariance matrix is not perfectly symmetric.")


# ============================================================
# Portfolio Statistics
# ============================================================

def calculate_portfolio_variance(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """
    Calculates the total portfolio variance (w^T * Sigma * w).
    
    Args:
        weights: Portfolio weights vector of shape (N,).
        covariance_matrix: Covariance matrix of shape (N, N).
    Returns:
        Portfolio variance.
    """
    _validate_inputs(weights=weights, covariance_matrix=covariance_matrix)

    try:
        variance = float(weights.T @ covariance_matrix @ weights)
        if variance < 0.0 and abs(variance) < EPSILON:
            variance = 0.0
            
        logger.debug("Portfolio variance calculated successfully: %.8f", variance)
        return variance

    except Exception as exc:
        logger.exception("Failed to calculate portfolio variance.")
        raise RuntimeError("Portfolio variance calculation failed.") from exc


def calculate_portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Calculates the expected portfolio return (w^T * mu).
    
    Args:
        weights: Portfolio weights vector.
        expected_returns: Expected returns vector.
    Returns:
        Expected portfolio return.
    """
    _validate_inputs(weights=weights, expected_returns=expected_returns)

    try:
        portfolio_return = float(weights.T @ expected_returns)
        logger.debug("Portfolio return calculated successfully: %.8f", portfolio_return)
        return portfolio_return

    except Exception as exc:
        logger.exception("Failed to calculate portfolio return.")
        raise RuntimeError("Portfolio return calculation failed.") from exc


# ============================================================
# Objective Functions
# ============================================================

def mean_variance_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> float:
    """
    Calculates the classical Markowitz Mean-Variance objective function.
    Objective = (0.5 * variance) - (risk_aversion * return)
    
    Args:
        weights: Portfolio weights vector.
        expected_returns: Expected asset returns vector.
        covariance_matrix: Asset covariance matrix.
        risk_aversion: Risk-return trade-off coefficient (default: 1.0).
    Returns:
        Mean-Variance optimization objective value.
    """
    _validate_inputs(
        weights=weights,
        covariance_matrix=covariance_matrix,
        expected_returns=expected_returns,
    )

    if not isinstance(risk_aversion, (int, float)):
        logger.error("risk_aversion must be a numeric value.")
        raise TypeError("risk_aversion must be numeric.")
    if risk_aversion < 0:
        logger.error("risk_aversion cannot be negative.")
        raise ValueError("risk_aversion must be non-negative.")

    try:
        portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
        portfolio_return = calculate_portfolio_return(weights, expected_returns)
        
        objective_value = (0.5 * portfolio_variance) - (risk_aversion * portfolio_return)

        logger.debug(
            "Mean-Variance objective calculated successfully. "
            "Variance=%.8f, Return=%.8f, Risk Aversion=%.4f, Objective=%.8f",
            portfolio_variance, portfolio_return, risk_aversion, objective_value
        )
        return float(objective_value)

    except Exception as exc:
        logger.exception("Failed to calculate Mean-Variance objective.")
        raise RuntimeError("Mean-Variance objective calculation failed.") from exc


def sharpe_ratio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculates the negative Sharpe Ratio objective for minimization.
    Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio StdDev
    
    Args:
        weights: Portfolio weights vector.
        expected_returns: Expected asset returns vector.
        covariance_matrix: Asset covariance matrix.
        risk_free_rate: Baseline risk-free return (default: 0.0).
    Returns:
        Negative Sharpe Ratio.
    """
    _validate_inputs(
        weights=weights,
        covariance_matrix=covariance_matrix,
        expected_returns=expected_returns,
    )

    if not isinstance(risk_free_rate, (int, float)):
        logger.error("risk_free_rate must be a numeric value.")
        raise TypeError("risk_free_rate must be numeric.")

    try:
        portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
        portfolio_return = calculate_portfolio_return(weights, expected_returns)

        portfolio_variance = max(portfolio_variance, 0.0)
        portfolio_std = np.sqrt(portfolio_variance)

        if portfolio_std < EPSILON:
            logger.warning(
                "Portfolio standard deviation is approximately zero. "
                "Returning objective value of 0.0."
            )
            return 0.0

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        objective_value = -float(sharpe_ratio)

        logger.debug(
            "Sharpe Ratio objective calculated successfully. "
            "Return=%.8f, StdDev=%.8f, Risk-Free Rate=%.6f, Sharpe=%.8f",
            portfolio_return, portfolio_std, risk_free_rate, sharpe_ratio
        )
        return objective_value

    except Exception as exc:
        logger.exception("Failed to calculate Sharpe Ratio objective.")
        raise RuntimeError("Sharpe Ratio objective calculation failed.") from exc