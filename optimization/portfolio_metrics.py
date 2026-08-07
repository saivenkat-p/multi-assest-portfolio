"""
Module: portfolio_metrics
=========================

Computes standard financial performance metrics for an optimized portfolio.

This module evaluates a portfolio using:

    • Expected Return
    • Portfolio Variance
    • Portfolio Volatility
    • Sharpe Ratio

The module is designed for the ARQPO (Adaptive Regime-Aware Quantum Portfolio
Optimizer) pipeline and is executed immediately after adaptive portfolio
weights are generated.

The implementation emphasizes:

    • Numerical stability
    • Production-grade validation
    • Robust logging
    • Backward compatibility
    • Financial correctness

Public API:
    evaluate_portfolio()
"""

import logging
from typing import Dict

import numpy as np

# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =============================================================================
# Module Constants
# =============================================================================

EPSILON = 1e-8

# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_inputs(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
) -> None:
    """
    Validate portfolio evaluation inputs.

    Args:
        weights:
            Portfolio weight vector.

        expected_returns:
            Expected return vector.

        covariance_matrix:
            Asset covariance matrix.

        risk_free_rate:
            Risk-free rate.

    Raises:
        TypeError:
            If input types are invalid.

        ValueError:
            If dimensions, shapes, or numerical values are invalid.
    """

    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy ndarray.")

    if not isinstance(expected_returns, np.ndarray):
        raise TypeError("expected_returns must be a NumPy ndarray.")

    if not isinstance(covariance_matrix, np.ndarray):
        raise TypeError("covariance_matrix must be a NumPy ndarray.")

    if not isinstance(risk_free_rate, (int, float)):
        raise TypeError("risk_free_rate must be numeric.")

    if weights.size == 0:
        raise ValueError("weights cannot be empty.")

    if expected_returns.size == 0:
        raise ValueError("expected_returns cannot be empty.")

    if covariance_matrix.size == 0:
        raise ValueError("covariance_matrix cannot be empty.")

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")

    if expected_returns.ndim != 1:
        raise ValueError("expected_returns must be one-dimensional.")

    if covariance_matrix.ndim != 2:
        raise ValueError("covariance_matrix must be two-dimensional.")

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be square.")

    if len(weights) != len(expected_returns):
        raise ValueError(
            "weights and expected_returns must have the same length."
        )

    if covariance_matrix.shape[0] != len(weights):
        raise ValueError(
            "covariance_matrix dimensions must match weights."
        )

    if not np.allclose(covariance_matrix, covariance_matrix.T):
        raise ValueError("covariance_matrix must be symmetric.")

    arrays = {
        "weights": weights,
        "expected_returns": expected_returns,
        "covariance_matrix": covariance_matrix,
    }

    for name, array in arrays.items():

        if np.isnan(array).any():
            raise ValueError(f"{name} contains NaN values.")

        if np.isinf(array).any():
            raise ValueError(f"{name} contains infinite values.")


def evaluate_portfolio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate financial performance metrics for a portfolio.

    Args:
        weights:
            Portfolio allocation vector.

        expected_returns:
            Expected annualized return of each asset.

        covariance_matrix:
            Asset covariance matrix.

        risk_free_rate:
            Risk-free rate used for Sharpe Ratio.

    Returns:
        Dictionary containing:

            - Expected_Return
            - Volatility
            - Variance
            - Sharpe_Ratio

    Raises:
        TypeError:
            If input types are invalid.

        ValueError:
            If inputs contain invalid numerical values or incompatible shapes.

        RuntimeError:
            If numerical computation fails.
    """
    try:
        logger.info("Evaluating portfolio performance metrics.")

        _validate_inputs(
            weights,
            expected_returns,
            covariance_matrix,
            risk_free_rate,
        )

        # ---------------------------------------------------------------------
        # Portfolio Variance
        # ---------------------------------------------------------------------
        variance = float(weights.T @ covariance_matrix @ weights)

        # Handle tiny negative values caused by floating-point precision
        if variance < 0 and abs(variance) < EPSILON:
            logger.warning(
                "Small negative variance detected due to floating-point "
                "precision. Clamping variance to zero."
            )
            variance = 0.0

        if variance < 0:
            raise RuntimeError(
                "Computed portfolio variance is negative."
            )

        # ---------------------------------------------------------------------
        # Portfolio Volatility
        # ---------------------------------------------------------------------
        volatility = float(np.sqrt(variance))

        # ---------------------------------------------------------------------
        # Expected Portfolio Return
        # ---------------------------------------------------------------------
        expected_return = float(weights.T @ expected_returns)

        # ---------------------------------------------------------------------
        # Sharpe Ratio
        # ---------------------------------------------------------------------
        if volatility < EPSILON:
            logger.warning(
                "Portfolio volatility is near zero. "
                "Sharpe Ratio set to 0.0."
            )
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (
                expected_return - risk_free_rate
            ) / volatility

        metrics = {
            "Expected_Return": expected_return,
            "Volatility": volatility,
            "Variance": variance,
            "Sharpe_Ratio": sharpe_ratio,
        }

        logger.info("Portfolio metrics computed successfully.")
        logger.debug("Portfolio Metrics: %s", metrics)

        return metrics

    except Exception:
        logger.exception(
            "Failed to evaluate portfolio metrics."
        )
        raise