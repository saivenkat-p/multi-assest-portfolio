"""
Module: 03_optimization.transaction_cost

Provides utilities for estimating portfolio transaction (rebalancing) costs
based on changes in asset allocation.

The transaction cost is modeled as a linear function of portfolio turnover:

    Transaction Cost = Cost Rate × Σ |Target Weight - Current Weight|

This module is designed to be compatible with both classical portfolio
optimization workflows and downstream quantum optimization pipelines.
"""

from __future__ import annotations

import logging

import numpy as np

# -----------------------------------------------------------------------------
# Module Logger
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Module Constants
# -----------------------------------------------------------------------------

DEFAULT_COST_RATE: float = 0.001


# -----------------------------------------------------------------------------
# Validation Helpers
# -----------------------------------------------------------------------------

def _validate_weights(
    weights: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Validate a portfolio weight vector.

    Args:
        weights: Portfolio weights.
        name: Variable name used in error messages.

    Returns:
        Validated NumPy array.

    Raises:
        TypeError:
            If weights is not a NumPy array.
        ValueError:
            If weights is empty, contains NaN/Inf values,
            or is not one-dimensional.
    """
    if not isinstance(weights, np.ndarray):
        logger.error("%s must be a NumPy array.", name)
        raise TypeError(f"{name} must be a NumPy array.")

    if weights.size == 0:
        logger.error("%s cannot be empty.", name)
        raise ValueError(f"{name} cannot be empty.")

    if weights.ndim != 1:
        logger.error("%s must be one-dimensional.", name)
        raise ValueError(f"{name} must be a one-dimensional array.")

    if not np.issubdtype(weights.dtype, np.number):
        logger.error("%s must contain numeric values.", name)
        raise TypeError(f"{name} must contain numeric values.")

    if np.isnan(weights).any():
        logger.error("%s contains NaN values.", name)
        raise ValueError(f"{name} cannot contain NaN values.")

    if np.isinf(weights).any():
        logger.error("%s contains infinite values.", name)
        raise ValueError(f"{name} cannot contain infinite values.")

    return weights


def _validate_cost_rate(cost_rate: float) -> float:
    """
    Validate the transaction cost rate.

    Args:
        cost_rate: Transaction cost per unit traded.

    Returns:
        Validated cost rate.

    Raises:
        TypeError:
            If cost_rate is not numeric.
        ValueError:
            If cost_rate is negative, NaN, or infinite.
    """
    if not isinstance(cost_rate, (int, float, np.floating)):
        logger.error("cost_rate must be numeric.")
        raise TypeError("cost_rate must be numeric.")

    if np.isnan(cost_rate):
        logger.error("cost_rate cannot be NaN.")
        raise ValueError("cost_rate cannot be NaN.")

    if np.isinf(cost_rate):
        logger.error("cost_rate cannot be infinite.")
        raise ValueError("cost_rate cannot be infinite.")

    if cost_rate < 0:
        logger.error("cost_rate cannot be negative.")
        raise ValueError("cost_rate must be non-negative.")

    return float(cost_rate)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def calculate_rebalancing_cost(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_rate: float = DEFAULT_COST_RATE,
) -> float:
    """
    Calculate the transaction cost required to rebalance a portfolio.

    The cost is computed using a linear turnover model:

        Cost = Cost Rate × Σ |Target Weight − Current Weight|

    Args:
        current_weights:
            Current portfolio allocation.
        target_weights:
            Desired portfolio allocation.
        cost_rate:
            Transaction fee per unit of traded weight.
            Default is 0.001 (0.1%).

    Returns:
        Estimated total transaction cost.

    Raises:
        TypeError:
            If inputs are of invalid types.
        ValueError:
            If inputs are empty, contain invalid values,
            have mismatched shapes, or if cost_rate is invalid.
    """
    logger.info("Calculating portfolio transaction cost.")

    current_weights = _validate_weights(
        current_weights,
        "current_weights",
    )
    target_weights = _validate_weights(
        target_weights,
        "target_weights",
    )
    cost_rate = _validate_cost_rate(cost_rate)

    if current_weights.shape != target_weights.shape:
        logger.error(
            "Weight arrays must have identical shapes."
        )
        raise ValueError(
            "current_weights and target_weights must have the same shape."
        )

    weight_changes = np.abs(target_weights - current_weights)
    total_turnover = float(np.sum(weight_changes))

    total_cost = float(total_turnover * cost_rate)

    logger.info(
        "Transaction cost calculated successfully."
    )

    logger.debug(
        "Turnover=%.6f | Cost Rate=%.6f | Total Cost=%.6f",
        total_turnover,
        cost_rate,
        total_cost,
    )

    return total_cost
