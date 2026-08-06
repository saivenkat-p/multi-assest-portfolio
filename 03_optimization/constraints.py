"""
Module: 03_optimization.constraints

Provides portfolio optimization constraints and penalty functions for
classical and quantum portfolio optimization.

Functions:
    - get_budget_constraint()
    - get_weight_bounds()
    - budget_penalty()
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Module Logger
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Module Constants
# -----------------------------------------------------------------------------

DEFAULT_PENALTY_FACTOR = 1000.0


# -----------------------------------------------------------------------------
# Validation Helpers
# -----------------------------------------------------------------------------

def _validate_bounds(
    n_assets: int,
    lower_bound: float,
    upper_bound: float
) -> None:
    """
    Validate weight bound parameters.

    Args:
        n_assets: Number of assets.
        lower_bound: Minimum allowable asset weight.
        upper_bound: Maximum allowable asset weight.

    Raises:
        TypeError:
            If inputs have invalid types.
        ValueError:
            If values are outside valid ranges.
    """
    if not isinstance(n_assets, int):
        raise TypeError("n_assets must be an integer.")

    if n_assets <= 0:
        raise ValueError("n_assets must be greater than zero.")

    if not isinstance(lower_bound, (int, float)):
        raise TypeError("lower_bound must be numeric.")

    if not isinstance(upper_bound, (int, float)):
        raise TypeError("upper_bound must be numeric.")

    if not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite.")

    if not np.isfinite(upper_bound):
        raise ValueError("upper_bound must be finite.")

    if lower_bound > upper_bound:
        raise ValueError(
            "lower_bound cannot be greater than upper_bound."
        )


def _validate_weights(
    weights: np.ndarray,
    penalty_factor: float
) -> np.ndarray:
    """
    Validate portfolio weights and penalty factor.

    Args:
        weights: Portfolio weights.
        penalty_factor: Penalty scaling factor.

    Returns:
        Validated NumPy array.

    Raises:
        TypeError:
            If input types are invalid.
        ValueError:
            If inputs contain invalid values.
    """
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy ndarray.")

    if weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional array.")

    if weights.size == 0:
        raise ValueError("weights cannot be empty.")

    if not np.issubdtype(weights.dtype, np.number):
        raise TypeError("weights must contain numeric values.")

    if np.isnan(weights).any():
        raise ValueError("weights contain NaN values.")

    if np.isinf(weights).any():
        raise ValueError("weights contain infinite values.")

    if not isinstance(penalty_factor, (int, float)):
        raise TypeError("penalty_factor must be numeric.")

    if not np.isfinite(penalty_factor):
        raise ValueError("penalty_factor must be finite.")

    if penalty_factor <= 0:
        raise ValueError("penalty_factor must be greater than zero.")

    return weights.astype(float)


# -----------------------------------------------------------------------------
# Public Functions
# -----------------------------------------------------------------------------

def get_budget_constraint() -> Dict[str, Any]:
    """
    Create a SciPy-compatible equality constraint enforcing
    the portfolio budget constraint.

    Constraint:
        sum(weights) = 1

    Returns:
        Dict[str, Any]:
            SciPy optimization constraint dictionary.
    """
    logger.debug("Creating budget constraint.")

    return {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1.0,
    }


def get_weight_bounds(
    n_assets: int,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> Tuple[Tuple[float, float], ...]:
    """
    Generate optimization bounds for portfolio weights.

    Args:
        n_assets:
            Number of assets.

        lower_bound:
            Minimum allowable weight.

        upper_bound:
            Maximum allowable weight.

    Returns:
        Tuple[Tuple[float, float], ...]:
            Bounds for each asset.

    Raises:
        TypeError:
            If argument types are invalid.

        ValueError:
            If arguments contain invalid values.
    """
    logger.info(
        "Generating weight bounds for %d assets.",
        n_assets,
    )

    _validate_bounds(
        n_assets,
        lower_bound,
        upper_bound,
    )

    return tuple(
        (lower_bound, upper_bound)
        for _ in range(n_assets)
    )


def budget_penalty(
    weights: np.ndarray,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
) -> float:
    """
    Compute the quadratic penalty associated with violating
    the portfolio budget constraint.

    Penalty Function:

        penalty = λ (Σw - 1)^2

    where:

        λ = penalty_factor

    This formulation is commonly used when converting constrained
    optimization problems into unconstrained QUBO formulations
    suitable for quantum optimization algorithms.

    Args:
        weights:
            Portfolio weights.

        penalty_factor:
            Penalty scaling coefficient.

    Returns:
        float:
            Budget constraint penalty.

    Raises:
        TypeError:
            If input types are invalid.

        ValueError:
            If inputs contain invalid values.
    """
    logger.debug("Computing budget penalty.")

    weights = _validate_weights(
        weights,
        penalty_factor,
    )

    penalty = penalty_factor * (np.sum(weights) - 1.0) ** 2

    logger.debug("Budget penalty = %.10f", penalty)

    return float(penalty)