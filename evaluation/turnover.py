"""
ARQPO - Adaptive Regime-Aware Quantum Portfolio Optimizer

Module:
    07_evaluation.turnover

Purpose:
    Production-grade evaluation of portfolio turnover. This module
    computes the turnover required to rebalance a portfolio from the
    current asset allocation to a target allocation.

Workflow Position:
    Current Portfolio Allocation
                │
                ▼
    Target Portfolio Allocation
                │
                ▼
    Allocation Difference
                │
                ▼
        Portfolio Turnover
                │
                ▼
    Benchmarking / Explainability

Mathematical Objective:
    Compute portfolio turnover as one-half of the sum of absolute
    differences between current and target portfolio weights.

Design Philosophy:
    • Defensive Programming
    • Numerical Robustness
    • Stable Public API
    • Production Logging
    • Modular Architecture
    • Downstream Compatibility

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Competition:
    WISER × Vanguard Quantum Challenge 2026
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE: float = 1e-10
LOGGER_NAME: str = __name__

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


class TurnoverCalculator:
    """
    Computes the portfolio turnover required to rebalance from the
    current allocation to the desired target allocation.

    The public API is intentionally stable to support downstream
    benchmarking, explainability, and dashboard modules.
    """

    def __init__(self) -> None:
        """
        Initialize the turnover calculator.

        Notes:
            Constructor retained for API stability and future
            extensibility.
        """
        logger.info("TurnoverCalculator initialized.")

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_inputs(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> None:
        """
        Validate turnover calculation inputs.

        Args:
            current_weights:
                Current portfolio allocation.

            target_weights:
                Desired portfolio allocation.

        Raises:
            TypeError:
                If inputs are not NumPy arrays.

            ValueError:
                If dimensions, sizes, or numerical values are invalid.
        """

        if not isinstance(current_weights, np.ndarray):
            raise TypeError(
                "current_weights must be a NumPy ndarray."
            )

        if not isinstance(target_weights, np.ndarray):
            raise TypeError(
                "target_weights must be a NumPy ndarray."
            )

        if current_weights.ndim != 1:
            raise ValueError(
                "current_weights must be one-dimensional."
            )

        if target_weights.ndim != 1:
            raise ValueError(
                "target_weights must be one-dimensional."
            )

        if current_weights.size == 0:
            raise ValueError(
                "current_weights cannot be empty."
            )

        if target_weights.size == 0:
            raise ValueError(
                "target_weights cannot be empty."
            )

        if current_weights.size != target_weights.size:
            raise ValueError(
                "Both weight vectors must have identical length."
            )

        if not np.isfinite(current_weights).all():
            raise ValueError(
                "current_weights contains NaN or infinite values."
            )

        if not np.isfinite(target_weights).all():
            raise ValueError(
                "target_weights contains NaN or infinite values."
            )

        if np.isclose(
            np.sum(current_weights),
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "Current portfolio weights cannot sum to zero."
            )

        if np.isclose(
            np.sum(target_weights),
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "Target portfolio weights cannot sum to zero."
            )
        
        if np.any(current_weights < 0.0):
            raise ValueError(
                "current_weights cannot contain negative values."
            )

        if np.any(target_weights < 0.0):
            raise ValueError(
                "target_weights cannot contain negative values."
            )
          
        logger.info(
            "Input validation completed successfully."
        )

    # =========================================================================
    # Numerical Utilities
    # =========================================================================

    def _cleanup_array(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Clean numerical artifacts from an array.

        Args:
            values:
                Numerical array.

        Returns:
            Cleaned NumPy array.
        """

        cleaned = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        cleaned[
            np.isclose(
                cleaned,
                0.0,
                atol=FLOAT_TOLERANCE,
            )
        ] = 0.0

        return cleaned

    def _cleanup_scalar(
        self,
        value: float,
    ) -> float:
        """
        Clean numerical artifacts from a scalar.

        Args:
            value:
                Floating-point value.

        Returns:
            Clean scalar.
        """

        if np.isclose(
            value,
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            return 0.0

        return float(value)

    # =========================================================================
    # Public API
    # =========================================================================

    def calculate(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> float:
        """
        Calculate portfolio turnover.

        Portfolio turnover is defined as one-half of the sum of the
        absolute differences between the current and target portfolio
        allocations.

        Args:
            current_weights:
                Current portfolio allocation.

            target_weights:
                Desired portfolio allocation.

        Returns:
            Portfolio turnover.

        Raises:
            RuntimeError:
                If turnover calculation fails.
        """
        logger.info("Starting portfolio turnover calculation.")

        start_time = time.perf_counter()

        try:
            # -------------------------------------------------------------
            # Input Validation
            # -------------------------------------------------------------
            self._validate_inputs(
                current_weights,
                target_weights,
            )

            # -------------------------------------------------------------
            # Allocation Difference
            # -------------------------------------------------------------
            absolute_difference = np.abs(
                target_weights - current_weights
            )

            absolute_difference = self._cleanup_array(
                absolute_difference
            )

            # -------------------------------------------------------------
            # Turnover Calculation
            # -------------------------------------------------------------
            turnover = float(
                np.sum(absolute_difference) / 2.0
            )

            turnover = self._cleanup_scalar(
                turnover
            )

            if turnover < 0.0:
                logger.warning(
                    "Negative turnover detected."
                )

            runtime = time.perf_counter() - start_time

            logger.info(
                "Portfolio turnover calculated successfully."
            )

            logger.info(
                "Turnover : %.6f",
                turnover,
            )

            logger.info(
                "Assets Processed : %d",
                current_weights.size,
            )

            logger.info(
                "Execution Time : %.6f seconds",
                runtime,
            )

            return turnover

        except Exception as exc:
            logger.exception(
                "Portfolio turnover calculation failed."
            )
            raise RuntimeError(
                "Failed to calculate portfolio turnover."
            ) from exc

# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":

    logger.info("=" * 80)
    logger.info("Starting TurnoverCalculator Runtime Tests")
    logger.info("=" * 80)

    print("\nPython Runtime Tests for TurnoverCalculator\n")

    calculator = TurnoverCalculator()

    # -------------------------------------------------------------------------
    # Test 1: Happy Path
    # -------------------------------------------------------------------------
    print("[TEST 1] Happy Path")

    current = np.array([0.40, 0.30, 0.30])
    target = np.array([0.35, 0.35, 0.30])

    turnover = calculator.calculate(current, target)

    print("Turnover :", turnover)

    # -------------------------------------------------------------------------
    # Test 2: Medium Input
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Medium Input")

    np.random.seed(42)

    current = np.random.dirichlet(np.ones(20))
    target = np.random.dirichlet(np.ones(20))

    turnover = calculator.calculate(current, target)

    print("Turnover :", turnover)

    # -------------------------------------------------------------------------
    # Test 3: Large Input
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Large Input")

    current = np.random.dirichlet(np.ones(500))
    target = np.random.dirichlet(np.ones(500))

    turnover = calculator.calculate(current, target)

    print("Turnover :", turnover)

    # -------------------------------------------------------------------------
    # Test 4: Invalid Shape
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Invalid Shape")

    try:
        calculator.calculate(
            np.array([[0.5, 0.5]]),
            np.array([0.5, 0.5]),
        )
    except Exception as exc:
        print("Expected Exception:", exc)

    # -------------------------------------------------------------------------
    # Test 5: Invalid Values
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Invalid Values")

    try:
        calculator.calculate(
            np.array([0.5, np.nan]),
            np.array([0.5, 0.5]),
        )
    except Exception as exc:
        print("Expected Exception:", exc)

    # -------------------------------------------------------------------------
    # Test 6: Edge Case
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Identical Allocations")

    current = np.array([0.25, 0.25, 0.25, 0.25])
    target = np.array([0.25, 0.25, 0.25, 0.25])

    turnover = calculator.calculate(current, target)

    print("Turnover :", turnover)

    # -------------------------------------------------------------------------
    # Test 7: Determinism
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Determinism")

    current = np.array([0.50, 0.30, 0.20])
    target = np.array([0.20, 0.50, 0.30])

    result1 = calculator.calculate(current, target)
    result2 = calculator.calculate(current, target)

    print("Deterministic :", np.isclose(result1, result2))

    # -------------------------------------------------------------------------
    # Test 8: Stress Test
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Stress Test")

    current = np.random.dirichlet(np.ones(5000))
    target = np.random.dirichlet(np.ones(5000))

    turnover = calculator.calculate(current, target)

    print("Turnover :", turnover)

    # -------------------------------------------------------------------------
    # Test 9: Numerical Robustness
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Numerical Robustness")

    current = np.array([
        0.500000000001,
        0.499999999999,
    ])

    target = np.array([
        0.5,
        0.5,
    ])

    turnover = calculator.calculate(current, target)

    print("Turnover :", turnover)

    # -------------------------------------------------------------------------
    # Test 10: Integration Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Integration Verification")

    current = np.random.dirichlet(np.ones(15))
    target = np.random.dirichlet(np.ones(15))

    turnover = calculator.calculate(current, target)
    
    if (
        np.isfinite(turnover)
        and 0.0 <= turnover <= 1.0
    ):
        print("Integration Passed")
    else:
        print("Integration Failed")

    logger.info("=" * 80)
    logger.info("All Turnover Runtime Tests Completed Successfully")
    logger.info("=" * 80)