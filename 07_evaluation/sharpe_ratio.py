"""
ARQPO - Adaptive Regime-Aware Quantum Portfolio Optimizer
Module: 07_evaluation.sharpe_ratio

Purpose:
    Provides a production-grade implementation of the Sharpe Ratio for
    evaluating the risk-adjusted performance of portfolio allocations.
    This module serves as the canonical Sharpe Ratio evaluator for both
    classical and quantum portfolio optimization algorithms within ARQPO.

Workflow Position:
    Portfolio Allocation
            ↓
    Portfolio Statistics
            ↓
    Sharpe Ratio Evaluation
            ↓
    Benchmark Comparison
            ↓
    Explainability
            ↓
    Streamlit Dashboard

Mathematical Objective:
    Sharpe Ratio = (Portfolio Return − Risk-Free Rate)
                   / Portfolio Volatility

Design Philosophy:
    - Defensive programming
    - Numerical robustness
    - Stable public API
    - Deterministic computation
    - Production-quality logging
    - Reusable evaluation component
"""

from __future__ import annotations

import logging
import time
from typing import Final

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE: Final[float] = 1e-8
DEFAULT_RISK_FREE_RATE: Final[float] = 0.02

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


class SharpeRatioCalculator:
    """
    Computes the Sharpe Ratio for a portfolio allocation.

    The Sharpe Ratio measures excess return generated per unit
    of total portfolio risk.

    Attributes:
        risk_free_rate:
            Annualized risk-free rate used when computing
            excess portfolio return.
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> None:
        """
        Initialize the Sharpe Ratio calculator.

        Args:
            risk_free_rate:
                Annualized risk-free return.

        Raises:
            TypeError:
                If the risk-free rate is not numeric.

            ValueError:
                If the value is NaN or infinite.
        """
        logger.info("Initializing SharpeRatioCalculator.")

        if not isinstance(risk_free_rate, (int, float, np.floating)):
            raise TypeError(
                "risk_free_rate must be a numeric value."
            )

        if not np.isfinite(risk_free_rate):
            raise ValueError(
                "risk_free_rate must be finite."
            )

        self.risk_free_rate = float(risk_free_rate)

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_weights(
        self,
        weights: np.ndarray,
    ) -> None:
        """
        Validate portfolio weights.

        Args:
            weights:
                Portfolio allocation vector.

        Raises:
            TypeError:
                If input is not a NumPy array.

            ValueError:
                If weights contain invalid values.
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                "weights must be a NumPy array."
            )

        if weights.ndim != 1:
            raise ValueError(
                "weights must be one-dimensional."
            )

        if weights.size == 0:
            raise ValueError(
                "weights cannot be empty."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "weights contain NaN or infinite values."
            )

    def _validate_expected_returns(
        self,
        expected_returns: np.ndarray,
    ) -> None:
        """
        Validate expected asset returns.

        Args:
            expected_returns:
                Expected return vector.

        Raises:
            TypeError:
                If input is not a NumPy array.

            ValueError:
                If the vector is invalid.
        """
        if not isinstance(expected_returns, np.ndarray):
            raise TypeError(
                "expected_returns must be a NumPy array."
            )

        if expected_returns.ndim != 1:
            raise ValueError(
                "expected_returns must be one-dimensional."
            )

        if expected_returns.size == 0:
            raise ValueError(
                "expected_returns cannot be empty."
            )

        if not np.all(np.isfinite(expected_returns)):
            raise ValueError(
                "expected_returns contain NaN or infinite values."
            )

    def _validate_covariance_matrix(
        self,
        cov_matrix: np.ndarray,
    ) -> None:
        """
        Validate covariance matrix.

        Args:
            cov_matrix:
                Covariance matrix.

        Raises:
            TypeError:
                If the matrix is not a NumPy array.

            ValueError:
                If the matrix is invalid.
        """
        if not isinstance(cov_matrix, np.ndarray):
            raise TypeError(
                "cov_matrix must be a NumPy array."
            )

        if cov_matrix.ndim != 2:
            raise ValueError(
                "cov_matrix must be two-dimensional."
            )

        rows, cols = cov_matrix.shape

        if rows != cols:
            raise ValueError(
                "cov_matrix must be square."
            )

        if not np.all(np.isfinite(cov_matrix)):
            raise ValueError(
                "cov_matrix contains NaN or infinite values."
            )

        if not np.allclose(
            cov_matrix,
            cov_matrix.T,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "cov_matrix must be symmetric."
            )
        
        if np.any(np.diag(cov_matrix) < 0.0):
            raise ValueError(
                "Covariance matrix diagonal entries "
                "must be non-negative."
            )
        
            

    def _validate_inputs(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> None:
        """
        Validate all calculation inputs.

        Raises:
            ValueError:
                If array dimensions are incompatible.
        """
        logger.info("Validating Sharpe Ratio inputs.")

        self._validate_weights(weights)
        self._validate_expected_returns(expected_returns)
        self._validate_covariance_matrix(cov_matrix)

        asset_count = weights.size

        if expected_returns.size != asset_count:
            raise ValueError(
                "weights and expected_returns "
                "must have identical length."
            )

        if cov_matrix.shape != (
            asset_count,
            asset_count,
        ):
            raise ValueError(
                "cov_matrix dimensions do not match "
                "the number of assets."
            )
        if not np.isclose(
            np.sum(weights),
            1.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "Portfolio weights must sum to 1."
            )    

    # =========================================================================
    # Numerical Helper Methods
    # =========================================================================

    @staticmethod
    def _cleanup_scalar(value: float) -> float:
        """
        Clean numerical artifacts from a scalar value.

        Args:
            value:
                Scalar value to clean.

        Returns:
            Cleaned scalar value.
        """
        if np.isclose(value, 0.0, atol=FLOAT_TOLERANCE):
            return 0.0

        return float(value)

    @staticmethod
    def _cleanup_array(array: np.ndarray) -> np.ndarray:
        """
        Remove insignificant floating-point artifacts.

        Args:
            array:
                Input NumPy array.

        Returns:
            Cleaned NumPy array.
        """
        cleaned = np.asarray(array, dtype=float).copy()

        cleaned[np.isclose(cleaned, 0.0, atol=FLOAT_TOLERANCE)] = 0.0

        return cleaned

    # =========================================================================
    # Public API
    # =========================================================================

    def calculate(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """
        Calculate the portfolio Sharpe Ratio.

        Args:
            weights:
                Portfolio allocation vector.

            expected_returns:
                Expected asset returns.

            cov_matrix:
                Asset covariance matrix.

        Returns:
            Portfolio Sharpe Ratio.

        Raises:
            RuntimeError:
                If the computation fails unexpectedly.

        Notes:
            The public API intentionally returns only the Sharpe Ratio
            to preserve backward compatibility with downstream modules.
        """
        start_time = time.perf_counter()

        logger.info("Starting Sharpe Ratio calculation.")

        try:
            self._validate_inputs(
                weights,
                expected_returns,
                cov_matrix,
            )

            weights = self._cleanup_array(weights)
            expected_returns = self._cleanup_array(expected_returns)
            cov_matrix = self._cleanup_array(cov_matrix)

            portfolio_return = self._cleanup_scalar(
                float(np.dot(weights, expected_returns))
            )

            portfolio_variance = self._cleanup_scalar(
                float(weights.T @ cov_matrix @ weights)
            )

            if (
                portfolio_variance < 0.0
                and np.isclose(
                    portfolio_variance,
                    0.0,
                    atol=FLOAT_TOLERANCE,
                )
            ):
                portfolio_variance = 0.0

            if portfolio_variance <= FLOAT_TOLERANCE:
                logger.warning(
                    "Portfolio variance is effectively zero. "
                    "Returning Sharpe Ratio of 0.0."
                )
                return 0.0

            portfolio_volatility = self._cleanup_scalar(
                float(np.sqrt(portfolio_variance))
            )

            sharpe_ratio = self._cleanup_scalar(
                (
                    portfolio_return
                    - self.risk_free_rate
                )
                / portfolio_volatility
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "Sharpe Ratio calculation completed successfully."
            )

            logger.info(
                "Return=%.6f | "
                "Volatility=%.6f | "
                "RiskFreeRate=%.6f | "
                "Sharpe=%.6f | "
                "ExecutionTime=%.6fs",
                portfolio_return,
                portfolio_volatility,
                self.risk_free_rate,
                sharpe_ratio,
                elapsed,
            )

            return sharpe_ratio
        
        except (TypeError, ValueError):
            raise


        except Exception as exc:
            logger.exception(
                "unexcepted error during Sharpe Ratio calculation failed."
            )

            raise RuntimeError(
                "Failed to calculate the portfolio Sharpe Ratio."
            ) from exc

# =============================================================================
# Runtime Tests
# =============================================================================

def _runtime_test_happy_path() -> None:
    """Verify Sharpe Ratio calculation for a valid portfolio."""
    calculator = SharpeRatioCalculator()

    weights = np.array([0.4, 0.3, 0.3])
    expected_returns = np.array([0.12, 0.08, 0.10])
    covariance = np.array(
        [
            [0.040, 0.010, 0.005],
            [0.010, 0.030, 0.004],
            [0.005, 0.004, 0.020],
        ]
    )

    result = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert isinstance(result, float)
    print("✓ Happy Path Test Passed")


def _runtime_test_medium_input() -> None:
    """Verify calculation with a medium-sized portfolio."""
    calculator = SharpeRatioCalculator()

    n_assets = 20

    weights = np.full(n_assets, 1.0 / n_assets)
    expected_returns = np.linspace(0.05, 0.15, n_assets)
    covariance = np.eye(n_assets) * 0.03

    result = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert np.isfinite(result)
    print("✓ Medium Input Test Passed")


def _runtime_test_large_input() -> None:
    """Verify calculation with a large portfolio."""
    calculator = SharpeRatioCalculator()

    n_assets = 200

    weights = np.full(n_assets, 1.0 / n_assets)
    expected_returns = np.linspace(0.04, 0.16, n_assets)
    covariance = np.eye(n_assets) * 0.02

    result = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert np.isfinite(result)
    print("✓ Large Input Test Passed")


def _runtime_test_invalid_shape() -> None:
    """Verify mismatched dimensions are rejected."""
    calculator = SharpeRatioCalculator()

    try:
        calculator.calculate(
            np.array([0.5, 0.5]),
            np.array([0.10]),
            np.eye(2),
        )
    except ValueError:
        print("✓ Invalid Shape Test Passed")
        return

    raise AssertionError("Invalid shape was accepted.")


def _runtime_test_invalid_values() -> None:
    """Verify NaN values are rejected."""
    calculator = SharpeRatioCalculator()

    try:
        calculator.calculate(
            np.array([0.5, np.nan]),
            np.array([0.10, 0.08]),
            np.eye(2),
        )
    except ValueError:
        print("✓ Invalid Values Test Passed")
        return

    raise AssertionError("NaN values were accepted.")


def _runtime_test_edge_case() -> None:
    """Verify zero-variance portfolios return zero."""
    calculator = SharpeRatioCalculator()

    weights = np.array([1.0])
    expected_returns = np.array([0.10])
    covariance = np.array([[0.0]])

    result = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert np.isclose(
        result,
        0.0,
        atol=FLOAT_TOLERANCE,
    )
    

    print("✓ Edge Case Test Passed")


def _runtime_test_determinism() -> None:
    """Verify deterministic outputs."""
    calculator = SharpeRatioCalculator()

    weights = np.array([0.4, 0.6])
    expected_returns = np.array([0.12, 0.08])
    covariance = np.eye(2) * 0.03

    first = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    second = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert np.isclose(first, second)

    print("✓ Determinism Test Passed")


def _runtime_test_stress() -> None:
    """Repeated execution stress test."""
    calculator = SharpeRatioCalculator()

    weights = np.array([0.4, 0.3, 0.3])
    expected_returns = np.array([0.10, 0.08, 0.12])
    covariance = np.eye(3) * 0.02
    
    iterations = 1000
    previous_level = logger.level
    logger.setLevel(logging.WARNING)

    try:
        for _ in range(iterations):
            calculator.calculate(
                weights,
                expected_returns,
                covariance,
            )
    finally:
        logger.setLevel(previous_level)
    

    
    print("✓ Stress Test Passed")


def _runtime_test_numerical_robustness() -> None:
    """Verify very small numerical values are handled safely."""
    calculator = SharpeRatioCalculator()

    weights = np.array([0.5, 0.5])
    expected_returns = np.array([1e-12, 1e-12])
    covariance = np.eye(2) * 1e-12

    result = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert np.isfinite(result)

    print("✓ Numerical Robustness Test Passed")


def _runtime_test_integration() -> None:
    """Basic integration verification."""
    calculator = SharpeRatioCalculator()

    weights = np.array([0.5, 0.5])
    expected_returns = np.array([0.15, 0.09])
    covariance = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.03],
        ]
    )

    result = calculator.calculate(
        weights,
        expected_returns,
        covariance,
    )

    assert isinstance(result, float)

    print("✓ Integration Verification Test Passed")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Running Sharpe Ratio Runtime Tests")
    print("=" * 60)

    _runtime_test_happy_path()
    _runtime_test_medium_input()
    _runtime_test_large_input()
    _runtime_test_invalid_shape()
    _runtime_test_invalid_values()
    _runtime_test_edge_case()
    _runtime_test_determinism()
    _runtime_test_stress()
    _runtime_test_numerical_robustness()
    _runtime_test_integration()

    print("=" * 60)
    print("All runtime tests passed successfully.")
    print("=" * 60)