"""
Module: comparison

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Provides a production-quality comparison framework for evaluating
    multiple classical portfolio optimization strategies.

    The module compares the exact Mean-Variance Optimization (MVO)
    solution obtained using the SciPy SLSQP optimizer against the
    heuristic Greedy Baseline solver. Both solutions are evaluated
    under the same mean-variance objective, allowing downstream
    evaluation modules to quantify approximation quality and
    optimization trade-offs.

Workflow:
    Expected Returns
            │
            ▼
    SciPy Mean-Variance Optimization
            │
            ├──────────────┐
            ▼              │
    Greedy Baseline        │
            │              │
            ▼              ▼
      Objective Evaluation
            │
            ▼
     Optimality Comparison
            │
            ▼
    Structured Comparison Results

Mathematical Objective:
    Compare multiple portfolio allocation strategies under a common
    mean-variance objective

        f(w) = λ wᵀΣw − μᵀw

    where

        μ : expected returns
        Σ : covariance matrix
        λ : risk-aversion coefficient

Design Philosophy:
    - Defensive programming
    - Deterministic behaviour
    - Numerical robustness
    - Backward compatibility
    - Production-quality logging
    - Modular reusable architecture

Author:
    ARQPO Research Project
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np

from greedy_baseline import GreedyBaselineSolver
from mean_variance import MeanVarianceObjective
from scipy_optimizer import SciPyOptimizer

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_RISK_AVERSION: float = 1.0
DEFAULT_TOP_K: int = 3

DEFAULT_OBJECTIVE_VALUE: float = 0.0

FLOAT_TOLERANCE: float = 1e-9

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


# =============================================================================
# Baseline Comparison
# =============================================================================

class BaselineComparer:
    """
    Compares multiple classical portfolio optimization strategies.

    The comparer executes both the exact SciPy mean-variance optimizer
    and the greedy heuristic baseline before evaluating each solution
    under the same optimization objective.

    This produces a consistent comparison that can be consumed by
    later validation, evaluation and dashboard modules.

    Attributes:
        risk_aversion:
            Mean-variance risk coefficient.

        top_k:
            Number of assets selected by the greedy baseline.

        scipy_solver:
            Exact classical optimizer.

        greedy_solver:
            Greedy heuristic optimizer.

        mvo_objective:
            Shared objective evaluator.
    """

    def __init__(
        self,
        risk_aversion: float = DEFAULT_RISK_AVERSION,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Initialize the baseline comparison engine.

        Args:
            risk_aversion:
                Risk-aversion coefficient.

            top_k:
                Number of assets selected by the greedy baseline.

        Raises:
            ValueError:
                If constructor parameters are invalid.
        """
        logger.info("Initializing BaselineComparer.")

        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.risk_aversion = float(risk_aversion)
        self.top_k = int(top_k)

        self.scipy_solver = SciPyOptimizer(
            risk_aversion=self.risk_aversion
        )

        self.greedy_solver = GreedyBaselineSolver(
            top_k=self.top_k
        )

        self.mvo_objective = MeanVarianceObjective(
            risk_aversion=self.risk_aversion
        )

        logger.info("BaselineComparer initialized successfully.")

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    @staticmethod
    def _validate_inputs(
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:
        """
        Validate comparison inputs.

        Args:
            expected_returns:
                Expected asset returns.

            covariance_matrix:
                Asset covariance matrix.

        Raises:
            TypeError:
                If inputs are not NumPy arrays.

            ValueError:
                If dimensions or numerical properties are invalid.
        """
        if not isinstance(expected_returns, np.ndarray):
            raise TypeError(
                "expected_returns must be a NumPy array."
            )

        if not isinstance(covariance_matrix, np.ndarray):
            raise TypeError(
                "covariance_matrix must be a NumPy array."
            )

        if expected_returns.ndim != 1:
            raise ValueError(
                "expected_returns must be one-dimensional."
            )

        if covariance_matrix.ndim != 2:
            raise ValueError(
                "covariance_matrix must be two-dimensional."
            )

        rows, cols = covariance_matrix.shape

        if rows != cols:
            raise ValueError(
                "covariance_matrix must be square."
            )

        if rows != expected_returns.size:
            raise ValueError(
                "Dimension mismatch between returns and covariance."
            )

        if expected_returns.size == 0:
            raise ValueError(
                "At least one asset is required."
            )

        if not np.all(np.isfinite(expected_returns)):
            raise ValueError(
                "expected_returns contains non-finite values."
            )

        if not np.all(np.isfinite(covariance_matrix)):
            raise ValueError(
                "covariance_matrix contains non-finite values."
            )

        if not np.allclose(
            covariance_matrix,
            covariance_matrix.T,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "covariance_matrix must be symmetric."
            )

    # =========================================================================
    # Numerical Utilities
    # =========================================================================

    @staticmethod
    def _cleanup_scalar(value: float) -> float:
        """
        Clean floating-point numerical artifacts.

        Args:
            value:
                Scalar value.

        Returns:
            Clean scalar.
        """
        value = float(value)

        if not np.isfinite(value):
            raise ValueError("Encountered non-finite value.")

        if np.isclose(value, 0.0, atol=FLOAT_TOLERANCE):
            return 0.0

        return value

    @staticmethod
    def _cleanup_array(values: np.ndarray) -> np.ndarray:
        """
        Clean floating-point artifacts in arrays.

        Args:
            values:
                Input array.

        Returns:
            Cleaned array.
        """
        cleaned = np.asarray(values, dtype=float).copy()

        if not np.all(np.isfinite(cleaned)):
            raise ValueError(
                "Array contains non-finite values."
            )

        cleaned[np.isclose(
            cleaned,
            0.0,
            atol=FLOAT_TOLERANCE,
        )] = 0.0

        return cleaned

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _compute_optimality_gap(
        self,
        scipy_objective: float,
        greedy_objective: float,
    ) -> float:
        """
        Compute the optimality gap between the greedy and SciPy solutions.

        Lower objective values indicate better portfolio quality.

        Args:
            scipy_objective:
                Objective value returned by the SciPy optimizer.

            greedy_objective:
                Objective value computed for the greedy allocation.

        Returns:
            Optimality gap.

        Raises:
            ValueError:
                If either objective value is non-finite.
        """
        scipy_objective = self._cleanup_scalar(scipy_objective)
        greedy_objective = self._cleanup_scalar(greedy_objective)

        return self._cleanup_scalar(
            greedy_objective - scipy_objective
        )

    def _determine_winner(
        self,
        scipy_objective: float,
        greedy_objective: float,
    ) -> str:
        """
        Determine which optimization strategy performed better.

        Lower objective values correspond to better portfolios.

        Args:
            scipy_objective:
                Objective value of the SciPy optimizer.

            greedy_objective:
                Objective value of the greedy baseline.

        Returns:
            Winner identifier.

            One of:
                - "scipy"
                - "greedy"
                - "tie"
        """
        if np.isclose(
            scipy_objective,
            greedy_objective,
            atol=FLOAT_TOLERANCE,
        ):
            return "tie"

        if scipy_objective < greedy_objective:
            return "scipy"

        return "greedy"

    def _build_results(
        self,
        scipy_weights: np.ndarray,
        greedy_weights: np.ndarray,
        scipy_metadata: Dict[str, Any],
        greedy_metadata: Dict[str, Any],
        scipy_objective: float,
        greedy_objective: float,
        runtime_seconds: float,
    ) -> Dict[str, Any]:
        """
        Construct the final comparison dictionary.

        Args:
            scipy_weights:
                SciPy allocation weights.

            greedy_weights:
                Greedy allocation weights.

            scipy_metadata:
                Metadata returned by SciPy.

            greedy_metadata:
                Metadata returned by Greedy.

            scipy_objective:
                SciPy objective value.

            greedy_objective:
                Greedy objective value.

            runtime_seconds:
                Total execution time.

        Returns:
            Structured comparison dictionary.
        """
        optimality_gap = self._compute_optimality_gap(
            scipy_objective,
            greedy_objective,
        )

        winner = self._determine_winner(
            scipy_objective,
            greedy_objective,
        )

        return {
            "scipy": {
                "weights": scipy_weights.tolist(),
                "objective_value": scipy_objective,
                "success": scipy_metadata.get("success"),
                "metadata": scipy_metadata,
            },
            "greedy": {
                "weights": greedy_weights.tolist(),
                "objective_value": greedy_objective,
                "top_k_selected": greedy_metadata.get(
                    "top_k_selected"
                ),
                "metadata": greedy_metadata,
            },
            "comparison": {
                "optimality_gap": optimality_gap,
                "winner": winner,
                "runtime_seconds": runtime_seconds,
            },
        }

    # =========================================================================
    # Public API
    # =========================================================================

    def compare(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare the exact classical optimizer against the greedy baseline.

        Both allocation strategies are evaluated using the same
        mean-variance objective, enabling a fair comparison of
        optimization quality.

        Args:
            expected_returns:
                One-dimensional expected return vector.

            covariance_matrix:
                Asset covariance matrix.

        Returns:
            Dictionary containing:

            - SciPy results
            - Greedy results
            - Comparison metrics

        Raises:
            RuntimeError:
                If comparison fails unexpectedly.
        """
        start_time = time.perf_counter()

        logger.info(
            "Starting classical baseline comparison."
        )

        try:
            self._validate_inputs(
                expected_returns,
                covariance_matrix,
            )

            scipy_weights, scipy_metadata = (
                self.scipy_solver.solve(
                    expected_returns,
                    covariance_matrix,
                )
            )

            scipy_weights = self._cleanup_array(
                scipy_weights
            )

            greedy_weights, greedy_metadata = (
                self.greedy_solver.solve(
                    expected_returns
                )
            )

            greedy_weights = self._cleanup_array(
                greedy_weights
            )

            scipy_objective = self._cleanup_scalar(
                scipy_metadata.get(
                    "objective_value",
                    DEFAULT_OBJECTIVE_VALUE,
                )
            )

            greedy_objective = (
                self.mvo_objective.calculate_objective(
                    weights=greedy_weights,
                    returns=expected_returns,
                    cov_matrix=covariance_matrix,
                )
            )

            greedy_objective = self._cleanup_scalar(
                greedy_objective
            )

            runtime_seconds = (
                time.perf_counter() - start_time
            )

            results = self._build_results(
                scipy_weights=scipy_weights,
                greedy_weights=greedy_weights,
                scipy_metadata=scipy_metadata,
                greedy_metadata=greedy_metadata,
                scipy_objective=scipy_objective,
                greedy_objective=greedy_objective,
                runtime_seconds=runtime_seconds,
            )

            logger.info(
                "Comparison completed successfully "
                "in %.6f seconds.",
                runtime_seconds,
            )

            return results

        except Exception as exc:
            logger.exception(
                "Classical baseline comparison failed."
            )
            raise RuntimeError(
                "Failed to compare classical baseline "
                "optimizers."
            ) from exc


# =============================================================================
# Runtime Tests
# =============================================================================

def _runtime_test_happy_path() -> None:
    """Verify normal comparison workflow."""
    logger.info("Running Happy Path test.")

    comparer = BaselineComparer()

    expected_returns = np.array(
        [0.08, 0.12, 0.05, 0.09]
    )

    covariance = np.array(
        [
            [0.10, 0.02, 0.01, 0.00],
            [0.02, 0.08, 0.02, 0.01],
            [0.01, 0.02, 0.09, 0.01],
            [0.00, 0.01, 0.01, 0.07],
        ]
    )

    results = comparer.compare(
        expected_returns,
        covariance,
    )

    assert "scipy" in results
    assert "greedy" in results
    assert "comparison" in results

    logger.info("Happy Path test passed.")


def _runtime_test_medium_input() -> None:
    """Verify comparison with medium-sized portfolio."""
    logger.info("Running Medium Input test.")

    np.random.seed(42)

    n_assets = 20

    expected_returns = np.random.uniform(
        0.02,
        0.20,
        size=n_assets,
    )

    matrix = np.random.rand(
        n_assets,
        n_assets,
    )

    covariance = matrix @ matrix.T

    comparer = BaselineComparer(top_k=5)

    results = comparer.compare(
        expected_returns,
        covariance,
    )

    assert len(results["scipy"]["weights"]) == n_assets

    logger.info("Medium Input test passed.")


def _runtime_test_large_input() -> None:
    """Verify comparison on a larger portfolio."""
    logger.info("Running Large Input test.")

    np.random.seed(123)

    n_assets = 100

    expected_returns = np.random.uniform(
        0.01,
        0.15,
        size=n_assets,
    )

    matrix = np.random.rand(
        n_assets,
        n_assets,
    )

    covariance = matrix @ matrix.T

    comparer = BaselineComparer(top_k=10)

    results = comparer.compare(
        expected_returns,
        covariance,
    )

    assert len(results["greedy"]["weights"]) == n_assets

    logger.info("Large Input test passed.")


def _runtime_test_invalid_shape() -> None:
    """Verify invalid dimensions raise an error."""
    logger.info("Running Invalid Shape test.")

    comparer = BaselineComparer()

    expected_returns = np.array(
        [0.10, 0.12, 0.08]
    )

    covariance = np.eye(4)

    try:
        comparer.compare(
            expected_returns,
            covariance,
        )
    except RuntimeError:
        logger.info("Invalid Shape test passed.")
        return

    raise AssertionError(
        "Expected RuntimeError."
    )


def _runtime_test_invalid_values() -> None:
    """Verify NaN values are rejected."""
    logger.info("Running Invalid Values test.")

    comparer = BaselineComparer()

    expected_returns = np.array(
        [0.10, np.nan, 0.05]
    )

    covariance = np.eye(3)

    try:
        comparer.compare(
            expected_returns,
            covariance,
        )
    except RuntimeError:
        logger.info("Invalid Values test passed.")
        return

    raise AssertionError(
        "Expected RuntimeError."
    )


def _runtime_test_edge_case() -> None:
    """Verify minimum supported portfolio."""
    logger.info("Running Edge Case test.")

    comparer = BaselineComparer(top_k=1)

    expected_returns = np.array([0.10])

    covariance = np.array([[0.05]])

    results = comparer.compare(
        expected_returns,
        covariance,
    )

    assert len(results["scipy"]["weights"]) == 1

    logger.info("Edge Case test passed.")


def _runtime_test_determinism() -> None:
    """Verify deterministic outputs."""
    logger.info("Running Determinism test.")

    expected_returns = np.array(
        [0.08, 0.12, 0.05, 0.09]
    )

    covariance = np.array(
        [
            [0.10, 0.02, 0.01, 0.00],
            [0.02, 0.08, 0.02, 0.01],
            [0.01, 0.02, 0.09, 0.01],
            [0.00, 0.01, 0.01, 0.07],
        ]
    )

    comparer = BaselineComparer()

    result_one = comparer.compare(
        expected_returns,
        covariance,
    )

    result_two = comparer.compare(
        expected_returns,
        covariance,
    )

    assert np.allclose(
        result_one["scipy"]["weights"],
        result_two["scipy"]["weights"],
    )

    logger.info("Determinism test passed.")


def _runtime_test_stress() -> None:
    """Repeated execution stability."""
    logger.info("Running Stress test.")

    comparer = BaselineComparer()

    expected_returns = np.array(
        [0.10, 0.09, 0.11, 0.07]
    )

    covariance = np.eye(4)

    for _ in range(50):
        comparer.compare(
            expected_returns,
            covariance,
        )

    logger.info("Stress test passed.")


def _runtime_test_numerical_robustness() -> None:
    """Verify tiny numerical values remain stable."""
    logger.info("Running Numerical Robustness test.")

    comparer = BaselineComparer()

    expected_returns = np.array(
        [
            1e-12,
            2e-12,
            3e-12,
        ]
    )

    covariance = np.eye(3) * 1e-12

    results = comparer.compare(
        expected_returns,
        covariance,
    )

    assert np.isfinite(
        results["comparison"]["optimality_gap"]
    )

    logger.info(
        "Numerical Robustness test passed."
    )


def _runtime_test_integration() -> None:
    """Verify integration with dependent modules."""
    logger.info("Running Integration test.")

    comparer = BaselineComparer()

    assert isinstance(
        comparer.scipy_solver,
        SciPyOptimizer,
    )

    assert isinstance(
        comparer.greedy_solver,
        GreedyBaselineSolver,
    )

    assert isinstance(
        comparer.mvo_objective,
        MeanVarianceObjective,
    )

    logger.info("Integration test passed.")


# =============================================================================
# Runtime Execution
# =============================================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Starting comparison.py runtime tests")
    logger.info("=" * 70)

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

    logger.info("=" * 70)
    logger.info("All runtime tests passed successfully.")
    logger.info("=" * 70)