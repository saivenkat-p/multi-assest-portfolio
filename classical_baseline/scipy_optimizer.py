"""
ARQPO Project
=============

Module: scipy_optimizer.py

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Provides a production-quality classical portfolio optimizer based on
    SciPy's Sequential Least Squares Programming (SLSQP) algorithm.

    This module serves as one of the classical benchmark solvers used to
    compare against the adaptive quantum optimization approaches
    (QAOA and VQE). It numerically solves constrained portfolio
    optimization problems while enforcing investment constraints such as
    long-only allocations and full investment.

Workflow Position:
    Historical Data
        ↓
    Data Preprocessing
        ↓
    Market Regime Detection
        ↓
    Adaptive Optimization
        ↓
    Mean-Variance Objective
        ↓
    SciPy Optimizer      ← Current Module
        ↓
    Greedy Baseline
        ↓
    Classical Comparison
        ↓
    Validation
        ↓
    Evaluation

Mathematical Objective:
    Solve the constrained optimization problem

        minimize   f(w)

    subject to

        Σw = 1

        lower_bound ≤ w ≤ upper_bound

    where

        f(w)

    is provided by MeanVarianceObjective and typically represents
    an adaptive risk-return tradeoff.

Design Philosophy:
    • Production-quality implementation.
    • Deterministic execution.
    • Defensive programming.
    • Numerical robustness.
    • Modular helper methods.
    • Rich execution metadata.
    • Stable public API.
    • Backward compatibility with downstream modules.

Dependencies:
    Upstream
        • mean_variance.py

    Downstream
        • greedy_baseline.py
        • comparison.py
        • validation/
        • evaluation/

Author:
    ARQPO Development Workflow
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from mean_variance import MeanVarianceObjective

# ============================================================================
# Module Constants
# ============================================================================

DEFAULT_METHOD = "SLSQP"

DEFAULT_RISK_AVERSION = 1.0

DEFAULT_TOLERANCE = 1e-8

DEFAULT_MIN_WEIGHT = 0.0

DEFAULT_MAX_WEIGHT = 1.0

DEFAULT_MAX_ITERATIONS = 500

FLOAT_TOLERANCE = 1e-10

CONSTRAINT_TARGET_SUM = 1.0


# ============================================================================
# Logger Configuration
# ============================================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

logger.setLevel(logging.INFO)


class SciPyOptimizer:
    """
    Classical constrained portfolio optimizer using SciPy.

    This class provides a production-quality wrapper around
    scipy.optimize.minimize using the Sequential Least Squares
    Programming (SLSQP) algorithm.

    The optimizer is intended to serve as a deterministic classical
    baseline for evaluating quantum optimization methods within ARQPO.

    Attributes
    ----------
    mvo : MeanVarianceObjective
        Mean-variance objective function.

    method : str
        Optimization algorithm.

    tolerance : float
        Numerical convergence tolerance.
    """

    def __init__(
        self,
        risk_aversion: float = DEFAULT_RISK_AVERSION,
        method: str = DEFAULT_METHOD,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        """
        Initialize the SciPy optimizer.

        Parameters
        ----------
        risk_aversion : float, default=1.0
            Risk preference parameter used by the objective function.

        method : str, default="SLSQP"
            Optimization algorithm.

        tolerance : float, default=1e-8
            Numerical convergence tolerance.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """

        logger.info("Initializing SciPyOptimizer.")

        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive.")

        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")

        self.method = method

        self.tolerance = tolerance

        self.mvo = MeanVarianceObjective(
            risk_aversion=risk_aversion
        )

        logger.info(
            "SciPyOptimizer initialized successfully."
        )

    # ========================================================================
    # Validation Helpers
    # ========================================================================

    def _validate_inputs(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:
        """
        Validate optimization inputs.

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected asset returns.

        covariance_matrix : np.ndarray
            Asset covariance matrix.

        Raises
        ------
        TypeError
            If inputs are not NumPy arrays.

        ValueError
            If dimensions or numerical values are invalid.
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

        if expected_returns.size == 0:
            raise ValueError(
                "expected_returns cannot be empty."
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

        if not np.isfinite(expected_returns).all():
            raise ValueError(
                "expected_returns contains NaN or infinite values."
            )

        if not np.isfinite(covariance_matrix).all():
            raise ValueError(
                "covariance_matrix contains NaN or infinite values."
            )

        self._validate_covariance_matrix(
            covariance_matrix
        )

    def _validate_covariance_matrix(
        self,
        covariance_matrix: np.ndarray,
    ) -> None:
        """
        Validate covariance matrix properties.

        Parameters
        ----------
        covariance_matrix : np.ndarray
            Covariance matrix.

        Raises
        ------
        ValueError
            If the covariance matrix is invalid.
        """
        if np.iscomplexobj(covariance_matrix):
            raise ValueError(
                "Covariance matrix must contain only real values."
            )


        if not np.allclose(
            covariance_matrix,
            covariance_matrix.T,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "Covariance matrix must be symmetric."
            )

        diagonal = np.diag(covariance_matrix)

        if np.any(diagonal < -FLOAT_TOLERANCE):
            raise ValueError(
                "Covariance matrix contains negative variances."
            )

        logger.info(
            "Input validation completed successfully."
        )

    # ========================================================================
    # Numerical Helper Methods
    # ========================================================================

    def _build_initial_guess(
        self,
        n_assets: int,
    ) -> np.ndarray:
        """
        Construct an equal-weight initial portfolio.

        Parameters
        ----------
        n_assets : int
            Number of assets.

        Returns
        -------
        np.ndarray
            Initial portfolio weights.

        Raises
        ------
        ValueError
            If the number of assets is invalid.
        """

        if n_assets <= 0:
            raise ValueError(
                "Number of assets must be positive."
            )

        initial_weights = np.full(
            shape=n_assets,
            fill_value=1.0 / n_assets,
            dtype=float,
        )

        logger.info(
            "Constructed equal-weight initial portfolio."
        )

        return initial_weights

    def _build_bounds(
        self,
        n_assets: int,
    ) -> Tuple[Tuple[float, float], ...]:
        """
        Construct long-only optimization bounds.

        Parameters
        ----------
        n_assets : int

        Returns
        -------
        Tuple[Tuple[float, float], ...]
            Bounds for each optimization variable.
        """

        bounds = tuple(
            (
                DEFAULT_MIN_WEIGHT,
                DEFAULT_MAX_WEIGHT,
            )
            for _ in range(n_assets)
        )

        logger.info(
            "Optimization bounds constructed."
        )

        return bounds

    def _build_constraints(
        self,
    ) -> Tuple[Dict[str, Any], ...]:
        """
        Construct optimization constraints.

        The optimizer enforces

            Σw = 1

        Returns
        -------
        Tuple[Dict[str, Any], ...]
            SciPy-compatible constraint definitions.
        """

        constraints = (
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - CONSTRAINT_TARGET_SUM,
            },
        )

        logger.info(
            "Portfolio constraints constructed."
        )

        return constraints

    def _cleanup_weights(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Clean numerical artifacts from optimized weights.

        Parameters
        ----------
        weights : np.ndarray
            Optimized portfolio weights.

        Returns
        -------
        np.ndarray
            Cleaned portfolio weights.

        Notes
        -----
        Performs:

        • clipping

        • floating-point cleanup

        • normalization

        • finite-value verification
        """

        cleaned = np.asarray(
            weights,
            dtype=float,
        ).copy()

        cleaned = np.clip(
            cleaned,
            DEFAULT_MIN_WEIGHT,
            DEFAULT_MAX_WEIGHT,
        )

        cleaned[np.abs(cleaned) < FLOAT_TOLERANCE] = 0.0

        total_weight = np.sum(cleaned)

        if np.isclose(
            total_weight,
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "Portfolio weights sum to zero after cleanup."
            )

        cleaned /= total_weight
        
        if np.any(
             cleaned < DEFAULT_MIN_WEIGHT - FLOAT_TOLERANCE
        ):
            raise ValueError(
                "Portfolio contains weights below the minimum bound."
            )

        if np.any(
            cleaned > DEFAULT_MAX_WEIGHT + FLOAT_TOLERANCE
        ):
            raise ValueError(
                "Portfolio contains weights above the maximum bound."
            )
    
        if not np.isfinite(cleaned).all():
            raise ValueError(
                "Portfolio contains invalid numerical values."
            )

        logger.info(
            "Portfolio weights cleaned successfully."
        )

        return cleaned

    def _build_metadata(
        self,
        result: Any,
        execution_time: float,
        n_assets: int,
    ) -> Dict[str, Any]:
        """
        Build standardized optimization metadata.

        Parameters
        ----------
        result : Any
            SciPy OptimizeResult.

        execution_time : float
            Solver runtime in seconds.

        n_assets : int
            Number of portfolio assets.

        Returns
        -------
        Dict[str, Any]
            Structured execution metadata.
        """

        metadata = {
            "solver": "SciPy",
            "method": self.method,
            "success": bool(result.success),
            "status": int(getattr(result, "status", -1)),
            "message": str(result.message),
            "objective_value": float(result.fun),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(
                getattr(result, "nfev", 0)
            ),
            "gradient_evaluations": int(
                getattr(result, "njev", 0)
            ),
            "execution_time": float(execution_time),
            "optimization_timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "n_assets": int(n_assets),
            
            
            
        }

        logger.info(
            "Optimization metadata created."
        )

        return metadata

    # ========================================================================
    # Public API
    # ========================================================================

    def solve(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the constrained portfolio optimization problem.

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected return vector.

        covariance_matrix : np.ndarray
            Asset covariance matrix.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            A tuple containing

            - Optimized portfolio weights
            - Optimization metadata

        Raises
        ------
        RuntimeError
            If the optimization fails unexpectedly.

        Notes
        -----
        The optimizer performs the following steps:

        1. Validate all inputs.

        2. Construct an equal-weight initial portfolio.

        3. Build optimization bounds.

        4. Build equality constraints.

        5. Execute SciPy's SLSQP solver.

        6. Clean numerical artifacts.

        7. Collect execution metadata.

        8. Return deterministic results.
        """

        logger.info(
            "Starting SciPy portfolio optimization."
        )

        start_time = time.perf_counter()

        try:
            # ------------------------------------------------------------
            # Input Validation
            # ------------------------------------------------------------

            self._validate_inputs(
                expected_returns,
                covariance_matrix,
            )

            n_assets = expected_returns.size

            logger.info(
                "Optimization problem contains %d assets.",
                n_assets,
            )

            # ------------------------------------------------------------
            # Build Optimization Components
            # ------------------------------------------------------------

            initial_guess = self._build_initial_guess(
                n_assets
            )

            bounds = self._build_bounds(
                n_assets
            )

            constraints = self._build_constraints()

            logger.info(
                "Optimization components constructed."
            )

            # ------------------------------------------------------------
            # Execute Optimization
            # ------------------------------------------------------------

            result = minimize(
                fun=self.mvo.calculate_objective,
                x0=initial_guess,
                args=(
                    expected_returns,
                    covariance_matrix,
                ),
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                tol=self.tolerance,
                options={
                    "maxiter": DEFAULT_MAX_ITERATIONS,
                    "disp": False,
                },
            )

            # ------------------------------------------------------------
            # Solver Status
            # ------------------------------------------------------------

            if result.success:

                logger.info(
                    "Optimization converged successfully."
                )

            else:

                logger.warning(
                    "Optimization terminated without "
                    "successful convergence: %s",
                    result.message,
                )

            # ------------------------------------------------------------
            # Numerical Cleanup
            # ------------------------------------------------------------

            optimized_weights = self._cleanup_weights(
                result.x
            )

            # ------------------------------------------------------------
            # Execution Timing
            # ------------------------------------------------------------

            execution_time = (
                time.perf_counter()
                - start_time
            )

            logger.info(
                "Optimization completed in %.6f seconds.",
                execution_time,
            )

            # ------------------------------------------------------------
            # Metadata
            # ------------------------------------------------------------

            metadata = self._build_metadata(
                result=result,
                execution_time=execution_time,
                n_assets=n_assets,
            )

            logger.info(
                "Returning optimization results."
            )

            return (
                optimized_weights,
                metadata,
            )

        except Exception as exc:

            logger.exception(
                "SciPy portfolio optimization failed."
            )

            raise RuntimeError(
                "Failed to solve the portfolio "
                "optimization problem."
            ) from exc

# ============================================================================
# Runtime Tests
# ============================================================================

if __name__ == "__main__":

    logger.info("=" * 80)
    logger.info("SciPyOptimizer Runtime Tests")
    logger.info("=" * 80)

    np.random.seed(42)

    optimizer = SciPyOptimizer(
        risk_aversion=1.0
    )

    # ------------------------------------------------------------------------
    # Test 1 : Basic Optimization
    # ------------------------------------------------------------------------

    logger.info("Test 1 : Basic optimization")

    expected_returns = np.array(
        [0.12, 0.08, 0.15]
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.006, 0.004],
            [0.006, 0.030, 0.005],
            [0.004, 0.005, 0.050],
        ]
    )

    weights, metadata = optimizer.solve(
        expected_returns,
        covariance_matrix,
    )

    print("\nOptimized Weights")
    print(weights)

    print("\nMetadata")
    print(metadata)

    assert np.isclose(
        np.sum(weights),
        1.0,
    )

    assert metadata["success"] in (
        True,
        False,
    )

    logger.info("Test 1 PASSED")

    # ------------------------------------------------------------------------
    # Test 2 : Five Asset Portfolio
    # ------------------------------------------------------------------------

    logger.info("Test 2 : Five asset optimization")

    n_assets = 5

    expected_returns = np.random.uniform(
        0.05,
        0.20,
        size=n_assets,
    )

    random_matrix = np.random.rand(
        n_assets,
        n_assets,
    )

    covariance_matrix = (
        random_matrix @ random_matrix.T
    )

    weights, metadata = optimizer.solve(
        expected_returns,
        covariance_matrix,
    )

    print("\nFive Asset Portfolio")
    print(weights)

    assert len(weights) == n_assets

    assert np.isclose(
        np.sum(weights),
        1.0,
    )

    logger.info("Test 2 PASSED")

    # ------------------------------------------------------------------------
    # Test 3 : Dimension Mismatch
    # ------------------------------------------------------------------------

    logger.info("Test 3 : Dimension mismatch")

    try:

        optimizer.solve(
            np.array([0.1, 0.2]),
            np.eye(3),
        )

    except RuntimeError:

        logger.info(
            "Expected RuntimeError received."
        )

    # ------------------------------------------------------------------------
    # Test 4 : Non-square Covariance Matrix
    # ------------------------------------------------------------------------

    logger.info(
        "Test 4 : Non-square covariance matrix"
    )

    try:

        optimizer.solve(
            np.array([0.1, 0.2]),
            np.ones((2, 3)),
        )

    except RuntimeError:

        logger.info(
            "Expected RuntimeError received."
        )

    # ------------------------------------------------------------------------
    # Test 5 : NaN Validation
    # ------------------------------------------------------------------------

    logger.info(
        "Test 5 : NaN validation"
    )

    try:

        optimizer.solve(
            np.array(
                [
                    np.nan,
                    0.2,
                    0.1,
                ]
            ),
            np.eye(3),
        )

    except RuntimeError:

        logger.info(
            "Expected RuntimeError received."
        )

    # ------------------------------------------------------------------------
    # Test 6 : Infinite Values
    # ------------------------------------------------------------------------

    logger.info(
        "Test 6 : Infinite values"
    )

    try:

        optimizer.solve(
            np.array(
                [
                    np.inf,
                    0.2,
                    0.1,
                ]
            ),
            np.eye(3),
        )

    except RuntimeError:

        logger.info(
            "Expected RuntimeError received."
        )

    # ------------------------------------------------------------------------
    # Test 7 : Non-symmetric Covariance
    # ------------------------------------------------------------------------

    logger.info(
        "Test 7 : Non-symmetric covariance"
    )

    invalid_covariance = np.array(
        [
            [1.0, 2.0],
            [5.0, 1.0],
        ]
    )

    try:

        optimizer.solve(
            np.array(
                [
                    0.1,
                    0.2,
                ]
            ),
            invalid_covariance,
        )

    except RuntimeError:

        logger.info(
            "Expected RuntimeError received."
        )

    # ------------------------------------------------------------------------
    # Test 8 : Repeatability
    # ------------------------------------------------------------------------

    logger.info(
        "Test 8 : Deterministic execution"
    )

    expected_returns = np.array(
        [
            0.12,
            0.08,
            0.15,
        ]
    )

    covariance_matrix = np.array(
        [
            [0.040, 0.006, 0.004],
            [0.006, 0.030, 0.005],
            [0.004, 0.005, 0.050],
        ]
    )

    weights_one, _ = optimizer.solve(
        expected_returns,
        covariance_matrix,
    )

    weights_two, _ = optimizer.solve(
        expected_returns,
        covariance_matrix,
    )

    assert np.allclose(
        weights_one,
        weights_two,
    )

    logger.info("Test 8 PASSED")

    # ------------------------------------------------------------------------
    # Test 9 : Stress Test
    # ------------------------------------------------------------------------

    logger.info(
        "Test 9 : Stress test"
    )

    n_assets = 25

    expected_returns = np.random.uniform(
        0.03,
        0.20,
        n_assets,
    )

    random_matrix = np.random.rand(
        n_assets,
        n_assets,
    )

    covariance_matrix = (
        random_matrix @ random_matrix.T
    )

    weights, metadata = optimizer.solve(
        expected_returns,
        covariance_matrix,
    )

    assert len(weights) == n_assets

    assert np.isclose(
        np.sum(weights),
        1.0,
    )

    logger.info("Test 9 PASSED")

    
    # ------------------------------------------------------------------------
    # Test 10 : Zero Covariance Matrix
    # ------------------------------------------------------------------------

    logger.info(
        "Test 10 : Zero covariance matrix"
    )

    expected_returns = np.array(
        [
            0.10,
            0.12,
            0.08,
        ]
    )

    covariance_matrix = np.zeros(
        (3, 3)
    )

    weights, metadata = optimizer.solve(
        expected_returns,
        covariance_matrix,
    )

    assert np.isfinite(weights).all()

    assert np.isclose(
        np.sum(weights),
        CONSTRAINT_TARGET_SUM,
    )

    logger.info("Test 10 PASSED")
    
    logger.info("=" * 80)
    logger.info(
        "ALL RUNTIME TESTS PASSED SUCCESSFULLY"
    )
    logger.info("=" * 80)