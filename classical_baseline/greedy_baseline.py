"""
ARQPO - Adaptive Regime-Aware Quantum Portfolio Optimizer
=========================================================

Module:
    greedy_baseline.py

Purpose:
    Implements a deterministic Top-K Greedy portfolio allocation strategy
    that serves as a lightweight classical baseline for comparison with
    optimization-based portfolio construction methods such as Mean-Variance,
    SciPy optimization, QAOA, and VQE.
"""

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np

# =============================================================================
# Constants & Configuration
# =============================================================================
DEFAULT_TOP_K = 3
FLOAT_TOLERANCE = 1e-10
DEFAULT_STRATEGY_NAME = "Top-K Greedy"
DEFAULT_SOLVER_NAME = "GreedyBaselineSolver"
MINIMUM_ASSETS = 1

# Configure logging module-level only
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
# Core Algorithm
# =============================================================================
class GreedyBaselineSolver:
    """
    Deterministically allocates equal weights to the Top-K assets based on expected returns.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        
        
        """
        Initialize the GreedyBaselineSolver.

        Args:
            top_k:
                Number of highest-return assets to select for the
                portfolio. Must be a positive integer.

        Raises:
            TypeError:
                If top_k is not an integer.

            ValueError:
                If top_k is less than one.
        """
        
        self.top_k = self._validate_top_k(top_k)
        
        

    # =============================================================================
    # Validation Helpers
    # =============================================================================
    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        """Validates the top_k parameter."""
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        return top_k
    @staticmethod
    def _validate_expected_returns(expected_returns: np.ndarray) -> np.ndarray:
        """Validates the expected returns input array."""
        if not isinstance(expected_returns, np.ndarray):
            raise RuntimeError("Input must be a numpy array.")
        if expected_returns.ndim != 1:
            raise RuntimeError("Input array must be 1-dimensional.")
        if np.isnan(expected_returns).any() or not np.isfinite(expected_returns).all():
            raise RuntimeError("Input contains NaN or infinite values.")
        if expected_returns.size < MINIMUM_ASSETS:
            raise RuntimeError("Input array is empty.")
        return expected_returns.astype(float, copy=False)
    # =============================================================================
    # Numerical Helpers
    # =============================================================================
    @staticmethod    
    
    def _cleanup_weights( weights: np.ndarray) -> np.ndarray:
        """
        Clean numerical artifacts from portfolio weights.

        Args:
            weights:
                Portfolio allocation vector.

        Returns:
            Cleaned portfolio weights.
        """
        cleaned = np.where(
            np.abs(weights) < FLOAT_TOLERANCE,
            0.0,
            weights,
        )

        weight_sum = cleaned.sum()

        if weight_sum > FLOAT_TOLERANCE:
            cleaned = cleaned / weight_sum

        return cleaned
    # =============================================================================
    # Internal Helpers
    # =============================================================================
    def _select_top_assets(self, expected_returns: np.ndarray) -> np.ndarray:
        """
        Select the indices of the Top-K assets.

        Args:
            expected_returns: Vector of expected asset returns.

        Returns:
            NumPy array containing the selected asset indices.
        """
        number_of_assets = expected_returns.size
        top_k = min(self.top_k, number_of_assets)
        
        sorted_indices = np.argsort(expected_returns)[::-1]
        selected_indices = sorted_indices[:top_k]

        logger.info(
            "Selected %d asset(s): %s",
            top_k,
            selected_indices.tolist(),
        )

        return selected_indices

    @staticmethod
    def _allocate_equal_weights(number_of_assets: int, selected_indices: np.ndarray) -> np.ndarray:
        """
        Allocate equal weights to the selected assets.

        Args:
            number_of_assets: Total number of assets.
            selected_indices: Selected asset indices.

        Returns:
            Portfolio weight vector.
        """
        weights = np.zeros(number_of_assets, dtype=float)
        
        if len(selected_indices) > 0:
            allocation = 1.0 / len(selected_indices)
            weights[selected_indices] = allocation

        return weights

    def _build_metadata(
        self,
        selected_indices: np.ndarray,
        execution_time: float,
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Build execution metadata.

        Args:
            selected_indices: Selected asset indices.
            execution_time: Total execution time.
            weights: Portfolio weights.

        Returns:
            Metadata dictionary.
        """
        return {
            "strategy": DEFAULT_STRATEGY_NAME,
            "solver": DEFAULT_SOLVER_NAME,
            "top_k_selected": int(len(selected_indices)),
            "selected_indices": selected_indices.tolist(),
            "execution_time_seconds": execution_time,
            "number_of_assets": int(weights.size),
            "weight_sum": float(weights.sum()),
            "deterministic": True,
        }

    # =============================================================================
    # Public API
    # =============================================================================
    def solve(self, expected_returns: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes the greedy allocation strategy.

        Args:
            expected_returns: Vector of expected asset returns.

        Returns:
            Tuple containing the weight array and metadata dictionary.
        """
        try:
            logger.info("Starting greedy baseline optimization.")
            expected_returns = self._validate_expected_returns(
                 expected_returns
            )

            start_time = time.perf_counter()
            logger.info(
                "Received %d asset(s).",
                expected_returns.size,
            )
            

            selected_indices = self._select_top_assets(expected_returns)
            weights = self._allocate_equal_weights(expected_returns.size, selected_indices)
            weights = self._cleanup_weights(weights)

            execution_time = time.perf_counter() - start_time
            metadata = self._build_metadata(selected_indices, execution_time, weights,)
            
            logger.info(
                "Greedy allocation completed in %.6f seconds.",
                execution_time,
            )
            
            return weights, metadata
            
        except Exception as exc:
            logger.exception("Greedy solver execution failed.")
            raise 
# =============================================================================
# Runtime Tests
# =============================================================================

def _runtime_test_happy_path() -> None:
    """Verify normal greedy portfolio allocation."""
    solver = GreedyBaselineSolver(top_k=2)
    expected_returns = np.array([0.08, 0.15, 0.04, 0.11])
    weights, metadata = solver.solve(expected_returns)

    assert weights.shape == (4,)
    assert np.isclose(weights.sum(), 1.0)
    assert metadata["top_k_selected"] == 2
    logger.info("Runtime Test 1 (Happy Path): PASSED")

def _runtime_test_medium_input() -> None:
    """Verify allocation on a medium-sized portfolio."""
    solver = GreedyBaselineSolver(top_k=5)
    expected_returns = np.random.rand(20)
    weights, metadata = solver.solve(expected_returns)

    assert weights.size == 20
    assert np.isclose(weights.sum(), 1.0)
    assert metadata["top_k_selected"] == 5
    logger.info("Runtime Test 2 (Medium Input): PASSED")

def _runtime_test_large_input() -> None:
    """Verify allocation on a large portfolio."""
    solver = GreedyBaselineSolver(top_k=20)
    expected_returns = np.random.rand(1000)
    weights, _ = solver.solve(expected_returns)

    assert weights.size == 1000
    assert np.isclose(weights.sum(), 1.0)
    logger.info("Runtime Test 3 (Large Input): PASSED")

def _runtime_test_invalid_shape() -> None:
    """Verify two-dimensional arrays are rejected."""
    solver = GreedyBaselineSolver()
    try:
        solver.solve(np.array([[1.0, 2.0], [3.0, 4.0]]))
    except RuntimeError:
        logger.info("Runtime Test 4 (Invalid Shape): PASSED")
        return
    raise AssertionError("Expected RuntimeError was not raised.")

def _runtime_test_invalid_values() -> None:
    """Verify NaN values are rejected."""
    solver = GreedyBaselineSolver()
    try:
        solver.solve(np.array([0.2, np.nan, 0.5]))
    except RuntimeError:
        logger.info("Runtime Test 5 (Invalid Values): PASSED")
        return
    raise AssertionError("Expected RuntimeError was not raised.")

def _runtime_test_edge_case() -> None:
    """Verify single-asset allocation."""
    solver = GreedyBaselineSolver(top_k=1)
    weights, metadata = solver.solve(np.array([0.35]))

    assert np.isclose(weights[0], 1.0)
    assert metadata["top_k_selected"] == 1
    logger.info("Runtime Test 6 (Edge Case): PASSED")

def _runtime_test_determinism() -> None:
    """Verify deterministic behaviour."""
    solver = GreedyBaselineSolver(top_k=3)
    expected_returns = np.array([0.10, 0.22, 0.17, 0.05, 0.19])
    
    weights1, _ = solver.solve(expected_returns)
    weights2, _ = solver.solve(expected_returns)

    assert np.allclose(weights1, weights2)
    logger.info("Runtime Test 7 (Determinism): PASSED")

def _runtime_test_stress() -> None:
    """Repeated execution stability test."""
    solver = GreedyBaselineSolver(top_k=10)
    expected_returns = np.random.rand(500)
    
    
    previous_level = logger.level
    logger.setLevel(logging.WARNING)

    try:
        for _ in range(100):
            weights, _ = solver.solve(expected_returns)
            assert np.isclose(weights.sum(), 1.0)
    finally:
        logger.setLevel(previous_level)

    logger.info("Runtime Test 8 (Stress Test): PASSED")
        
    
def _runtime_test_numerical_robustness() -> None:
    """Verify numerical stability."""
    solver = GreedyBaselineSolver(top_k=2)
    expected_returns = np.array([1e-12, 2e-12, 3e-12, 4e-12])
    weights, _ = solver.solve(expected_returns)

    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)
    logger.info("Runtime Test 9 (Numerical Robustness): PASSED")

def _runtime_test_integration() -> None:
    """Verify compatibility with downstream modules."""
    solver = GreedyBaselineSolver(top_k=3)
    expected_returns = np.random.rand(8)
    weights, metadata = solver.solve(expected_returns)

    assert isinstance(weights, np.ndarray)
    assert isinstance(metadata, dict)

    required_keys = (
        "strategy",
        "solver",
        "selected_indices",
        "execution_time_seconds",
    )
    for key in required_keys:
        assert key in metadata

    logger.info("Runtime Test 10 (Integration): PASSED")
def _error_test_invalid_top_k_type() -> None:
    """Verify non-integer top_k is rejected."""
    try:
        GreedyBaselineSolver(top_k="5")
    except TypeError:
        logger.info("Error Test 1 (Invalid top_k Type): PASSED")
        return

    raise AssertionError("Expected TypeError was not raised.")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Running Greedy Baseline Runtime and Error Tests")
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

    # =============================================================================
    # Error Tests
    # =============================================================================
    _error_test_invalid_top_k_type()

    logger.info("=" * 70)
    logger.info("ALL RUNTIME AND ERROR TESTS PASSED")
    logger.info("=" * 70)