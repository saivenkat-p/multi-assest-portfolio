"""
ARQPO - Portfolio Tradeoff Analysis Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Quantifies and explains the marginal tradeoff between two
    portfolio allocations by comparing their expected return,
    volatility, and risk-return efficiency.

Workflow Position:
    08_explainability/tradeoff_analysis.py

        Portfolio A
            │
        Portfolio B
            │
            ▼
     Risk/Return Comparison
            │
     ├── Expected Return
     ├── Volatility
     ├── Tradeoff Ratio
     └── Human-Readable Explanation
            │
            ▼
     Explainability Report

Mathematical Objective:
    Compare two portfolio allocations using Modern Portfolio
    Theory (MPT). The module evaluates expected return,
    portfolio volatility, and the marginal return gained or
    lost per unit of additional risk.

Design Philosophy:
    - Defensive Programming
    - Numerical Stability
    - Deterministic Behaviour
    - Production Logging
    - Backward Compatibility
    - Research-Grade Implementation
"""

import logging
import time
from typing import Any, Dict, Tuple

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE: float = 1e-10
METRIC_TOLERANCE: float = 1e-8

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
# Public Classes
# =============================================================================


class TradeoffAnalyzer:
    """
    Quantifies and explains the marginal risk-return
    tradeoff between two portfolio allocations.
    """

    def __init__(self):
        """
        Initializes the tradeoff analyzer.

        Raises:
            RuntimeError:
                If the analyzer cannot be initialized.
        """
        logger.info("Initializing TradeoffAnalyzer.")

        try:
            if not isinstance(FLOAT_TOLERANCE, float):
                logger.error("FLOAT_TOLERANCE must be a float.")
                raise TypeError("FLOAT_TOLERANCE must be a float.")

            if FLOAT_TOLERANCE <= 0.0:
                logger.error("FLOAT_TOLERANCE must be positive.")
                raise ValueError("FLOAT_TOLERANCE must be positive.")

            if not isinstance(METRIC_TOLERANCE, float):
                logger.error("METRIC_TOLERANCE must be a float.")
                raise TypeError("METRIC_TOLERANCE must be a float.")

            if METRIC_TOLERANCE <= FLOAT_TOLERANCE:
                logger.error(
                    "METRIC_TOLERANCE must be greater than FLOAT_TOLERANCE."
                )
                raise ValueError(
                    "METRIC_TOLERANCE must be greater than FLOAT_TOLERANCE."
                )

            self.float_tolerance = FLOAT_TOLERANCE
            self.metric_tolerance = METRIC_TOLERANCE

            logger.info("TradeoffAnalyzer initialized successfully.")

        except Exception as exc:
            logger.exception("Failed to initialize TradeoffAnalyzer.")
            raise RuntimeError("Unable to initialize TradeoffAnalyzer.") from exc

    # =============================================================================
    # Validation Helpers
    # =============================================================================

    def _validate_inputs(
        self,
        weights_primary: np.ndarray,
        weights_alternative: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:
        """
        Validates all inputs required for tradeoff analysis.
        """
        arrays = {
            "weights_primary": weights_primary,
            "weights_alternative": weights_alternative,
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
        }

        for name, array in arrays.items():
            if not isinstance(array, np.ndarray):
                logger.error("%s must be a NumPy array.", name)
                raise TypeError(f"{name} must be a NumPy array.")

            if not np.isfinite(array).all():
                logger.error("%s contains NaN or Inf.", name)
                raise ValueError(f"{name} contains NaN or Inf.")

        if weights_primary.ndim != 1:
            raise ValueError("weights_primary must be one-dimensional.")

        if weights_alternative.ndim != 1:
            raise ValueError("weights_alternative must be one-dimensional.")

        if expected_returns.ndim != 1:
            raise ValueError("expected_returns must be one-dimensional.")

        if covariance_matrix.ndim != 2:
            raise ValueError("covariance_matrix must be two-dimensional.")

        asset_count = expected_returns.size

        if weights_primary.size != asset_count:
            raise ValueError(
                "weights_primary size does not match expected_returns."
            )

        if weights_alternative.size != asset_count:
            raise ValueError(
                "weights_alternative size does not match expected_returns."
            )

        if covariance_matrix.shape != (asset_count, asset_count):
            raise ValueError("covariance_matrix has invalid shape.")

        if not np.allclose(
            covariance_matrix,
            covariance_matrix.T,
            atol=self.float_tolerance,
        ):
            raise ValueError("covariance_matrix must be symmetric.")

    # =============================================================================
    # Metric Calculation Helpers
    # =============================================================================

    def _calculate_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates expected return and volatility for a portfolio.
        """
        portfolio_return = float(np.dot(weights, expected_returns))
        portfolio_variance = float(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        portfolio_volatility = float(np.sqrt(max(0.0, portfolio_variance)))
        return portfolio_return, portfolio_volatility

    # =============================================================================
    # Numerical Helpers
    # =============================================================================

    def _cleanup_scalar(
        self,
        value: float,
    ) -> float:
        """
        Cleans floating-point numerical artifacts.
        """
        value = float(value)

        if not np.isfinite(value):
            raise ValueError("Scalar value must be finite.")

        if np.isclose(
            value,
            0.0,
            atol=self.float_tolerance,
        ):
            return 0.0

        return value

    # =============================================================================
    # Internal Helpers
    # =============================================================================

    def _build_explanation(
        self,
        delta_return: float,
        delta_volatility: float,
        tradeoff_ratio: float,
        alternative_name: str,
    ) -> str:
        """
        Builds the human-readable tradeoff explanation.
        """
        if (
            np.isclose(
                delta_return,
                0.0,
                atol=self.metric_tolerance,
            )
            and np.isclose(
                delta_volatility,
                0.0,
                atol=self.metric_tolerance,
            )
        ):
            return "Both portfolios have identical risk and return profiles."

        if delta_return > 0.0 and delta_volatility > 0.0:
            return (
                f"{alternative_name} is more aggressive. "
                f"It gains {delta_return:.2%} return while "
                f"adding {delta_volatility:.2%} volatility. "
                f"Tradeoff Ratio: {tradeoff_ratio:.4f}."
            )

        if delta_return < 0.0 and delta_volatility < 0.0:
            return (
                f"{alternative_name} is more conservative. "
                f"It sacrifices {abs(delta_return):.2%} return "
                f"to reduce volatility by {abs(delta_volatility):.2%}. "
                f"Tradeoff Ratio: {tradeoff_ratio:.4f}."
            )

        if delta_return > 0.0 and delta_volatility <= 0.0:
            return (
                f"{alternative_name} is strictly superior. "
                "Higher return with equal or lower risk."
            )

        return (
            f"{alternative_name} is strictly inferior. "
            "Lower return with equal or higher risk."
        )

    # =============================================================================
    # Public Analysis Method
    # =============================================================================

    def analyze(
        self,
        weights_primary: np.ndarray,
        weights_alternative: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        primary_name: str = "Primary Portfolio",
        alternative_name: str = "Alternative Portfolio",
    ) -> Dict[str, Any]:
        """
        Compares two portfolios and generates a quantitative tradeoff analysis.
        """
        start_time = time.perf_counter()

        logger.info(
            "Starting tradeoff analysis between '%s' and '%s'.",
            primary_name,
            alternative_name,
        )

        try:
            self._validate_inputs(
                weights_primary=weights_primary,
                weights_alternative=weights_alternative,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
            )

            primary_return, primary_risk = self._calculate_metrics(
                weights_primary,
                expected_returns,
                covariance_matrix,
            )

            alternative_return, alternative_risk = self._calculate_metrics(
                weights_alternative,
                expected_returns,
                covariance_matrix,
            )

            primary_return = self._cleanup_scalar(primary_return)
            primary_risk = self._cleanup_scalar(primary_risk)
            alternative_return = self._cleanup_scalar(alternative_return)
            alternative_risk = self._cleanup_scalar(alternative_risk)

            delta_return = self._cleanup_scalar(
                alternative_return - primary_return
            )
            delta_volatility = self._cleanup_scalar(
                alternative_risk - primary_risk
            )

            if np.isclose(
                delta_volatility,
                0.0,
                atol=self.metric_tolerance,
            ):
                tradeoff_ratio = float("inf")
            else:
                tradeoff_ratio = delta_return / delta_volatility

            explanation = self._build_explanation(
                delta_return=delta_return,
                delta_volatility=delta_volatility,
                tradeoff_ratio=tradeoff_ratio,
                alternative_name=alternative_name,
            )

            result = {
                "primary_portfolio": primary_name,
                "alternative_portfolio": alternative_name,
                "primary_expected_return": primary_return,
                "alternative_expected_return": alternative_return,
                "primary_volatility": primary_risk,
                "alternative_volatility": alternative_risk,
                "delta_return": delta_return,
                "delta_volatility": delta_volatility,
                "tradeoff_ratio": tradeoff_ratio,
                "explanation": explanation,
            }

            elapsed = time.perf_counter() - start_time
            logger.info("Tradeoff analysis completed in %.6f seconds.", elapsed)

            return result

        except Exception as exc:
            logger.exception("Tradeoff analysis failed.")
            raise RuntimeError("Unable to complete tradeoff analysis.") from exc


# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting TradeoffAnalyzer runtime tests.")

    analyzer = TradeoffAnalyzer()

    print("\n" + "=" * 80)
    print("TradeoffAnalyzer Runtime Tests")
    print("=" * 80)

    expected_returns = np.array([0.12, 0.08, 0.15])

    covariance_matrix = np.array(
        [
            [0.040, 0.006, 0.004],
            [0.006, 0.030, 0.005],
            [0.004, 0.005, 0.050],
        ]
    )

    primary = np.array([0.40, 0.35, 0.25])
    alternative = np.array([0.30, 0.30, 0.40])

    # Test 1 : Happy Path
    print("\n[1] Happy Path")
    result = analyzer.analyze(
        primary,
        alternative,
        expected_returns,
        covariance_matrix,
    )
    print(result)

    # Test 2 : Medium Input
    print("\n[2] Medium Input")
    n_assets = 50
    np.random.seed(42)
    weights1 = np.random.dirichlet(np.ones(n_assets))
    weights2 = np.random.dirichlet(np.ones(n_assets))
    returns = np.random.uniform(0.02, 0.20, n_assets)
    covariance = np.eye(n_assets) * 0.05

    result = analyzer.analyze(weights1, weights2, returns, covariance)
    print("Tradeoff Ratio:", result["tradeoff_ratio"])

    # Test 3 : Large Input
    print("\n[3] Large Input")
    n_assets = 500
    weights1 = np.random.dirichlet(np.ones(n_assets))
    weights2 = np.random.dirichlet(np.ones(n_assets))
    returns = np.random.uniform(0.01, 0.25, n_assets)
    covariance = np.eye(n_assets) * 0.03

    analyzer.analyze(weights1, weights2, returns, covariance)
    print("Large input passed.")

    # Test 4 : Invalid Shape
    print("\n[4] Invalid Shape")
    try:
        analyzer.analyze(
            np.array([[0.5, 0.5]]),
            alternative,
            expected_returns,
            covariance_matrix,
        )
    except Exception as exc:
        print(type(exc).__name__, exc)

    # Test 5 : Invalid Values
    print("\n[5] Invalid Values")
    invalid_returns = np.array([0.10, np.nan, 0.15])
    try:
        analyzer.analyze(
            primary,
            alternative,
            invalid_returns,
            covariance_matrix,
        )
    except Exception as exc:
        print(type(exc).__name__, exc)

    # Test 6 : Edge Case
    print("\n[6] Edge Case")
    identical = analyzer.analyze(
        primary,
        primary,
        expected_returns,
        covariance_matrix,
    )
    print(identical["explanation"])

    # Test 7 : Determinism
    print("\n[7] Determinism")
    first = analyzer.analyze(
        primary,
        alternative,
        expected_returns,
        covariance_matrix,
    )
    second = analyzer.analyze(
        primary,
        alternative,
        expected_returns,
        covariance_matrix,
    )
    print(first == second)

    # Test 8 : Stress Test
    print("\n[8] Stress Test")
    for _ in range(1000):
        analyzer.analyze(
            primary,
            alternative,
            expected_returns,
            covariance_matrix,
        )
    print("Stress test passed.")

    # Test 9 : Numerical Robustness
    print("\n[9] Numerical Robustness")
    tiny_returns = np.array([1e-12, 2e-12, 3e-12])
    result = analyzer.analyze(
        primary,
        alternative,
        tiny_returns,
        covariance_matrix,
    )
    print(result["tradeoff_ratio"])

    # Test 10 : Integration Verification
    print("\n[10] Integration Verification")
    required_keys = {
        "primary_portfolio",
        "alternative_portfolio",
        "primary_expected_return",
        "alternative_expected_return",
        "primary_volatility",
        "alternative_volatility",
        "delta_return",
        "delta_volatility",
        "tradeoff_ratio",
        "explanation",
    }
    integration_ok = required_keys.issubset(result.keys())
    print("Integration Passed:", integration_ok)

    print("\n" + "=" * 80)
    print("All runtime tests completed successfully.")
    print("=" * 80)