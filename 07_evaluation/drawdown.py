"""
ARQPO - Adaptive Regime-Aware Quantum Portfolio Optimizer

Module:
    07_evaluation.drawdown

Purpose:
    Production-grade evaluation of portfolio drawdown metrics.
    This module computes historical drawdown statistics from a
    weighted portfolio constructed using historical asset prices.

Workflow Position:
    Historical Prices
            │
            ▼
    Portfolio Construction
            │
            ▼
    Portfolio Value Series
            │
            ▼
    Running Peak
            │
            ▼
    Drawdown Series
            │
            ▼
    Maximum Drawdown
            │
            ▼
    Benchmarking
            │
            ▼
    Explainability
            │
            ▼
    Dashboard

Mathematical Objective:
    Evaluate downside risk by measuring the decline of a portfolio
    from its historical running maximum.

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
from typing import Any, Dict, Tuple

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE: float = 1e-10
INITIAL_PORTFOLIO_VALUE: float = 1.0
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


class DrawdownCalculator:
    """
    Computes historical drawdown metrics for portfolio performance
    evaluation.

    The calculator converts historical asset prices into a weighted
    portfolio value series before computing the running peak,
    drawdown series, and maximum drawdown.

    The public API is intentionally stable to support downstream
    evaluation, benchmarking, explainability, and dashboard modules.
    """

    def __init__(self) -> None:
        """
        Initialize the drawdown calculator.

        Notes:
            This constructor currently exposes no configurable
            parameters but is retained for API stability and
            future extensibility.
        """
        logger.info("DrawdownCalculator initialized.")

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_inputs(
        self,
        historical_prices: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """
        Validate all public API inputs.

        Args:
            historical_prices:
                Historical asset price matrix with shape
                (observations, assets).

            weights:
                Portfolio allocation vector.

        Raises:
            TypeError:
                If an input is not a NumPy array.

            ValueError:
                If dimensions, sizes, or numerical values
                are invalid.
        """

        if not isinstance(historical_prices, np.ndarray):
            raise TypeError(
                "historical_prices must be a NumPy ndarray."
            )

        if not isinstance(weights, np.ndarray):
            raise TypeError(
                "weights must be a NumPy ndarray."
            )

        if historical_prices.ndim != 2:
            raise ValueError(
                "historical_prices must be a two-dimensional array."
            )

        if weights.ndim != 1:
            raise ValueError(
                "weights must be a one-dimensional array."
            )

        observations, assets = historical_prices.shape

        if observations < 2:
            raise ValueError(
                "At least two observations are required."
            )

        if assets == 0:
            raise ValueError(
                "Historical price matrix contains no assets."
            )

        if weights.size == 0:
            raise ValueError(
                "Weight vector cannot be empty."
            )

        if assets != weights.size:
            raise ValueError(
                "Number of portfolio weights must equal "
                "number of asset columns."
            )

        if not np.isfinite(historical_prices).all():
            raise ValueError(
                "historical_prices contains NaN or infinite values."
            )

        if not np.isfinite(weights).all():
            raise ValueError(
                "weights contains NaN or infinite values."
            )
        if np.isclose(
            np.sum(weights),
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                "Portfolio weights cannot sum to zero."
            )    
              

        logger.info(
            "Input validation completed successfully."
        )

    # =========================================================================
    # Computation Helpers
    # =========================================================================

    def _compute_daily_returns(
        self,
        historical_prices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute simple daily asset returns.

        Args:
            historical_prices:
                Historical asset price matrix.

        Returns:
            Daily percentage returns for each asset.

        Raises:
            ValueError:
                If price values would produce invalid returns.
        """
        logger.info("Computing daily asset returns.")

        daily_returns = (
            historical_prices[1:] / historical_prices[:-1]
        ) - 1.0

        if not np.isfinite(daily_returns).all():
            raise ValueError(
                "Computed daily returns contain invalid values."
            )

        return daily_returns

    def _compute_portfolio_returns(
        self,
        daily_returns: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted portfolio returns.

        Args:
            daily_returns:
                Daily asset return matrix.

            weights:
                Portfolio allocation vector.

        Returns:
            One-dimensional portfolio return series.
        """
        logger.info("Computing portfolio return series.")

        portfolio_returns = np.dot(
            daily_returns,
            weights,
        )

        portfolio_returns = np.nan_to_num(
            portfolio_returns,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return portfolio_returns

    def _compute_cumulative_value(
        self,
        portfolio_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Build the cumulative portfolio value series.

        Args:
            portfolio_returns:
                Weighted portfolio return series.

        Returns:
            Portfolio value series beginning at 1.0.
        """
        logger.info("Building cumulative portfolio value.")

        cumulative_value = np.cumprod(
            INITIAL_PORTFOLIO_VALUE + portfolio_returns
        )

        cumulative_value = np.insert(
            cumulative_value,
            0,
            INITIAL_PORTFOLIO_VALUE,
        )

        return cumulative_value

    def _compute_running_peak(
        self,
        cumulative_value: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the running portfolio peak.

        Args:
            cumulative_value:
                Portfolio value series.

        Returns:
            Running maximum portfolio value.
        """
        logger.info("Computing running portfolio peak.")

        return np.maximum.accumulate(cumulative_value)

    def _compute_drawdown_series(
        self,
        cumulative_value: np.ndarray,
        running_peak: np.ndarray,
    ) -> np.ndarray:
        """
        Compute portfolio drawdown series.

        Args:
            cumulative_value:
                Portfolio value series.

            running_peak:
                Running maximum portfolio values.

        Returns:
            Drawdown percentage series.
        """
        logger.info("Computing drawdown series.")

        safe_peak = np.where(
            np.isclose(
                running_peak,
                0.0,
                atol=FLOAT_TOLERANCE,
            ),
            FLOAT_TOLERANCE,
            running_peak,
        )

        drawdown = (
            safe_peak - cumulative_value
        ) / safe_peak

        drawdown = np.nan_to_num(
            drawdown,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return drawdown

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
                Input numerical array.

        Returns:
            Cleaned NumPy array.
        """
        cleaned = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        cleaned[np.isclose(
            cleaned,
            0.0,
            atol=FLOAT_TOLERANCE,
        )] = 0.0

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
    # Metadata Helpers
    # =========================================================================

    def _compute_drawdown_duration(
        self,
        drawdowns: np.ndarray,
    ) -> int:
        """
        Compute the longest continuous drawdown period.

        Args:
            drawdowns:
                Portfolio drawdown series.

        Returns:
            Maximum drawdown duration.
        """
        longest = 0
        current = 0

        for value in drawdowns:
            if value > FLOAT_TOLERANCE:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        return longest

    def _build_evaluation_metadata(
        self,
        cumulative_value: np.ndarray,
        running_peak: np.ndarray,
        drawdowns: np.ndarray,
        runtime: float,
        asset_count: int 
    ) -> Dict[str, Any]:
        """
        Build production evaluation metadata.

        Args:
            cumulative_value:
                Portfolio value series.

            running_peak:
                Running peak series.

            drawdowns:
                Drawdown series.

            runtime:
                Execution time.

        Returns:
            Metadata dictionary.
        """

        max_index = int(np.argmax(drawdowns))

        metadata = {
            "cumulative_value": cumulative_value,
            "running_max": running_peak,
            "drawdowns": drawdowns,
            "maximum_drawdown": self._cleanup_scalar(
                float(np.max(drawdowns))
            ),
            "average_drawdown": self._cleanup_scalar(
                float(np.mean(drawdowns))
            ),
            "current_drawdown": self._cleanup_scalar(
                float(drawdowns[-1])
            ),
            "maximum_drawdown_index": max_index,
            "observation_count": len(cumulative_value),
            "runtime_seconds": runtime,
            "asset_count": asset_count,
            "drawdown_duration": self._compute_drawdown_duration(
                drawdowns
            ),
        }

        return metadata

    # =========================================================================
    # Public API
    # =========================================================================

    def calculate(
        self,
        historical_prices: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the Maximum Drawdown (MDD) of a weighted portfolio.

        This method constructs a weighted portfolio from historical asset
        prices, computes the cumulative portfolio value, determines the
        running portfolio peak, and evaluates the historical drawdown
        characteristics.

        Args:
            historical_prices:
                Historical asset price matrix of shape
                (observations, assets).

            weights:
                Portfolio allocation vector.

        Returns:
            Tuple containing:

                1. Maximum drawdown.

                2. Production metadata dictionary.

        Raises:
            RuntimeError:
                If drawdown evaluation fails.
        """
        logger.info("Starting drawdown evaluation.")

        start_time = time.perf_counter()

        try:
            # -------------------------------------------------------------
            # Validate inputs
            # -------------------------------------------------------------
            self._validate_inputs(
                historical_prices,
                weights,
            )

            # -------------------------------------------------------------
            # Compute portfolio return series
            # -------------------------------------------------------------
            daily_returns = self._compute_daily_returns(
                historical_prices
            )

            portfolio_returns = self._compute_portfolio_returns(
                daily_returns,
                weights,
            )

            portfolio_returns = self._cleanup_array(
                portfolio_returns
            )

            # -------------------------------------------------------------
            # Portfolio value series
            # -------------------------------------------------------------
            cumulative_value = self._compute_cumulative_value(
                portfolio_returns
            )

            cumulative_value = self._cleanup_array(
                cumulative_value
            )

            # -------------------------------------------------------------
            # Running peak
            # -------------------------------------------------------------
            running_peak = self._compute_running_peak(
                cumulative_value
            )

            running_peak = self._cleanup_array(
                running_peak
            )

            # -------------------------------------------------------------
            # Drawdown series
            # -------------------------------------------------------------
            drawdowns = self._compute_drawdown_series(
                cumulative_value,
                running_peak,
            )

            drawdowns = self._cleanup_array(
                drawdowns
            )

            # -------------------------------------------------------------
            # Maximum drawdown
            # -------------------------------------------------------------
            max_drawdown = self._cleanup_scalar(
                float(np.max(drawdowns))
            )

            runtime = time.perf_counter() - start_time

            metadata = self._build_evaluation_metadata(
                cumulative_value=cumulative_value,
                running_peak=running_peak,
                drawdowns=drawdowns,
                runtime=runtime,
                asset_count=weights.size,
            )

            logger.info(
                "Drawdown evaluation completed successfully."
            )

            logger.info(
                "Maximum Drawdown : %.6f",
                max_drawdown,
            )

            logger.info(
                "Average Drawdown : %.6f",
                metadata["average_drawdown"],
            )

            logger.info(
                "Execution Time : %.6f seconds",
                runtime,
            )

            return max_drawdown, metadata

        except Exception as exc:
            logger.exception(
                "Drawdown evaluation failed."
            )
            raise RuntimeError(
                "Failed to calculate portfolio drawdown."
            ) from exc

# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":

    logger.info("=" * 80)
    logger.info("Starting DrawdownCalculator Runtime Tests")
    logger.info("=" * 80)

    calculator = DrawdownCalculator()

    # -------------------------------------------------------------------------
    # Test 1: Happy Path
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Happy Path")

    prices = np.array([
        [100, 200, 150],
        [102, 198, 152],
        [101, 205, 149],
        [104, 210, 151],
        [103, 208, 153],
    ])

    weights = np.array([0.4, 0.3, 0.3])

    mdd, metadata = calculator.calculate(prices, weights)

    print("Maximum Drawdown :", mdd)
    print("Average Drawdown :", metadata["average_drawdown"])

    # -------------------------------------------------------------------------
    # Test 2: Medium Input
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Medium Input")

    np.random.seed(42)

    prices = np.cumprod(
        1 + np.random.normal(0.001, 0.02, (300, 5)),
        axis=0,
    ) * 100

    weights = np.ones(5) / 5

    mdd, _ = calculator.calculate(prices, weights)

    print("Maximum Drawdown :", mdd)

    # -------------------------------------------------------------------------
    # Test 3: Large Input
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Large Input")

    prices = np.cumprod(
        1 + np.random.normal(0.0005, 0.015, (5000, 10)),
        axis=0,
    ) * 100

    weights = np.ones(10) / 10

    mdd, _ = calculator.calculate(prices, weights)

    print("Maximum Drawdown :", mdd)

    # -------------------------------------------------------------------------
    # Test 4: Invalid Shape
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Invalid Shape")

    try:
        calculator.calculate(
            np.array([1, 2, 3]),
            np.array([1]),
        )
    except Exception as exc:
        print("Expected Exception:", exc)

    # -------------------------------------------------------------------------
    # Test 5: Invalid Values
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Invalid Values")

    bad_prices = np.array([
        [100.0, np.nan],
        [101.0, 200.0],
    ])

    try:
        calculator.calculate(
            bad_prices,
            np.array([0.5, 0.5]),
        )
    except Exception as exc:
        print("Expected Exception:", exc)

    # -------------------------------------------------------------------------
    # Test 6: Edge Case
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Constant Prices")

    prices = np.full((20, 3), 100.0)

    weights = np.array([0.3, 0.3, 0.4])

    mdd, _ = calculator.calculate(prices, weights)

    print("Maximum Drawdown :", mdd)

    # -------------------------------------------------------------------------
    # Test 7: Determinism
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Determinism")

    np.random.seed(7)

    prices = np.cumprod(
        1 + np.random.normal(0.001, 0.02, (200, 4)),
        axis=0,
    ) * 100

    weights = np.ones(4) / 4

    result1, _ = calculator.calculate(prices, weights)
    result2, _ = calculator.calculate(prices, weights)

    print("Deterministic :", np.isclose(result1, result2))

    # -------------------------------------------------------------------------
    # Test 8: Stress Test
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Stress Test")

    prices = np.cumprod(
        1 + np.random.normal(0.0002, 0.01, (10000, 20)),
        axis=0,
    ) * 100

    weights = np.ones(20) / 20

    mdd, _ = calculator.calculate(prices, weights)

    print("Stress Test MDD :", mdd)

    # -------------------------------------------------------------------------
    # Test 9: Numerical Robustness
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Numerical Robustness")

    prices = np.array([
        [100.0, 100.0],
        [100.0000000001, 100.0],
        [99.9999999999, 100.0],
        [100.0, 100.0],
    ])

    weights = np.array([0.5, 0.5])

    mdd, _ = calculator.calculate(prices, weights)

    print("Maximum Drawdown :", mdd)

    # -------------------------------------------------------------------------
    # Test 10: Integration Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Integration Verification")

    prices = np.cumprod(
        1 + np.random.normal(0.001, 0.015, (250, 6)),
        axis=0,
    ) * 100

    weights = np.ones(6) / 6

    _, metadata = calculator.calculate(prices, weights)

    required_keys = [
        "cumulative_value",
        "running_max",
        "drawdowns",
        "maximum_drawdown",
        "average_drawdown",
        "current_drawdown",
        "maximum_drawdown_index",
        "observation_count",
        "runtime_seconds",
        "drawdown_duration",
    ]

    missing = [key for key in required_keys if key not in metadata]

    if missing:
        print("Integration Failed:", missing)
    else:
        print("Integration Passed")

    logger.info("=" * 80)
    logger.info("All Drawdown Runtime Tests Completed Successfully")
    logger.info("=" * 80)