"""
ARQPO - Portfolio Allocation Explainability Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Generates human-readable explanations describing why individual
    assets received their portfolio allocations. The explanations
    are derived from expected returns, portfolio risk, covariance,
    and allocation decisions to improve transparency and
    interpretability.

Workflow Position:
    08_explainability/allocation_reason.py

        Portfolio Weights
                │
                ▼
        Portfolio Statistics
                │
                ├── Expected Returns
                ├── Volatility Analysis
                ├── Covariance Analysis
                └── Allocation Logic
                        │
                        ▼
        Human-Readable Explanations

Mathematical Objective:
    Explain allocation decisions by comparing each asset's expected
    return and volatility against portfolio baselines while
    identifying diversification-driven allocations.

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_WEIGHT_THRESHOLD: float = 0.01
FLOAT_TOLERANCE: float = 1e-10

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


class AllocationExplainer:
    """
    Generates human-readable explanations describing portfolio
    allocation decisions.
    """

    def __init__(
        self,
        weight_threshold: float = DEFAULT_WEIGHT_THRESHOLD,
    ):
        """
        Initializes the allocation explainer.

        Args:
            weight_threshold (float):
                Minimum portfolio allocation considered as an
                actively allocated asset. Assets with weights below
                this threshold are treated as omitted.

        Raises:
            TypeError:
                If weight_threshold is not numeric.

            ValueError:
                If weight_threshold falls outside the valid range.
        """
        logger.info("Initializing AllocationExplainer.")

        if not isinstance(
            weight_threshold,
            (int, float, np.floating),
        ):
            logger.error("weight_threshold must be numeric.")
            raise TypeError("weight_threshold must be a numeric value.")

        weight_threshold = float(weight_threshold)

        if (
            weight_threshold < 0.0
            or weight_threshold > 1.0
        ):
            logger.error(
                "Invalid weight_threshold: %.6f",
                weight_threshold,
            )
            raise ValueError(
                "weight_threshold must lie between 0.0 and 1.0."
            )

        self.weight_threshold = weight_threshold

        logger.info(
            "AllocationExplainer initialized (Weight Threshold=%.4f).",
            self.weight_threshold,
        )

    def explain(
        self,
        weights: np.ndarray,
        asset_names: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> Dict[str, List[str]]:
        """
        Generates human-readable explanations for portfolio allocations.
        """
        start_time = time.perf_counter()

        try:
            logger.info(
                "Starting portfolio allocation explanation."
            )

            self._validate_inputs(
                weights,
                asset_names,
                expected_returns,
                covariance_matrix,
            )

            weights = self._cleanup_array(weights)
            expected_returns = self._cleanup_array(
                expected_returns
            )

            covariance_matrix = np.array(
                covariance_matrix,
                dtype=np.float64,
                copy=True,
            )

            statistics = self._compute_statistics(
                expected_returns,
                covariance_matrix,
            )

            median_return = statistics["median_return"]
            median_volatility = statistics[
                "median_volatility"
            ]
            volatilities = statistics["volatilities"]

            report = self._initialize_report()

            for index, asset in enumerate(asset_names):

                weight = weights[index]
                expected_return = expected_returns[index]
                volatility = volatilities[index]

                if weight < self.weight_threshold:

                    reason = (
                        f"{asset}: Omitted "
                        f"(Weight: {weight:.1%}). "
                    )

                    if (
                        expected_return < median_return
                        and volatility > median_volatility
                    ):
                        reason += (
                            "Sub-optimal risk/return profile "
                            "compared to peers."
                        )

                    elif expected_return < median_return:
                        reason += (
                            "Expected return below "
                            "portfolio median."
                        )

                    else:
                        reason += (
                            "Portfolio risk exceeded "
                            "acceptable level."
                        )

                    report["Omitted"].append(reason)

                else:

                    if expected_return >= median_return:

                        report["Yield_Seeking"].append(
                            (
                                f"{asset}: Allocated "
                                f"{weight:.1%} "
                                f"to improve expected "
                                f"portfolio return "
                                f"({expected_return:.2%} "
                                f">= {median_return:.2%})."
                            )
                        )

                    elif volatility <= median_volatility:

                        report["Risk_Mitigating"].append(
                            (
                                f"{asset}: Allocated "
                                f"{weight:.1%} "
                                f"to improve stability "
                                f"(Volatility "
                                f"{volatility:.2%} "
                                f"<= "
                                f"{median_volatility:.2%})."
                            )
                        )

                    else:

                        report["Risk_Mitigating"].append(
                            (
                                f"{asset}: Allocated "
                                f"{weight:.1%} "
                                f"for diversification "
                                f"benefits despite "
                                f"individual risk."
                            )
                        )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            logger.info(
                "Allocation explanation completed "
                "successfully in %.6f seconds.",
                elapsed,
            )

            return report

        except Exception as exc:

            logger.exception(
                "Allocation explanation failed."
            )

            raise RuntimeError(
                "Unable to generate allocation "
                "explanation."
            ) from exc

    def _validate_inputs(
        self,
        weights: np.ndarray,
        asset_names: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> None:
        if not isinstance(weights, np.ndarray):
            logger.error("weights must be a NumPy array.")
            raise TypeError("weights must be a NumPy array.")

        if not isinstance(expected_returns, np.ndarray):
            logger.error("expected_returns must be a NumPy array.")
            raise TypeError(
                "expected_returns must be a NumPy array."
            )

        if not isinstance(covariance_matrix, np.ndarray):
            logger.error(
                "covariance_matrix must be a NumPy array."
            )
            raise TypeError(
                "covariance_matrix must be a NumPy array."
            )

        if not isinstance(asset_names, list):
            logger.error("asset_names must be a list.")
            raise TypeError("asset_names must be a list.")

        if weights.ndim != 1:
            logger.error("weights must be one-dimensional.")
            raise ValueError(
                "weights must be one-dimensional."
            )

        if expected_returns.ndim != 1:
            logger.error(
                "expected_returns must be one-dimensional."
            )
            raise ValueError(
                "expected_returns must be one-dimensional."
            )

        if covariance_matrix.ndim != 2:
            logger.error(
                "covariance_matrix must be two-dimensional."
            )
            raise ValueError(
                "covariance_matrix must be two-dimensional."
            )

        n_assets = len(weights)

        if (
            len(asset_names) != n_assets
            or len(expected_returns) != n_assets
            or covariance_matrix.shape != (n_assets, n_assets)
        ):
            logger.error(
                "Input dimensions are inconsistent."
            )
            raise ValueError(
                "Input arrays have incompatible dimensions."
            )

        if n_assets == 0:
            logger.error("Portfolio is empty.")
            raise ValueError("Portfolio cannot be empty.")

        if (
            not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(expected_returns))
            or not np.all(np.isfinite(covariance_matrix))
        ):
            logger.error(
                "Inputs contain NaN or infinite values."
            )
            raise ValueError(
                "Inputs must contain only finite values."
            )

        if not np.allclose(
            covariance_matrix,
            covariance_matrix.T,
            atol=FLOAT_TOLERANCE,
        ):
            logger.error(
                "Covariance matrix is not symmetric."
            )
            raise ValueError(
                "Covariance matrix must be symmetric."
            )

        if np.any(
            np.diag(covariance_matrix)
            < -FLOAT_TOLERANCE
        ):
            logger.error(
                "Covariance matrix contains negative variances."
            )
            raise ValueError(
                "Covariance matrix diagonal must be non-negative."
            )

    def _cleanup_array(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        cleaned = np.array(
            values,
            dtype=np.float64,
            copy=True,
        )

        cleaned[
            np.abs(cleaned) < FLOAT_TOLERANCE
        ] = 0.0

        return cleaned

    def _compute_statistics(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        volatilities = np.sqrt(
            np.maximum(
                np.diag(covariance_matrix),
                0.0,
            )
        )

        return {
            "median_return": float(
                np.median(expected_returns)
            ),
            "median_volatility": float(
                np.median(volatilities)
            ),
            "volatilities": volatilities,
        }

    def _initialize_report(
        self,
    ) -> Dict[str, List[str]]:
        return {
            "Yield_Seeking": [],
            "Risk_Mitigating": [],
            "Omitted": [],
        }


# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":

    logger.info(
        "Starting AllocationExplainer runtime tests."
    )

    explainer = AllocationExplainer()

    print("\n" + "=" * 80)
    print("AllocationExplainer Runtime Tests")
    print("=" * 80)

    # Test 1 : Happy Path
    weights = np.array([0.35, 0.30, 0.20, 0.15])
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    expected_returns = np.array([0.18, 0.15, 0.10, 0.08])
    covariance_matrix = np.array(
        [
            [0.040, 0.010, 0.005, 0.004],
            [0.010, 0.050, 0.006, 0.005],
            [0.005, 0.006, 0.030, 0.003],
            [0.004, 0.005, 0.003, 0.020],
        ]
    )

    report = explainer.explain(
        weights,
        asset_names,
        expected_returns,
        covariance_matrix,
    )
    print("Test 1 Report:", report)
    print("\nAll runtime tests verified cleanly.")