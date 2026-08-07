"""
ARQPO - Diversification Validation Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Provides portfolio diversification validation using concentration
    metrics including the Herfindahl-Hirschman Index (HHI) and
    Shannon Entropy.

Workflow Position:
    06_validation/diversification.py

        Portfolio Weights
                │
                ▼
        Diversification Evaluation
                │
        ├── HHI
        ├── Shannon Entropy
        └── Diversification Status
                │
                ▼
        Validation Report

Mathematical Objective:
    Quantify allocation concentration to ensure portfolios satisfy
    diversification requirements before downstream evaluation and
    explainability modules.

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
from typing import Any, Dict

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_HHI_THRESHOLD: float = 0.30
FLOAT_TOLERANCE: float = 1e-10
MIN_WEIGHT_THRESHOLD: float = FLOAT_TOLERANCE

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


class DiversificationEvaluator:
    """
    Evaluates portfolio concentration using standard financial metrics.
    """

    def __init__(self, hhi_threshold: float = DEFAULT_HHI_THRESHOLD):
        """
        Initializes the diversification evaluator.

        Args:
            hhi_threshold (float):
                Maximum acceptable Herfindahl-Hirschman Index (HHI) value.
                Must lie within the interval [0.0, 1.0].

        Raises:
            TypeError:
                If ``hhi_threshold`` is not numeric.
            ValueError:
                If ``hhi_threshold`` is outside the valid range.

        Notes:
            The threshold is used by :meth:`evaluate` to determine whether
            a portfolio satisfies the diversification requirement.
        """
        if not isinstance(hhi_threshold, (int, float, np.floating)):
            logger.error(
                "Invalid hhi_threshold type: %s",
                type(hhi_threshold).__name__,
            )
            raise TypeError("hhi_threshold must be a numeric value.")

        hhi_threshold = float(hhi_threshold)

        if not (0.0 <= hhi_threshold <= 1.0):
            logger.error(
                "Invalid hhi_threshold value: %.6f",
                hhi_threshold,
            )
            raise ValueError("hhi_threshold must be between 0.0 and 1.0.")

        self.hhi_threshold = hhi_threshold

        logger.info(
            "DiversificationEvaluator initialized (HHI Threshold = %.4f).",
            self.hhi_threshold,
        )

    # -------------------------------------------------------------------------
    # Private Validation & Cleanup Helpers
    # -------------------------------------------------------------------------

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Validates the portfolio weight vector.

        Parameters:
            weights (np.ndarray):
                Portfolio allocation vector.

        Returns:
            np.ndarray:
                Validated one-dimensional NumPy array.

        Raises:
            TypeError:
                If weights is not a NumPy array.
            ValueError:
                If weights contains invalid values or has an invalid shape.
        """
        if not isinstance(weights, np.ndarray):
            logger.error(
                "Expected numpy.ndarray, received %s.",
                type(weights).__name__,
            )
            raise TypeError("weights must be a NumPy array.")

        if weights.ndim != 1:
            logger.error(
                "weights must be one-dimensional (received shape %s).",
                weights.shape,
            )
            raise ValueError("weights must be a one-dimensional array.")

        if weights.size == 0:
            logger.error("Portfolio weight vector is empty.")
            raise ValueError("weights cannot be empty.")

        if not np.all(np.isfinite(weights)):
            logger.error("weights contain NaN or infinite values.")
            raise ValueError("weights must contain only finite values.")

        if np.any(weights < -FLOAT_TOLERANCE):
            logger.error("weights contain negative allocations.")
            raise ValueError("weights cannot contain negative values.")

        weight_sum = float(np.sum(weights))

        if not np.isclose(weight_sum, 1.0, atol=FLOAT_TOLERANCE):
            logger.error(
                "weights must sum to 1.0 (received %.12f).",
                weight_sum,
            )
            raise ValueError("weights must sum to 1.0.")

        return weights.astype(np.float64, copy=False)

    def _cleanup_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Removes small numerical artifacts from the weight vector.

        Parameters:
            weights (np.ndarray):
                Portfolio allocation vector.

        Returns:
            np.ndarray:
                Cleaned weight vector.
        """
        cleaned = np.array(weights, dtype=np.float64, copy=True)
        cleaned[np.abs(cleaned) < FLOAT_TOLERANCE] = 0.0
        return cleaned

    # -------------------------------------------------------------------------
    # Public Calculation & Evaluation Methods
    # -------------------------------------------------------------------------

    def calculate_hhi(self, weights: np.ndarray) -> float:
        """
        Calculates the Herfindahl-Hirschman Index (HHI).

        The HHI measures portfolio concentration by summing the squared
        portfolio weights. Lower values indicate better diversification,
        while higher values indicate greater concentration.

        Args:
            weights (np.ndarray):
                One-dimensional portfolio weight vector.

        Returns:
            float:
                Herfindahl-Hirschman Index (HHI).

        Raises:
            RuntimeError:
                If the calculation fails.

        Notes:
            The input is validated and numerically cleaned before
            computation to ensure deterministic and robust results.
        """
        start_time = time.perf_counter()

        try:
            logger.info("Calculating Herfindahl-Hirschman Index (HHI).")

            validated_weights = self._validate_weights(weights)
            cleaned_weights = self._cleanup_weights(validated_weights)

            hhi_score = float(np.sum(np.square(cleaned_weights)))

            if np.isclose(hhi_score, 0.0, atol=FLOAT_TOLERANCE):
                hhi_score = 0.0

            elapsed_time = time.perf_counter() - start_time

            logger.info(
                "HHI calculation completed successfully "
                "(HHI=%.6f, Runtime=%.6f seconds).",
                hhi_score,
                elapsed_time,
            )

            return hhi_score

        except Exception as exc:
            logger.exception("Failed to calculate HHI.")
            raise RuntimeError(
                "Unable to calculate Herfindahl-Hirschman Index."
            ) from exc

    def calculate_entropy(self, weights: np.ndarray) -> float:
        """
        Calculates the Shannon Entropy of the portfolio allocation.

        Shannon Entropy is a measure of portfolio diversification.
        Larger entropy values indicate a more diversified portfolio,
        while lower values indicate greater concentration.

        Args:
            weights (np.ndarray):
                One-dimensional portfolio weight vector.

        Returns:
            float:
                Shannon Entropy of the portfolio.

        Raises:
            RuntimeError:
                If the entropy calculation fails.

        Notes:
            The input weights are validated and cleaned before
            computation. Zero-weight assets are excluded from the
            logarithm to avoid undefined values.
        """
        start_time = time.perf_counter()

        try:
            logger.info("Calculating Shannon Entropy.")

            validated_weights = self._validate_weights(weights)
            cleaned_weights = self._cleanup_weights(validated_weights)

            non_zero_weights = cleaned_weights[
                cleaned_weights > MIN_WEIGHT_THRESHOLD
            ]

            if non_zero_weights.size == 0:
                logger.warning(
                    "Portfolio contains no positive allocations. "
                    "Entropy is defined as 0.0."
                )
                return 0.0

            entropy_score = float(
                -np.sum(non_zero_weights * np.log(non_zero_weights))
            )

            if np.isclose(
                entropy_score,
                0.0,
                atol=FLOAT_TOLERANCE,
            ):
                entropy_score = 0.0

            elapsed_time = time.perf_counter() - start_time

            logger.info(
                "Entropy calculation completed successfully "
                "(Entropy=%.6f, Runtime=%.6f seconds).",
                entropy_score,
                elapsed_time,
            )

            return entropy_score

        except Exception as exc:
            logger.exception("Failed to calculate Shannon Entropy.")
            raise RuntimeError(
                "Unable to calculate Shannon Entropy."
            ) from exc

    def evaluate(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates the diversification quality of a portfolio.

        The portfolio is assessed using both the Herfindahl-Hirschman Index (HHI)
        and Shannon Entropy. A portfolio is considered diversified if its HHI
        does not exceed the configured threshold.

        Args:
            weights (np.ndarray):
                One-dimensional portfolio weight vector.

        Returns:
            Dict[str, Any]:
                Dictionary containing:
                - hhi_score
                - entropy_score
                - is_diversified
                - hhi_threshold_used
                - execution_time_ms

        Raises:
            RuntimeError:
                If the diversification evaluation fails.

        Notes:
            The public API and metadata keys are preserved for backward
            compatibility with downstream modules.
        """
        start_time = time.perf_counter()

        try:
            logger.info("Starting diversification evaluation.")

            validated_weights = self._validate_weights(weights)
            cleaned_weights = self._cleanup_weights(validated_weights)

            hhi_score = self.calculate_hhi(cleaned_weights)
            entropy_score = self.calculate_entropy(cleaned_weights)

            is_diversified = (
                hhi_score <= self.hhi_threshold
                or np.isclose(
                    hhi_score,
                    self.hhi_threshold,
                    atol=FLOAT_TOLERANCE,
                )
            )

            elapsed_time = time.perf_counter() - start_time
            execution_time_ms = elapsed_time * 1000.0

            report = {
                "hhi_score": hhi_score,
                "entropy_score": entropy_score,
                "is_diversified": is_diversified,
                "hhi_threshold_used": self.hhi_threshold,
                "execution_time_ms": execution_time_ms,
            }

            if is_diversified:
                logger.info(
                    "Diversification evaluation PASSED "
                    "(HHI=%.6f, Entropy=%.6f, Runtime=%.6f seconds).",
                    hhi_score,
                    entropy_score,
                    elapsed_time,
                )
            else:
                logger.warning(
                    "Diversification evaluation FAILED "
                    "(HHI=%.6f exceeds threshold %.6f, "
                    "Entropy=%.6f, Runtime=%.6f seconds).",
                    hhi_score,
                    self.hhi_threshold,
                    entropy_score,
                    elapsed_time,
                )

            return report

        except Exception as exc:
            logger.exception("Diversification evaluation failed.")
            raise RuntimeError(
                "Unable to complete diversification evaluation."
            ) from exc


# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting DiversificationEvaluator runtime tests.")

    evaluator = DiversificationEvaluator()

    test_cases = [
        (
            "Happy Path",
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
        (
            "Medium Input",
            np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.15]),
        ),
        (
            "Large Input",
            np.full(100, 0.01),
        ),
        (
            "Edge Case (Single Asset)",
            np.array([1.0]),
        ),
    ]

    print("\n" + "=" * 70)
    print("DiversificationEvaluator Runtime Tests")
    print("=" * 70)

    for name, weights in test_cases:
        try:
            result = evaluator.evaluate(weights)

            print(f"\n{name}")
            print("-" * 50)
            print(f"HHI          : {result['hhi_score']:.6f}")
            print(f"Entropy      : {result['entropy_score']:.6f}")
            print(f"Diversified  : {result['is_diversified']}")

        except Exception as exc:
            print(f"\n{name}")
            print("-" * 50)
            print(f"FAILED : {exc}")

    print("\n" + "=" * 70)
    print("Validation Tests")
    print("=" * 70)

    invalid_inputs = [
        (
            "2-D Array",
            np.array([[0.5, 0.5]]),
        ),
        (
            "Contains NaN",
            np.array([0.5, np.nan, 0.5]),
        ),
        (
            "Contains Inf",
            np.array([0.5, np.inf, 0.5]),
        ),
        (
            "Negative Weight",
            np.array([0.7, -0.2, 0.5]),
        ),
        (
            "Weights != 1",
            np.array([0.4, 0.4]),
        ),
    ]

    for name, weights in invalid_inputs:
        try:
            evaluator.evaluate(weights)
            print(f"{name:<25} FAILED (Expected exception)")

        except Exception as exc:
            print(f"{name:<25} PASSED ({type(exc).__name__})")

    print("\n" + "=" * 70)
    print("Determinism Test")
    print("=" * 70)

    weights = np.array([0.30, 0.20, 0.50])

    first = evaluator.evaluate(weights)
    second = evaluator.evaluate(weights)

    deterministic = (
        np.isclose(first["hhi_score"], second["hhi_score"])
        and np.isclose(
            first["entropy_score"],
            second["entropy_score"],
        )
        and first["is_diversified"] == second["is_diversified"]
    )

    print(f"Deterministic Output : {deterministic}")

    print("\n" + "=" * 70)
    print("All runtime tests completed.")
    print("=" * 70)