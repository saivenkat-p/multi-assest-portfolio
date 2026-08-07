"""
ARQPO - Exposure Limits Validation Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Validates portfolio allocations against predefined individual
    asset exposure constraints. The module ensures that each asset
    weight remains within its configured minimum and maximum bounds
    before downstream evaluation and explainability stages.

Workflow Position:
    06_validation/exposure_limits.py

        Portfolio Weights
                │
                ▼
        Exposure Limit Validation
                │
                ├── Asset Bound Validation
                ├── Constraint Verification
                └── Exposure Status
                        │
                        ▼
                Validation Report

Mathematical Objective:
    Verify that every portfolio allocation satisfies predefined
    lower and upper exposure limits while accounting for floating-
    point numerical tolerances.

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

DEFAULT_TOLERANCE: float = 1e-6
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


class ExposureValidator:
    """
    Checks portfolio weights against custom individual asset exposure limits.
    """

    def __init__(
        self,
        asset_bounds: Optional[List[Tuple[float, float]]] = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        """
        Initializes the exposure validator.

        Args:
            asset_bounds (Optional[List[Tuple[float, float]]]):
                List containing minimum and maximum allocation limits for each
                asset as (min_weight, max_weight) tuples. If None, exposure
                validation is skipped.
            tolerance (float):
                Floating-point tolerance applied during boundary comparisons.

        Raises:
            TypeError:
                If asset_bounds or tolerance has an invalid type.
            ValueError:
                If tolerance is negative or asset bounds are invalid.
        """
        logger.info("Initializing ExposureValidator.")

        if asset_bounds is not None:
            if not isinstance(asset_bounds, list):
                logger.error("asset_bounds must be a list of tuples.")
                raise TypeError(
                    "asset_bounds must be a list of (min_weight, max_weight) tuples."
                )

            for index, bounds in enumerate(asset_bounds):
                if not isinstance(bounds, tuple) or len(bounds) != 2:
                    logger.error("Invalid asset bound at index %d.", index)
                    raise TypeError(
                        "Each asset bound must be a (min_weight, max_weight) tuple."
                    )

                min_weight, max_weight = bounds

                if not isinstance(
                    min_weight, (int, float, np.floating)
                ) or not isinstance(
                    max_weight, (int, float, np.floating)
                ):
                    logger.error(
                        "Asset bound %d contains non-numeric values.", index
                    )
                    raise TypeError(
                        "Asset bounds must contain numeric values."
                    )

                min_weight = float(min_weight)
                max_weight = float(max_weight)

                if min_weight > max_weight:
                    logger.error(
                        "Minimum bound exceeds maximum bound at index %d.", index
                    )
                    raise ValueError(
                        "Each minimum bound must be less than or equal to its maximum bound."
                    )

                if (
                    min_weight < -FLOAT_TOLERANCE
                    or max_weight > 1.0 + FLOAT_TOLERANCE
                ):
                    logger.error(
                        "Asset bounds at index %d are outside the valid range [0, 1].",
                        index,
                    )
                    raise ValueError(
                        "Asset bounds must lie between 0.0 and 1.0."
                    )

        if not isinstance(tolerance, (int, float, np.floating)):
            logger.error("Tolerance must be numeric.")
            raise TypeError("tolerance must be a numeric value.")

        tolerance = float(tolerance)

        if tolerance < 0.0:
            logger.error("Tolerance cannot be negative.")
            raise ValueError("tolerance must be non-negative.")

        self.asset_bounds = asset_bounds
        self.tolerance = tolerance

        self._validate_asset_bounds()

        logger.info(
            "ExposureValidator initialized (Assets=%s, Tolerance=%.2e).",
            len(asset_bounds) if asset_bounds else 0,
            self.tolerance,
        )

    def check_asset_bounds(
        self,
        weights: np.ndarray,
    ) -> Tuple[bool, List[bool]]:
        """
        Validates individual weights against their specific limits.

        Args:
            weights (np.ndarray): Portfolio allocation vector.

        Returns:
            Tuple[bool, List[bool]]: Boolean flag indicating if all weights pass,
            and a list of individual boolean check results for each asset.
        """
        weights = self._validate_weights(weights)
        weights = self._cleanup_weights(weights)
        self._validate_dimensions(weights)

        if not self.asset_bounds:
            return True, [True] * len(weights)

        validations = []

        for i, (w, (min_w, max_w)) in enumerate(
            zip(weights, self.asset_bounds)
        ):
            is_within_bounds = (
                (min_w - self.tolerance)
                <= w
                <= (max_w + self.tolerance)
            )

            validations.append(is_within_bounds)

            if not is_within_bounds:
                logger.debug(
                    "Asset %d weight %.4f violated bounds [%.4f, %.4f].",
                    i,
                    w,
                    min_w,
                    max_w,
                )

        all_valid = all(validations)

        if not all_valid:
            logger.warning(
                "One or more assets violated specific exposure limits."
            )
        else:
            logger.info("All assets are within specified exposure limits.")

        return all_valid, validations

    def validate(
        self,
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Executes all exposure limit validation checks.

        Args:
            weights (np.ndarray): One-dimensional portfolio allocation vector.

        Returns:
            Dict[str, Any]: Dictionary containing 'exposure_limits_passed'
            and 'individual_asset_checks'.

        Raises:
            RuntimeError: If the exposure validation process fails.
        """
        start_time = time.perf_counter()

        try:
            logger.info("Starting exposure limit validation.")

            validated_weights = self._validate_weights(weights)
            cleaned_weights = self._cleanup_weights(validated_weights)

            assets_pass, asset_checks = self.check_asset_bounds(cleaned_weights)

            report = {
                "exposure_limits_passed": assets_pass,
                "individual_asset_checks": asset_checks,
            }

            elapsed_time = time.perf_counter() - start_time

            if assets_pass:
                logger.info(
                    "Exposure validation PASSED (Runtime=%.6f seconds).",
                    elapsed_time,
                )
            else:
                failed_assets = sum(not passed for passed in asset_checks)
                logger.warning(
                    "Exposure validation FAILED (Failed Assets=%d, Runtime=%.6f seconds).",
                    failed_assets,
                    elapsed_time,
                )

            return report

        except Exception as exc:
            logger.exception("Exposure validation failed.")
            raise RuntimeError(
                "Unable to complete exposure validation."
            ) from exc

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        if not isinstance(weights, np.ndarray):
            logger.error(
                "Expected numpy.ndarray, received %s.",
                type(weights).__name__,
            )
            raise TypeError("weights must be provided as a NumPy array.")

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

        return weights.astype(np.float64, copy=False)

    def _validate_asset_bounds(self) -> None:
        if self.asset_bounds is not None and len(self.asset_bounds) == 0:
            logger.error("asset_bounds cannot be an empty list.")
            raise ValueError("asset_bounds cannot be empty.")

    def _validate_dimensions(self, weights: np.ndarray) -> None:
        if self.asset_bounds is None:
            return

        if len(weights) != len(self.asset_bounds):
            logger.error(
                "Dimension mismatch: %d weights vs %d bounds.",
                len(weights),
                len(self.asset_bounds),
            )
            raise ValueError(
                "Length of weights must match length of asset_bounds."
            )

    def _cleanup_weights(self, weights: np.ndarray) -> np.ndarray:
        cleaned = np.array(weights, dtype=np.float64, copy=True)
        cleaned[np.abs(cleaned) < FLOAT_TOLERANCE] = 0.0
        return cleaned


# =============================================================================
# Runtime Verification Runner
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting ExposureValidator runtime test execution.")

    asset_bounds = [(0.00, 0.50), (0.00, 0.40), (0.00, 0.60), (0.00, 0.70)]
    validator = ExposureValidator(
        asset_bounds=asset_bounds, tolerance=DEFAULT_TOLERANCE
    )

    print("\n" + "=" * 70)
    print("ExposureValidator Runtime Verification Suite")
    print("=" * 70)

    # 1. Standard Allocation Pass
    print("\n[1] Standard Allocation Verification")
    weights = np.array([0.25, 0.20, 0.30, 0.25])
    print("Result:", validator.validate(weights))

    # 2. Vector Scale Testing
    print("\n[2] Multi-Asset Scale Testing (N=20)")
    medium_validator = ExposureValidator([(0.0, 0.30)] * 20)
    print("Result:", medium_validator.validate(np.full(20, 0.05)))

    # 3. Large Array Scale Testing
    print("\n[3] Granular Allocation Testing (N=100)")
    large_validator = ExposureValidator([(0.0, 0.02)] * 100)
    print("Result:", large_validator.validate(np.full(100, 0.01)))

    # 4. Shape Boundary Guard
    print("\n[4] Matrix Guard Verification")
    try:
        validator.validate(np.array([[0.5, 0.5]]))
    except Exception as exc:
        print("Caught Expected Exception:", type(exc).__name__, "-", exc)

    # 5. Invalid Input Filtering
    print("\n[5] Invalid Value Filtering")
    for case in [
        np.array([0.20, np.nan, 0.30, 0.50]),
        np.array([0.20, np.inf, 0.30, 0.50]),
        np.array([-0.20, 0.30, 0.40, 0.50]),
    ]:
        try:
            validator.validate(case)
        except Exception as exc:
            print("Caught Expected Exception:", type(exc).__name__, "-", exc)

    # 6. Edge Bound Pass
    print("\n[6] Edge Boundary Verification")
    edge_validator = ExposureValidator([(0.0, 1.0)])
    print("Result:", edge_validator.validate(np.array([1.0])))

    # 7. Determinism Validation
    print("\n[7] Determinism Verification")
    first, second = validator.validate(weights), validator.validate(weights)
    print("Deterministic Output:", first == second)

    # 8. High-Dimension Stress Pass
    print("\n[8] High-Dimension Stress Verification (N=1000)")
    stress_validator = ExposureValidator([(0.0, 0.01)] * 1000)
    print(
        "Passed:",
        stress_validator.validate(np.full(1000, 0.001))["exposure_limits_passed"],
    )

    # 9. Floating-Point Tolerance Guard
    print("\n[9] Floating-Point Tolerance Guard")
    numerical_weights = np.array(
        [0.300000000001, 0.299999999999, 0.200000000000, 0.200000000000]
    )
    print("Result:", validator.validate(numerical_weights))

    # 10. Metadata Structure Compliance
    print("\n[10] Integration Contract Verification")
    report = validator.validate(weights)
    required_keys = {"exposure_limits_passed", "individual_asset_checks"}
    print("Contract Satisfied:", required_keys.issubset(report.keys()))

    print("\n" + "=" * 70)
    print("All runtime verification checks completed successfully.")
    print("=" * 70)