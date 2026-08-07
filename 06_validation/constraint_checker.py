# ============================================================================
# Standard Library Imports
# ============================================================================

import logging
import time
from typing import Dict, Final, Tuple

# ============================================================================
# Third-Party Imports
# ============================================================================

import numpy as np

# ============================================================================
# Module Constants
# ============================================================================

DEFAULT_FLOAT_TOLERANCE: Final[float] = 1e-6
EXPECTED_DIMENSIONS: Final[int] = 1
MINIMUM_ASSETS: Final[int] = 1

# ============================================================================
# Public API
# ============================================================================

__all__ = ["ConstraintChecker"]

# ============================================================================
# Module Logger
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


class ConstraintChecker:
    """
    Validates portfolio allocations against mathematical and
    investment constraints.

    This class performs feasibility validation only.
    It never modifies the supplied portfolio weights.
    """

    def __init__(self, tolerance: float = DEFAULT_FLOAT_TOLERANCE) -> None:
        """
        Initialize the constraint checker.

        Args:
            tolerance:
                Floating-point tolerance used for numerical comparisons.

        Raises:
            TypeError:
                If tolerance is not numeric.
            ValueError:
                If tolerance is negative or non-finite.
        """
        self.tolerance = self._validate_tolerance(tolerance)

        logger.info(
            "ConstraintChecker initialized with tolerance=%e",
            self.tolerance,
        )

    # =======================================================================
    # Validation Helpers
    # =======================================================================

    @staticmethod
    def _validate_tolerance(tolerance: float) -> float:
        """
        Validate numerical tolerance.
        """
        if not isinstance(tolerance, (float, int)):
            raise TypeError("Tolerance must be a numeric value.")

        tolerance = float(tolerance)

        if not np.isfinite(tolerance):
            raise ValueError("Tolerance must be finite.")

        if tolerance < 0.0:
            raise ValueError("Tolerance cannot be negative.")

        return tolerance

    def _cleanup_array(self, weights: np.ndarray) -> np.ndarray:
        """
        Remove numerical artifacts from a portfolio weight vector.

        Values whose absolute magnitude is smaller than the configured
        floating-point tolerance are rounded to exactly zero.
        """
        cleaned = np.asarray(weights, dtype=float).copy()

        cleaned[
            np.isclose(
                cleaned,
                0.0,
                atol=self.tolerance,
            )
        ] = 0.0

        return cleaned

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Validate the portfolio weight vector.
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                "weights must be provided as a NumPy ndarray."
            )

        if weights.ndim != EXPECTED_DIMENSIONS:
            raise ValueError(
                "Portfolio weights must be a one-dimensional array."
            )

        if weights.size < MINIMUM_ASSETS:
            raise ValueError(
                "Portfolio weight vector cannot be empty."
            )

        if not np.issubdtype(weights.dtype, np.number):
            raise TypeError(
                "Portfolio weights must contain numeric values."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "Portfolio weights contain NaN or infinite values."
            )

        return self._cleanup_array(weights)
    
    # -----------------------------------------------------------------------
    # Logging Helpers
    # -----------------------------------------------------------------------

    def _log_execution_time(
        self,
        operation: str,
        start_time: float,
    ) -> None:
        """
        Log the execution time of a validation operation.
        """
        elapsed = time.perf_counter() - start_time

        logger.info(
            "%s completed in %.6f seconds.",
            operation,
            elapsed,
        )

    # =======================================================================
    # Public API
    # =======================================================================

    def check_budget(self, weights: np.ndarray) -> bool:
        """
        Validate the portfolio budget constraint.

        The portfolio is considered feasible when:
            Σ wᵢ = 1
        within the configured numerical tolerance.
        """
        start_time = time.perf_counter()

        try:
            weights = self._validate_weights(weights)

            total_weight = float(np.sum(weights))
            if not np.isfinite(total_weight):
                raise ValueError("Portfolio weight sum must be finite.")

            is_valid = bool(
                np.isclose(
                    total_weight,
                    1.0,
                    atol=self.tolerance,
                )
            )

            if is_valid:
                logger.info(
                    "Budget constraint satisfied "
                    "(sum=%0.10f).",
                    total_weight,
                )
            else:
                logger.warning(
                    "Budget constraint failed (sum=%0.10f).",
                    total_weight,
                )

            self._log_execution_time(
                "Budget validation",
                start_time,
            )

            return is_valid
        except (TypeError, ValueError):
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error during budget validation."
            )
            raise RuntimeError(
                "Failed to validate budget constraint."
            ) from exc

    def check_bounds(self, weights: np.ndarray) -> bool:
        """
        Validate long-only portfolio bounds.

        The portfolio is considered feasible when every asset weight
        satisfies:
            0 ≤ wᵢ ≤ 1
        within the configured numerical tolerance.
        """
        start_time = time.perf_counter()

        try:
            weights = self._validate_weights(weights)

            lower_valid = bool(
                np.all(weights >= (-self.tolerance))
            )

            upper_valid = bool(
                np.all(weights <= (1.0 + self.tolerance))
            )

            is_valid = lower_valid and upper_valid

            if is_valid:
                logger.info(
                    "Bounds constraint satisfied."
                )
            else:
                logger.warning(
                    "Bounds constraint failed."
                )

            self._log_execution_time(
                "Bounds validation",
                start_time,
            )

            return is_valid

        except (TypeError, ValueError):
            # Expected validation failures.
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error during bounds validation."
            )
            raise RuntimeError(
                "Failed to validate portfolio bounds."
            ) from exc

    def validate_portfolio(
        self,
        weights: np.ndarray,
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Execute all portfolio feasibility checks.
        """
        start_time = time.perf_counter()

        try:
            weights = self._validate_weights(weights)

            logger.info(
                "Starting portfolio constraint validation for %d assets.",
                weights.size,
            )

            budget_pass = self.check_budget(weights)
            bounds_pass = self.check_bounds(weights)
            overall_valid = all(
                (
                    budget_pass,
                    bounds_pass,
                )
            )
                
            report: Dict[str, bool] = {
                "budget_constraint": budget_pass,
                "bounds_constraint": bounds_pass,
                "overall_valid": overall_valid,
            }

            if overall_valid:
                logger.info(
                    "Portfolio validation succeeded."
                )
            else:
                logger.warning(
                    "Portfolio validation failed."
                )

            self._log_execution_time(
                "Portfolio validation",
                start_time,
            )

            return overall_valid, report

        except (TypeError, ValueError):
            # Expected validation failures.
            raise

        except Exception as exc:
            logger.exception(
                "Unexpected error during portfolio validation."
            )
            raise RuntimeError(
                "Portfolio validation failed."
            ) from exc


# ============================================================================
# Runtime Example
# ============================================================================

if __name__ == "__main__":

    checker = ConstraintChecker()

    test_cases = {
        "Valid Portfolio": np.array(
            [0.20, 0.25, 0.15, 0.30, 0.10],
            dtype=float,
        ),
        "Invalid Budget": np.array(
            [0.30, 0.30, 0.20, 0.20, 0.20],  # Sum = 1.20
            dtype=float,
        ),
        "Invalid Bounds": np.array(
            [0.60, 0.30, -0.10, 0.20, 0.00],  # Negative weight
            dtype=float,
        ),
        "NaN Input": np.array(
            [0.40, np.nan, 0.30, 0.20, 0.10],
            dtype=float,
        ),
    }

    for test_name, weights in test_cases.items():

        print("\n" + "=" * 60)
        print(f"Running Test : {test_name}")
        print("=" * 60)

        try:
            is_valid, report = checker.validate_portfolio(weights)

            print(f"Portfolio Valid : {is_valid}")

            print("\nValidation Report")
            print("-" * 35)

            for key, value in report.items():
                print(f"{key:<25}: {value}")

        except (TypeError, ValueError) as exc:
            print("\nExpected Validation Failure")
            print("-" * 35)
            print(f"Test Case : {test_name}")
            print(f"Reason    : {str(exc)}")

        except Exception as exc:
            print("\nUnexpected Runtime Error")
            print("=" * 35)
            print(exc)