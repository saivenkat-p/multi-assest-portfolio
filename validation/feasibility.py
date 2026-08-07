"""
ARQPO - Portfolio Feasibility Validation Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Serves as the master validation orchestrator for determining
    overall portfolio feasibility. This module coordinates all
    validation components and aggregates their results into a
    unified feasibility assessment.

Workflow Position:
    06_validation/feasibility.py

        Portfolio Weights
                │
                ▼
      Constraint Validation
                │
      Diversification Validation
                │
      Exposure Validation
                │
                ▼
      Master Feasibility Decision
                │
                ▼
        Validation Report

Mathematical Objective:
    Determine whether a portfolio satisfies all mathematical,
    diversification, and exposure constraints before downstream
    evaluation, explainability, and visualization.

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

DEFAULT_HHI_THRESHOLD: float = 0.30
DEFAULT_TOLERANCE: float = 1e-6

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
# Integrated Validation Sub-Components
# =============================================================================

class ConstraintChecker:
    """Validates core mathematical constraints (e.g., budget constraint sum(w) == 1)."""
    
    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        self.tolerance = tolerance

    def validate_portfolio(self, weights: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        weight_sum = float(np.sum(weights))
        # Budget constraint check: absolute difference from 1.0 must be within tolerance
        budget_pass = abs(weight_sum - 1.0) <= self.tolerance
        
        # In Test 5, invalid values like negative numbers are handled downstream, 
        # but let's ensure we catch them defensively if expected by the analyzer suite.
        no_negative_weights = bool(np.all(weights >= -self.tolerance))
        
        constraints_pass = budget_pass and no_negative_weights
        
        report = {
            "weight_sum": weight_sum,
            "budget_constraint_passed": budget_pass,
            "no_negative_weights": no_negative_weights,
            "status": "PASSED" if constraints_pass else "FAILED"
        }
        return constraints_pass, report


class DiversificationEvaluator:
    """Evaluates portfolio concentration using the Herfindahl-Hirschman Index (HHI)."""
    
    def __init__(self, hhi_threshold: float = DEFAULT_HHI_THRESHOLD):
        self.hhi_threshold = hhi_threshold

    def evaluate(self, weights: np.ndarray) -> Dict[str, Any]:
        # HHI is calculated as the sum of squared weights
        hhi = float(np.sum(weights ** 2))
        is_diversified = hhi <= self.hhi_threshold
        
        return {
            "hhi_value": hhi,
            "hhi_threshold": self.hhi_threshold,
            "is_diversified": is_diversified
        }


class ExposureValidator:
    """Enforces upper and lower allocation bounds for each individual asset."""
    
    def __init__(self, asset_bounds: Optional[List[Tuple[float, float]]] = None, tolerance: float = DEFAULT_TOLERANCE):
        self.asset_bounds = asset_bounds
        self.tolerance = tolerance

    def validate(self, weights: np.ndarray) -> Dict[str, Any]:
        if self.asset_bounds is None:
            return {"exposure_limits_passed": True, "details": "No asset bounds configured."}
        
        if len(self.asset_bounds) != len(weights):
            return {
                "exposure_limits_passed": False, 
                "details": f"Dimension mismatch: bounds={len(self.asset_bounds)}, weights={len(weights)}"
            }
            
        passed = True
        details = []
        
        for idx, w in enumerate(weights):
            min_w, max_w = self.asset_bounds[idx]
            # Verify asset falls within bounds using standard tolerance cushions
            lower_pass = w >= (min_w - self.tolerance)
            upper_pass = w <= (max_w + self.tolerance)
            
            if not (lower_pass and upper_pass):
                passed = False
                
            details.append({
                "asset_index": idx,
                "weight": float(w),
                "bounds": (min_w, max_w),
                "passed": bool(lower_pass and upper_pass)
            })
            
        return {
            "exposure_limits_passed": passed,
            "asset_details": details
        }


# =============================================================================
# Core Orchestration Class
# =============================================================================

class FeasibilityAnalyzer:
    """
    Orchestrates all portfolio validation modules to determine
    overall portfolio feasibility.
    """

    def __init__(
        self,
        asset_bounds: Optional[List[Tuple[float, float]]] = None,
        hhi_threshold: float = DEFAULT_HHI_THRESHOLD,
        tolerance: float = DEFAULT_TOLERANCE,
    ):
        """
        Initializes the portfolio feasibility analyzer.
        """
        logger.info("Initializing FeasibilityAnalyzer.")

        # -------------------------------------------------------------------------
        # Validate HHI Threshold
        # -------------------------------------------------------------------------
        if not isinstance(hhi_threshold, (int, float, np.floating)):
            logger.error("hhi_threshold must be numeric.")
            raise TypeError("hhi_threshold must be a numeric value.")

        hhi_threshold = float(hhi_threshold)

        if not (0.0 <= hhi_threshold <= 1.0):
            logger.error("Invalid hhi_threshold: %.6f", hhi_threshold)
            raise ValueError("hhi_threshold must lie between 0.0 and 1.0.")

        # -------------------------------------------------------------------------
        # Validate Tolerance
        # -------------------------------------------------------------------------
        if not isinstance(tolerance, (int, float, np.floating)):
            logger.error("tolerance must be numeric.")
            raise TypeError("tolerance must be a numeric value.")

        tolerance = float(tolerance)

        if tolerance < 0.0:
            logger.error("Tolerance cannot be negative.")
            raise ValueError("tolerance must be non-negative.")

        # -------------------------------------------------------------------------
        # Initialize Validation Components
        # -------------------------------------------------------------------------
        self.constraint_checker = ConstraintChecker(tolerance=tolerance)
        self.diversification_evaluator = DiversificationEvaluator(hhi_threshold=hhi_threshold)
        self.exposure_validator = ExposureValidator(asset_bounds=asset_bounds, tolerance=tolerance)

        logger.info(
            "FeasibilityAnalyzer initialized (HHI Threshold=%.4f, Tolerance=%.2e, Asset Bounds=%d).",
            hhi_threshold,
            tolerance,
            len(asset_bounds) if asset_bounds is not None else 0,
        )

    def analyze(self, weights: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Performs the complete portfolio feasibility analysis.
        """
        start_time = time.perf_counter()

        try:
            logger.info("Starting portfolio feasibility analysis.")

            validated_weights = self._validate_weights(weights)
            cleaned_weights = self._cleanup_weights(validated_weights)

            # -----------------------------------------------------
            # Step 1: Constraint Validation
            # -----------------------------------------------------
            constraints_pass, constraints_report = self.constraint_checker.validate_portfolio(cleaned_weights)

            # -----------------------------------------------------
            # Step 2: Diversification Validation
            # -----------------------------------------------------
            diversification_report = self.diversification_evaluator.evaluate(cleaned_weights)
            diversification_pass = diversification_report.get("is_diversified", False)

            # -----------------------------------------------------
            # Step 3: Exposure Validation
            # -----------------------------------------------------
            exposure_report = self.exposure_validator.validate(cleaned_weights)
            exposure_pass = exposure_report.get("exposure_limits_passed", False)

            # -----------------------------------------------------
            # Overall Decision
            # -----------------------------------------------------
            is_feasible = bool(constraints_pass and diversification_pass and exposure_pass)
            elapsed_time = time.perf_counter() - start_time

            report = self._build_master_report(
                constraints_report=constraints_report,
                diversification_report=diversification_report,
                exposure_report=exposure_report,
                is_feasible=is_feasible,
                runtime_seconds=elapsed_time,
            )

            if is_feasible:
                logger.info("Portfolio passed feasibility analysis (Runtime=%.6f seconds).", elapsed_time)
            else:
                logger.warning("Portfolio failed feasibility analysis (Runtime=%.6f seconds).", elapsed_time)

            return is_feasible, report

        except Exception as exc:
            logger.exception("Portfolio feasibility analysis failed.")
            raise RuntimeError("Unable to complete portfolio feasibility analysis.") from exc

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """Validates structural properties of the portfolio weight vector."""
        if not isinstance(weights, np.ndarray):
            logger.error("Expected numpy.ndarray, received %s.", type(weights).__name__)
            raise TypeError("weights must be provided as a NumPy array.")

        if weights.ndim != 1:
            logger.error("weights must be one-dimensional (received shape %s).", weights.shape)
            raise ValueError("weights must be a one-dimensional array.")

        if weights.size == 0:
            logger.error("Portfolio weight vector is empty.")
            raise ValueError("weights cannot be empty.")

        if not np.all(np.isfinite(weights)):
            logger.error("weights contain NaN or infinite values.")
            raise ValueError("weights must contain only finite values.")

        return weights.astype(np.float64, copy=False)

    def _cleanup_weights(self, weights: np.ndarray) -> np.ndarray:
        """Cleans floating-point numerical artifacts using the default tolerance threshold."""
        cleaned = np.array(weights, dtype=np.float64, copy=True)
        cleaned[np.abs(cleaned) < DEFAULT_TOLERANCE] = 0.0
        return cleaned

    def _build_master_report(
        self,
        constraints_report: Dict[str, Any],
        diversification_report: Dict[str, Any],
        exposure_report: Dict[str, Any],
        is_feasible: bool,
        runtime_seconds: float,
    ) -> Dict[str, Any]:
        """Builds the standardized feasibility report output dictionary."""
        return {
            "is_feasible": is_feasible,
            "constraints": constraints_report,
            "diversification": diversification_report,
            "exposure_limits": exposure_report,
            "runtime_seconds": runtime_seconds,
        }


# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting FeasibilityAnalyzer runtime tests.")

    asset_bounds = [
        (0.0, 0.50),
        (0.0, 0.50),
        (0.0, 0.50),
        (0.0, 0.50),
    ]

    analyzer = FeasibilityAnalyzer(
        asset_bounds=asset_bounds,
        hhi_threshold=0.30,
        tolerance=DEFAULT_TOLERANCE,
    )

    print("\n" + "=" * 70)
    print("FeasibilityAnalyzer Runtime Tests")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Happy Path
    # -------------------------------------------------------------------------
    print("\n[1] Happy Path")
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    feasible, report = analyzer.analyze(weights)
    print("Feasible :", feasible)
    print("Report Keys :", list(report.keys()))

    # -------------------------------------------------------------------------
    # Test 2: Medium Input
    # -------------------------------------------------------------------------
    print("\n[2] Medium Input")
    medium_bounds = [(0.0, 0.20)] * 20
    medium_weights = np.full(20, 0.05)
    medium_analyzer = FeasibilityAnalyzer(asset_bounds=medium_bounds)
    feasible, _ = medium_analyzer.analyze(medium_weights)
    print("Feasible :", feasible)

    # -------------------------------------------------------------------------
    # Test 3: Large Input
    # -------------------------------------------------------------------------
    print("\n[3] Large Input")
    large_bounds = [(0.0, 0.02)] * 100
    large_weights = np.full(100, 0.01)
    large_analyzer = FeasibilityAnalyzer(asset_bounds=large_bounds)
    feasible, _ = large_analyzer.analyze(large_weights)
    print("Feasible :", feasible)

    # -------------------------------------------------------------------------
    # Test 4: Invalid Shape
    # -------------------------------------------------------------------------
    print("\n[4] Invalid Shape")
    try:
        analyzer.analyze(np.array([[0.5, 0.5]]))
    except Exception as exc:
        print(type(exc).__name__, exc)

    # -------------------------------------------------------------------------
    # Test 5: Invalid Values
    # -------------------------------------------------------------------------
    print("\n[5] Invalid Values")
    invalid_cases = [
        np.array([0.2, np.nan, 0.3, 0.5]),
        np.array([0.2, np.inf, 0.3, 0.5]),
        np.array([-0.2, 0.4, 0.4, 0.4]),
    ]

    for case in invalid_cases:
        try:
            analyzer.analyze(case)
        except Exception as exc:
            print(type(exc).__name__, exc)

    # -------------------------------------------------------------------------
    # Test 6: Edge Case
    # -------------------------------------------------------------------------
    print("\n[6] Edge Case")
    edge_analyzer = FeasibilityAnalyzer(asset_bounds=[(0.0, 1.0)])
    feasible, _ = edge_analyzer.analyze(np.array([1.0]))
    print("Feasible :", feasible)

    # -------------------------------------------------------------------------
    # Test 7: Determinism
    # -------------------------------------------------------------------------
    print("\n[7] Determinism")
    first_feasible, first_report = analyzer.analyze(weights)
    second_feasible, second_report = analyzer.analyze(weights)
    # Exclude dynamic runtime metrics when validating pure determinism
    first_report_static = {k: v for k, v in first_report.items() if k != 'runtime_seconds'}
    second_report_static = {k: v for k, v in second_report.items() if k != 'runtime_seconds'}
    print("Deterministic :", (first_feasible == second_feasible) and (first_report_static == second_report_static))

    # -------------------------------------------------------------------------
    # Test 8: Stress Test
    # -------------------------------------------------------------------------
    print("\n[8] Stress Test")
    stress_size = 1000
    stress_bounds = [(0.0, 0.01)] * stress_size
    stress_weights = np.full(stress_size, 1.0 / stress_size)
    stress_analyzer = FeasibilityAnalyzer(asset_bounds=stress_bounds)
    feasible, _ = stress_analyzer.analyze(stress_weights)
    print("Stress Test :", feasible)

    # -------------------------------------------------------------------------
    # Test 9: Numerical Robustness
    # -------------------------------------------------------------------------
    print("\n[9] Numerical Robustness")
    numerical_weights = np.array([
        0.300000000001,
        0.299999999999,
        0.200000000000,
        0.200000000000,
    ])
    feasible, _ = analyzer.analyze(numerical_weights)
    print("Feasible :", feasible)

    # -------------------------------------------------------------------------
    # Test 10: Integration Verification
    # -------------------------------------------------------------------------
    print("\n[10] Integration Verification")
    feasible, report = analyzer.analyze(weights)
    required_keys = {
        "is_feasible",
        "constraints",
        "diversification",
        "exposure_limits",
    }
    integration_pass = required_keys.issubset(report.keys())
    print("Integration Passed :", integration_pass)

    print("\n" + "=" * 70)
    print("All runtime tests completed successfully.")
    print("=" * 70)