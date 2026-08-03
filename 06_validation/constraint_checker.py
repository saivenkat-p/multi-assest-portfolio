"""
Module: 06 validation.constraint_checker
Validates portfolio weight vectors against mathematical and business constraints.
"""

import logging
import numpy as np
from typing import Tuple, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ConstraintChecker:
    """
    Checks portfolio weights against standard investment constraints.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initializes the constraint checker.

        Args:
            tolerance (float): Floating-point tolerance for equality bounds.
        """
        self.tolerance = tolerance

    def check_budget(self, weights: np.ndarray) -> bool:
        """
        Verifies that the sum of weights is exactly 1.0 (fully invested).
        """
        total_weight = np.sum(weights)
        is_valid = np.isclose(total_weight, 1.0, atol=self.tolerance)
        if not is_valid:
            logger.warning(f"Budget constraint failed. Sum of weights: {total_weight:.6f}")
        return bool(is_valid)

    def check_bounds(self, weights: np.ndarray) -> bool:
        """
        Verifies that all weights are between 0 and 1 (long-only).
        """
        lower_bound_valid = np.all(weights >= (0.0 - self.tolerance))
        upper_bound_valid = np.all(weights <= (1.0 + self.tolerance))
        is_valid = lower_bound_valid and upper_bound_valid
        
        if not is_valid:
            logger.warning("Bounds constraint failed. Weights fall outside [0, 1].")
        return bool(is_valid)

    def validate_portfolio(self, weights: np.ndarray) -> Tuple[bool, Dict[str, bool]]:
        """
        Executes all basic constraint checks on the portfolio.

        Args:
            weights (np.ndarray): The portfolio weight vector.

        Returns:
            Tuple[bool, Dict]: Overall validity and the detailed report.
        """
        budget_pass = self.check_budget(weights)
        bounds_pass = self.check_bounds(weights)
        
        is_valid = budget_pass and bounds_pass
        
        report = {
            "budget_constraint": budget_pass,
            "bounds_constraint": bounds_pass,
            "overall_valid": is_valid
        }
        
        if is_valid:
            logger.info("Portfolio passed all basic constraint checks.")
        else:
            logger.error("Portfolio failed constraint validation.")
            
        return is_valid, report