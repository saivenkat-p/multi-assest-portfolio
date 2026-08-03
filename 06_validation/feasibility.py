"""
Module: 06 validation.feasibility
Master validation module to assess overall portfolio viability.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from constraint_checker import ConstraintChecker
from diversification import DiversificationEvaluator
from exposure_limits import ExposureValidator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class FeasibilityAnalyzer:
    """
    Orchestrates all validation constraints to determine total portfolio feasibility.
    """

    def __init__(
        self, 
        asset_bounds: Optional[List[Tuple[float, float]]] = None,
        hhi_threshold: float = 0.30,
        tolerance: float = 1e-6
    ):
        """
        Initializes the feasibility analyzer and its sub-components.
        """
        self.constraint_checker = ConstraintChecker(tolerance=tolerance)
        self.diversification_evaluator = DiversificationEvaluator(hhi_threshold=hhi_threshold)
        self.exposure_validator = ExposureValidator(asset_bounds=asset_bounds, tolerance=tolerance)

    def analyze(self, weights: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Runs all validation suites against the weight vector.
        """
        logger.info("Initiating master feasibility analysis.")
        
        # 1. Basic Mathematical Constraints
        constraints_pass, constraints_report = self.constraint_checker.validate_portfolio(weights)
        
        # 2. Diversification Checks
        diversification_report = self.diversification_evaluator.evaluate(weights)
        diversification_pass = diversification_report.get("is_diversified", False)
        
        # 3. Custom Exposure Limits
        exposure_report = self.exposure_validator.validate(weights)
        exposure_pass = exposure_report.get("exposure_limits_passed", False)
        
        # 4. Master Evaluation
        is_feasible = constraints_pass and diversification_pass and exposure_pass
        
        if is_feasible:
            logger.info("Portfolio successfully passed all feasibility checks.")
        else:
            logger.warning("Portfolio failed one or more feasibility checks.")
            
        master_report = {
            "is_feasible": is_feasible,
            "constraints": constraints_report,
            "diversification": diversification_report,
            "exposure_limits": exposure_report
        }
        
        return is_feasible, master_report