"""
Module: 06 validation.exposure_limits
Validates portfolio weights against specific asset boundary constraints.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ExposureValidator:
    """
    Checks portfolio weights against custom individual asset exposure limits.
    """

    def __init__(self, asset_bounds: List[Tuple[float, float]] = None, tolerance: float = 1e-6):
        """
        Initializes the exposure validator.

        Args:
            asset_bounds (List[Tuple[float, float]]): The (min, max) limits for each asset.
            tolerance (float): Floating-point tolerance for edge conditions.
        """
        self.asset_bounds = asset_bounds
        self.tolerance = tolerance

    def check_asset_bounds(self, weights: np.ndarray) -> Tuple[bool, List[bool]]:
        """
        Validates individual weights against their specific limits.
        """
        if not self.asset_bounds:
            # If no bounds are provided, assume all weights pass automatically
            return True, [True] * len(weights)
        
        if len(weights) != len(self.asset_bounds):
            logger.error(f"Dimension mismatch: {len(weights)} weights vs {len(self.asset_bounds)} bounds.")
            raise ValueError("Length of weights array must match length of asset_bounds.")

        validations = []
        for i, (w, (min_w, max_w)) in enumerate(zip(weights, self.asset_bounds)):
            is_within_bounds = (min_w - self.tolerance) <= w <= (max_w + self.tolerance)
            validations.append(is_within_bounds)
            if not is_within_bounds:
                logger.debug(f"Asset {i} weight {w:.4f} violated bounds [{min_w}, {max_w}].")

        all_valid = all(validations)
        if not all_valid:
            logger.warning("One or more assets violated specific exposure limits.")
        else:
            logger.info("All assets are within specified exposure limits.")

        return all_valid, validations

    def validate(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Executes the exposure checks and returns a formatted report.

        Args:
            weights (np.ndarray): The portfolio allocation vector.

        Returns:
            Dict: Exposure validation results.
        """
        assets_pass, specific_validations = self.check_asset_bounds(weights)
        
        report = {
            "exposure_limits_passed": assets_pass,
            "individual_asset_checks": specific_validations
        }
        
        return report