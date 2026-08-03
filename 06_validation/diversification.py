"""
Module: 06 validation.diversification
Calculates concentration metrics (HHI, Entropy) for portfolio weights.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DiversificationEvaluator:
    """
    Evaluates portfolio concentration using standard financial metrics.
    """

    def __init__(self, hhi_threshold: float = 0.30):
        """
        Initializes the evaluator.

        Args:
            hhi_threshold (float): Maximum acceptable HHI score (0 to 1).
        """
        self.hhi_threshold = hhi_threshold

    def calculate_hhi(self, weights: np.ndarray) -> float:
        """
        Calculates the Herfindahl-Hirschman Index (HHI).
        """
        return float(np.sum(np.square(weights)))

    def calculate_entropy(self, weights: np.ndarray) -> float:
        """
        Calculates the Shannon Entropy of the portfolio weights.
        """
        # Filter out zero weights to avoid log(0) warnings/NaNs
        non_zero_weights = weights[weights > 1e-8]
        if len(non_zero_weights) == 0:
            return 0.0
            
        entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
        return float(entropy)

    def evaluate(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Runs all diversification checks and returns a summary report.

        Args:
            weights (np.ndarray): The portfolio allocation vector.

        Returns:
            Dict: Diversification metrics and pass/fail status.
        """
        hhi_score = self.calculate_hhi(weights)
        entropy_score = self.calculate_entropy(weights)
        
        is_diversified = hhi_score <= self.hhi_threshold
        
        if not is_diversified:
            logger.warning(f"Portfolio highly concentrated. HHI ({hhi_score:.4f}) exceeds threshold ({self.hhi_threshold}).")
        else:
            logger.info(f"Portfolio passes diversification check. HHI: {hhi_score:.4f}")

        report = {
            "hhi_score": hhi_score,
            "entropy_score": entropy_score,
            "is_diversified": is_diversified,
            "hhi_threshold_used": self.hhi_threshold
        }
        
        return report