"""
Module: 07_evaluation.turnover
Calculates the turnover rate between two portfolio allocations.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TurnoverCalculator:
    """
    Computes the portfolio turnover required to achieve a target allocation.
    """

    def __init__(self):
        """
        Initializes the turnover calculator.
        """
        pass

    def calculate(self, current_weights: np.ndarray, target_weights: np.ndarray) -> float:
        """
        Calculates the turnover rate.

        Args:
            current_weights (np.ndarray): The starting allocation.
            target_weights (np.ndarray): The desired target allocation.

        Returns:
            float: The turnover rate (0.0 to 1.0).
        """
        if len(current_weights) != len(target_weights):
            logger.error("Mismatch in array lengths for turnover calculation.")
            raise ValueError("current_weights and target_weights must have the same length.")

        # Calculate absolute element-wise differences
        absolute_diffs = np.abs(target_weights - current_weights)
        
        # Calculate standard turnover (sum of absolute diffs divided by 2)
        turnover = float(np.sum(absolute_diffs) / 2.0)
        
        logger.info(f"Portfolio turnover calculated: {turnover:.2%}")
        return turnover