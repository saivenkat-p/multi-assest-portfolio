"""
Module: 05 classical baseline.mean_variance
Defines the mathematical formulations for Mean-Variance Optimization.
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


class MeanVarianceObjective:
    """
    Handles the mathematical objective for classical Markowitz optimization.
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        Initializes the Mean-Variance objective.

        Args:
            risk_aversion (float): Risk tolerance parameter (higher = more risk-averse).
        """
        self.risk_aversion = risk_aversion

    def calculate_objective(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Calculates the penalized portfolio variance.
        
        Args:
            weights (np.ndarray): The asset allocation vector.
            returns (np.ndarray): The expected returns for each asset.
            cov_matrix (np.ndarray): The covariance matrix of the assets.
            
        Returns:
            float: The objective value to be minimized.
        """
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_return = weights.T @ returns
        
        # We minimize (Variance - Return), effectively maximizing Return and minimizing Variance
        objective_value = float(0.5 * portfolio_variance - self.risk_aversion * portfolio_return)
        
        return objective_value