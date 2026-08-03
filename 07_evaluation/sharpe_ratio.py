"""
Module: 07_evaluation.sharpe_ratio
Calculates the risk-adjusted Sharpe Ratio for portfolio allocations.
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


class SharpeRatioCalculator:
    """
    Computes the Sharpe Ratio to evaluate risk-adjusted returns.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initializes the calculator.

        Args:
            risk_free_rate (float): The theoretical return of a zero-risk investment.
        """
        self.risk_free_rate = risk_free_rate

    def calculate(self, weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Calculates the Sharpe Ratio for a given weight distribution.

        Args:
            weights (np.ndarray): Asset allocation vector.
            expected_returns (np.ndarray): Expected returns for the assets.
            cov_matrix (np.ndarray): Covariance matrix of the assets.

        Returns:
            float: The Sharpe Ratio.
        """
        # Calculate expected portfolio return
        portfolio_return = float(np.dot(weights, expected_returns))
        
        # Calculate portfolio variance
        portfolio_variance = float(weights.T @ cov_matrix @ weights)
        
        # Safety check to avoid division by zero or negative sqrt
        if portfolio_variance <= 1e-8:
            logger.warning("Portfolio variance is near zero. Returning 0.0 for Sharpe Ratio.")
            return 0.0
            
        # Calculate portfolio volatility (standard deviation)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        logger.info(f"Sharpe Ratio calculated: {sharpe_ratio:.4f} (Return: {portfolio_return:.4f}, Volatility: {portfolio_volatility:.4f})")
        return sharpe_ratio