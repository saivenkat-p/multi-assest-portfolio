"""
Module: 03_optimization.portfolio_metrics
Calculates standard financial performance metrics for a given portfolio weight vector.
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def evaluate_portfolio(
    weights: np.ndarray, 
    expected_returns: np.ndarray, 
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Evaluates the mathematical performance metrics of a portfolio.

    Args:
        weights (np.ndarray): The asset allocation vector.
        expected_returns (np.ndarray): The expected returns for each asset.
        covariance_matrix (np.ndarray): The covariance matrix of the assets.
        risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation.

    Returns:
        Dict[str, float]: A dictionary containing Return, Volatility, Variance, and Sharpe Ratio.
    """
    try:
        # Calculate Variance
        variance = float(weights.T @ covariance_matrix @ weights)
        
        # Calculate Volatility (Standard Deviation)
        volatility = np.sqrt(variance)
        
        # Calculate Expected Return
        expected_return = float(weights.T @ expected_returns)
        
        # Calculate Sharpe Ratio (with safety check for division by zero)
        if volatility < 1e-8:
            logger.warning("Portfolio volatility is near zero. Sharpe ratio set to 0.0.")
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (expected_return - risk_free_rate) / volatility

        metrics = {
            "Expected_Return": expected_return,
            "Volatility": volatility,
            "Variance": variance,
            "Sharpe_Ratio": sharpe_ratio
        }
        
        logger.debug(f"Calculated Portfolio Metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        raise