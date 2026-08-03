"""
Module: 03_optimization.objective_function
Provides the core objective functions to be minimized during portfolio optimization.
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


def calculate_portfolio_variance(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """
    Calculates the total variance of the portfolio.

    Args:
        weights (np.ndarray): Asset weights vector of shape (N,).
        covariance_matrix (np.ndarray): Covariance matrix of shape (N, N).

    Returns:
        float: Portfolio variance.
    """
    return float(weights.T @ covariance_matrix @ weights)


def calculate_portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """
    Calculates the expected return of the portfolio.

    Args:
        weights (np.ndarray): Asset weights vector of shape (N,).
        expected_returns (np.ndarray): Expected returns vector of shape (N,).

    Returns:
        float: Portfolio expected return.
    """
    return float(weights.T @ expected_returns)


def mean_variance_objective(
    weights: np.ndarray, 
    expected_returns: np.ndarray, 
    covariance_matrix: np.ndarray, 
    risk_aversion: float = 1.0
) -> float:
    """
    Calculates the standard Markowitz Mean-Variance objective function (cost to minimize).

    Args:
        weights (np.ndarray): Asset weights vector.
        expected_returns (np.ndarray): Expected returns vector.
        covariance_matrix (np.ndarray): Covariance matrix.
        risk_aversion (float): Trade-off parameter (gamma). 

    Returns:
        float: The objective cost. Lower is better.
    """
    port_variance = calculate_portfolio_variance(weights, covariance_matrix)
    port_return = calculate_portfolio_return(weights, expected_returns)
    
    # Cost function to minimize: (0.5 * Risk) - (Risk_Aversion * Return)
    cost = (0.5 * port_variance) - (risk_aversion * port_return)
    return cost


def sharpe_ratio_objective(
    weights: np.ndarray, 
    expected_returns: np.ndarray, 
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculates the negative Sharpe Ratio (cost to minimize).
    Useful for Stable regimes where risk-adjusted performance is prioritized.

    Args:
        weights (np.ndarray): Asset weights vector.
        expected_returns (np.ndarray): Expected returns vector.
        covariance_matrix (np.ndarray): Covariance matrix.
        risk_free_rate (float): The baseline risk-free return.

    Returns:
        float: Negative Sharpe ratio (minimizing this maximizes the actual Sharpe ratio).
    """
    port_variance = calculate_portfolio_variance(weights, covariance_matrix)
    port_return = calculate_portfolio_return(weights, expected_returns)
    
    port_std_dev = np.sqrt(port_variance)
    
    # Avoid division by zero
    if port_std_dev < 1e-8:
        return 0.0
        
    sharpe_ratio = (port_return - risk_free_rate) / port_std_dev
    
    # Return negative because optimizers typically minimize the objective
    return -sharpe_ratio