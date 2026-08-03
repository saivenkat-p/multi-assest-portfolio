"""
Module: 03_optimization.transaction_cost
Calculates penalties associated with rebalancing portfolio weights.
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


def calculate_rebalancing_cost(
    current_weights: np.ndarray, 
    target_weights: np.ndarray, 
    cost_rate: float = 0.001
) -> float:
    """
    Calculates the transaction cost to rebalance the portfolio.

    Args:
        current_weights (np.ndarray): The existing asset weights.
        target_weights (np.ndarray): The optimized asset weights.
        cost_rate (float): The fee rate per unit of weight traded.

    Returns:
        float: The total transaction cost.
    """
    if current_weights.shape != target_weights.shape:
        logger.error("Weight array dimensions do not match.")
        raise ValueError("current_weights and target_weights must have the same shape.")

    # Calculate absolute turnover for each asset
    weight_changes = np.abs(target_weights - current_weights)
    total_turnover = float(np.sum(weight_changes))
    
    # Calculate total financial cost
    total_cost = total_turnover * cost_rate
    
    logger.debug(f"Total Turnover: {total_turnover:.4f} | Transaction Cost: {total_cost:.4f}")
    return total_cost