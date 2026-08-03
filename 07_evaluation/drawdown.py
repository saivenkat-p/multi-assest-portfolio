"""
Module: 07_evaluation.drawdown
Calculates the Maximum Drawdown metric for historical risk evaluation.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DrawdownCalculator:
    """
    Computes historical drawdown metrics for a given portfolio allocation.
    """

    def __init__(self):
        """
        Initializes the drawdown calculator.
        """
        pass

    def calculate(self, historical_prices: np.ndarray, weights: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Calculates Maximum Drawdown (MDD) from a historical price matrix.

        Args:
            historical_prices (np.ndarray): 2D array of prices (rows = time, cols = assets).
            weights (np.ndarray): 1D array of portfolio weights.

        Returns:
            Tuple[float, Dict]: The max drawdown value and the associated time series data.
        """
        logger.info("Calculating Maximum Drawdown for the portfolio.")
        
        # 1. Calculate relative daily price changes (returns)
        daily_returns = historical_prices[1:] / historical_prices[:-1] - 1.0
        
        # 2. Apply weights to get daily portfolio returns
        portfolio_daily_returns = np.dot(daily_returns, weights)
        
        # 3. Calculate cumulative portfolio value (starting at 1.0)
        cumulative_value = np.cumprod(1.0 + portfolio_daily_returns)
        cumulative_value = np.insert(cumulative_value, 0, 1.0)
        
        # 4. Calculate the running maximum (peak)
        running_max = np.maximum.accumulate(cumulative_value)
        
        # 5. Calculate drawdown at each step
        # Set a floor on running_max to avoid division by zero
        running_max_safe = np.where(running_max == 0, 1e-8, running_max)
        drawdowns = (running_max_safe - cumulative_value) / running_max_safe
        
        # 6. Extract the maximum drawdown
        max_drawdown = float(np.max(drawdowns))
        
        timeseries_data = {
            "cumulative_value": cumulative_value,
            "running_max": running_max,
            "drawdowns": drawdowns
        }
        
        logger.info(f"Maximum Drawdown calculated: {max_drawdown:.2%}")
        return max_drawdown, timeseries_data