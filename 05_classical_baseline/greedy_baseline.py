"""
Module: greedy_baseline
Executes a heuristic top-K portfolio allocation strategy.
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


class GreedyBaselineSolver:
    """
    Allocates portfolio weights using a greedy heuristic based purely on expected returns.
    """

    def __init__(self, top_k: int = 3):
        """
        Initializes the greedy solver.

        Args:
            top_k (int): Number of top assets to select.
        """
        self.top_k = top_k

    def solve(self, expected_returns: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes the greedy allocation.

        Args:
            expected_returns (np.ndarray): Asset returns vector.

        Returns:
            Tuple[np.ndarray, Dict]: The heuristic weights and metadata.
        """
        n_assets = len(expected_returns)
        logger.info(f"Starting Greedy allocation for top {self.top_k} out of {n_assets} assets.")
        
        # Ensure top_k is within bounds
        k = min(self.top_k, n_assets)
        
        if k <= 0:
            logger.error("top_k must be a positive integer.")
            raise ValueError("top_k must be greater than 0.")
            
        # Get indices of the top K expected returns (descending order)
        # argsort sorts ascending, so we reverse it with [::-1]
        sorted_indices = np.argsort(expected_returns)[::-1]
        top_indices = sorted_indices[:k]
        
        # Allocate equal weights to the top K assets
        weights = np.zeros(n_assets)
        weights[top_indices] = 1.0 / k
        
        metadata = {
            "strategy": "Top-K Greedy",
            "top_k_selected": k,
            "selected_indices": top_indices.tolist()
        }
        
        logger.info("Greedy allocation execution completed.")
        return weights, metadata