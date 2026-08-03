"""
Module: scipy_optimizer
Executes classical portfolio optimization using SciPy's SLSQP solver.
"""

import logging
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Any

# Ensure your folder structure allows this import, or run from within the directory
from mean_variance import MeanVarianceObjective

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SciPyOptimizer:
    """
    Runs the classical SLSQP optimizer for portfolio allocation.
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        Initializes the SciPy optimizer.

        Args:
            risk_aversion (float): Risk tolerance parameter.
        """
        self.mvo = MeanVarianceObjective(risk_aversion=risk_aversion)

    def solve(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes the SLSQP optimization.

        Args:
            expected_returns (np.ndarray): Asset returns vector.
            covariance_matrix (np.ndarray): Asset covariance matrix.

        Returns:
            Tuple[np.ndarray, Dict]: The optimal weights and solver metadata.
        """
        n_assets = len(expected_returns)
        logger.info(f"Starting SciPy SLSQP optimization for {n_assets} assets.")
        
        # Initial guess: Equal weight distribution
        initial_weights = np.ones(n_assets) / n_assets
        
        # Bounds: Long-only (0 to 1 for each asset)
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Constraints: Sum of weights == 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Execute optimization
        result = minimize(
            fun=self.mvo.calculate_objective,
            x0=initial_weights,
            args=(expected_returns, covariance_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            tol=1e-8
        )
        
        if not result.success:
            logger.warning(f"SciPy optimization did not converge: {result.message}")
            
        optimal_weights = result.x
        
        metadata = {
            "success": result.success,
            "message": result.message,
            "objective_value": result.fun,
            "iterations": result.nit
        }
        
        logger.info("SciPy optimization execution completed.")
        return optimal_weights, metadata