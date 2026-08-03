"""
Module: comparison
Compares classical MVO against heuristic greedy baselines.
"""

import logging
import numpy as np
from typing import Dict, Any

from scipy_optimizer import SciPyOptimizer
from greedy_baseline import GreedyBaselineSolver
from mean_variance import MeanVarianceObjective

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BaselineComparer:
    """
    Executes and compares multiple classical portfolio allocation strategies.
    """

    def __init__(self, risk_aversion: float = 1.0, top_k: int = 3):
        """
        Initializes the comparer with strategy parameters.
        """
        self.risk_aversion = risk_aversion
        self.top_k = top_k
        self.scipy_solver = SciPyOptimizer(risk_aversion=risk_aversion)
        self.greedy_solver = GreedyBaselineSolver(top_k=top_k)
        self.mvo_objective = MeanVarianceObjective(risk_aversion=risk_aversion)

    def compare(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Runs both baselines and evaluates their objective performance.
        """
        logger.info("Starting classical baseline comparison.")
        
        # 1. Run SciPy SLSQP (Exact MVO)
        scipy_weights, scipy_meta = self.scipy_solver.solve(expected_returns, covariance_matrix)
        
        # 2. Run Greedy Heuristic
        greedy_weights, greedy_meta = self.greedy_solver.solve(expected_returns)
        
        # 3. Evaluate Greedy weights against the MVO objective
        greedy_objective_val = self.mvo_objective.calculate_objective(
            weights=greedy_weights,
            returns=expected_returns,
            cov_matrix=covariance_matrix
        )
        
        # 4. Calculate Optimality Gap (Lower objective is better)
        scipy_objective_val = scipy_meta.get("objective_value", 0.0)
        optimality_gap = greedy_objective_val - scipy_objective_val
        
        results = {
            "scipy": {
                "weights": scipy_weights.tolist(),
                "objective_value": scipy_objective_val,
                "success": scipy_meta.get("success")
            },
            "greedy": {
                "weights": greedy_weights.tolist(),
                "objective_value": greedy_objective_val,
                "top_k_selected": greedy_meta.get("top_k_selected")
            },
            "comparison": {
                "optimality_gap": optimality_gap,
                "winner": "scipy" if scipy_objective_val < greedy_objective_val else "greedy"
            }
        }
        
        logger.info(f"Comparison complete. Optimality gap: {optimality_gap:.6f}")
        return results