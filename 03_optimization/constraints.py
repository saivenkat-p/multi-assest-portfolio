"""
Module: 03_optimization.constraints
Provides constraints and penalty functions for portfolio optimization.
"""

import numpy as np
from typing import Tuple, Dict, Any

def get_budget_constraint() -> Dict[str, Any]:
    """
    Creates a Scipy-compatible equality constraint ensuring weights sum to 1.0.
    
    Returns:
        Dict[str, Any]: Dictionary defining the equality constraint.
    """
    return {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1.0
    }

def get_weight_bounds(n_assets: int, lower_bound: float = 0.0, upper_bound: float = 1.0) -> Tuple[Tuple[float, float], ...]:
    """
    Generates optimization bounds for individual asset weights.
    
    Args:
        n_assets (int): Number of assets in the portfolio.
        lower_bound (float): Minimum weight per asset (default 0.0 for long-only).
        upper_bound (float): Maximum weight per asset (default 1.0).
        
    Returns:
        Tuple[Tuple[float, float], ...]: A sequence of (min, max) bounds for each asset.
    """
    return tuple((lower_bound, upper_bound) for _ in range(n_assets))

def budget_penalty(weights: np.ndarray, penalty_factor: float = 1000.0) -> float:
    """
    Calculates a quadratic penalty for violating the budget constraint (sum = 1).
    Used to convert constrained problems into unconstrained QUBO models for Quantum Solvers.
    
    Args:
        weights (np.ndarray): Current asset weights.
        penalty_factor (float): Scaling factor to enforce strict adherence.
        
    Returns:
        float: The computed penalty value (0.0 if perfectly constrained).
    """
    return penalty_factor * (np.sum(weights) - 1.0)**2