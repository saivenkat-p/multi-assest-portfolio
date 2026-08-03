"""
Module: 07_evaluation.benchmark
Generates comparative performance reports across all quantum and classical solvers.
"""

import logging
import numpy as np
import os
import importlib.util
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Robust import for directories containing spaces
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, '..', '05 classical baseline', 'mean_variance.py'))

try:
    spec = importlib.util.spec_from_file_location("mean_variance", module_path)
    mean_variance_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mean_variance_module)
    MeanVarianceObjective = mean_variance_module.MeanVarianceObjective
    logger.info("Successfully imported MeanVarianceObjective.")
except Exception as e:
    logger.error(f"Failed to import MeanVarianceObjective from {module_path}. Error: {e}")
    MeanVarianceObjective = None


class BenchmarkEvaluator:
    """
    Evaluates and compares different solver outputs against the same objective function.
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        Initializes the benchmark evaluator.
        """
        self.risk_aversion = risk_aversion
        try:
            self.mvo = MeanVarianceObjective(risk_aversion=self.risk_aversion)
        except NameError:
            self.mvo = None
            logger.error("MeanVarianceObjective class is not available.")

    def _evaluate_weights(self, weights_dict: Dict[str, np.ndarray], returns: np.ndarray, cov: np.ndarray) -> Dict[str, float]:
        """
        Calculates the objective value for a dictionary of solver weights.
        """
        results = {}
        for solver_name, weights in weights_dict.items():
            if self.mvo and weights is not None:
                obj_val = self.mvo.calculate_objective(weights, returns, cov)
                results[solver_name] = obj_val
            else:
                results[solver_name] = None
        return results

    def run_benchmark(
        self, 
        expected_returns: np.ndarray, 
        covariance_matrix: np.ndarray, 
        quantum_weights: Dict[str, np.ndarray], 
        classical_weights: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Executes the comparative benchmark.
        """
        logger.info("Starting benchmarking process across all solvers.")
        
        # Calculate objective values
        q_results = self._evaluate_weights(quantum_weights, expected_returns, covariance_matrix)
        c_results = self._evaluate_weights(classical_weights, expected_returns, covariance_matrix)
        
        # Merge results
        all_results = {**c_results, **q_results}
        
        # Determine ground truth (SciPy is assumed to be the exact continuous minimum)
        ground_truth_val = all_results.get("scipy")
        
        optimality_gaps = {}
        if ground_truth_val is not None:
            for solver, obj_val in all_results.items():
                if obj_val is not None:
                    # Gap is the difference from the absolute minimum (lower objective is better)
                    optimality_gaps[solver] = obj_val - ground_truth_val
                else:
                    optimality_gaps[solver] = None
        
        report = {
            "objective_values": all_results,
            "optimality_gaps": optimality_gaps,
            "ground_truth_solver": "scipy",
            "risk_aversion_used": self.risk_aversion
        }
        
        logger.info("Benchmarking process completed.")
        return report