"""
Module: 03_optimization.hamiltonian_builder
Translates the classical Mean-Variance problem into a QUBO formulation for Quantum Solvers.
"""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HamiltonianBuilder:
    """
    Constructs QUBO matrices and prepares the Hamiltonian for Quantum optimization.
    """

    def __init__(self, risk_aversion: float = 1.0, penalty_factor: float = 1000.0, binary_bits: int = 3):
        """
        Initializes the builder with system parameters.

        Args:
            risk_aversion (float): Gamma parameter for risk/return tradeoff.
            penalty_factor (float): Lambda multiplier for budget constraint violations.
            binary_bits (int): Number of qubits used to discretize a single asset weight.
        """
        self.risk_aversion = risk_aversion
        self.penalty_factor = penalty_factor
        self.binary_bits = binary_bits
        self.max_val = (2 ** self.binary_bits) - 1

    def build_qubo(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Builds the QUBO matrix Q and constant offset for the optimization problem.

        Args:
            expected_returns (np.ndarray): Shape (N,).
            covariance_matrix (np.ndarray): Shape (N, N).

        Returns:
            Tuple[np.ndarray, float]: The QUBO matrix Q of shape (N*K, N*K) and the offset constant.
        """
        n_assets = len(expected_returns)
        total_qubits = n_assets * self.binary_bits
        
        logger.info(f"Building QUBO for {n_assets} assets using {total_qubits} total qubits.")

        # Initialize the Q matrix
        Q = np.zeros((total_qubits, total_qubits))
        offset = self.penalty_factor  # From the lambda * (-1)^2 term in budget constraint

        # Discretization scaling factor
        scale = 1.0 / self.max_val

        # 1. Variance Term: 0.5 * w^T * Cov * w
        for i in range(n_assets):
            for j in range(n_assets):
                for k1 in range(self.binary_bits):
                    for k2 in range(self.binary_bits):
                        q_idx1 = i * self.binary_bits + k1
                        q_idx2 = j * self.binary_bits + k2
                        
                        coeff = 0.5 * covariance_matrix[i, j] * scale * scale * (2 ** k1) * (2 ** k2)
                        Q[q_idx1, q_idx2] += coeff

        # 2. Expected Returns Term: -gamma * mu^T * w (Linear terms put on diagonal)
        for i in range(n_assets):
            for k in range(self.binary_bits):
                q_idx = i * self.binary_bits + k
                coeff = -self.risk_aversion * expected_returns[i] * scale * (2 ** k)
                Q[q_idx, q_idx] += coeff

        # 3. Budget Constraint Penalty: lambda * (sum(w) - 1)^2
        # Expansion: lambda * [ sum_{i} w_i^2 + sum_{i!=j} w_i w_j - 2 sum_{i} w_i + 1 ]
        for i in range(n_assets):
            for j in range(n_assets):
                for k1 in range(self.binary_bits):
                    for k2 in range(self.binary_bits):
                        q_idx1 = i * self.binary_bits + k1
                        q_idx2 = j * self.binary_bits + k2
                        
                        # Quadratic part: sum_{i} w_i^2 + sum_{i!=j} w_i w_j
                        coeff_quad = self.penalty_factor * scale * scale * (2 ** k1) * (2 ** k2)
                        Q[q_idx1, q_idx2] += coeff_quad

        for i in range(n_assets):
            for k in range(self.binary_bits):
                q_idx = i * self.binary_bits + k
                
                # Linear part: -2 * lambda * sum_{i} w_i
                coeff_lin = -2.0 * self.penalty_factor * scale * (2 ** k)
                Q[q_idx, q_idx] += coeff_lin

        # Make matrix symmetric to avoid upper/lower triangle execution errors in quantum backends
        Q = (Q + Q.T) / 2.0
        
        # Binary math property: x^2 = x. Add diagonal elements to handle this reduction naturally.
        logger.info("QUBO matrix construction completed.")
        return Q, offset