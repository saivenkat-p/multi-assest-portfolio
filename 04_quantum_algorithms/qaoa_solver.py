"""
Module: 04_quantum_algorithms.qaoa_solver
Executes the Quantum Approximate Optimization Algorithm for portfolio optimization.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any

# Qiskit imports
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class QAOAPortfolioSolver:
    """
    Solves the portfolio QUBO using the QAOA algorithm.
    """

    def __init__(self, reps: int = 2, optimizer_iter: int = 100):
        """
        Initializes the QAOA solver.

        Args:
            reps (int): Number of alternating layers (p).
            optimizer_iter (int): Maximum classical optimization iterations.
        """
        self.reps = reps
        self.optimizer = COBYLA(maxiter=optimizer_iter)
        # Using reference primitives for local execution; can be swapped for IBM Runtime
        self.sampler = Sampler()
        self.estimator = Estimator()

    def _qubo_to_ising(self, qubo: np.ndarray) -> SparsePauliOp:
        """
        Converts a QUBO matrix to a Qiskit SparsePauliOp (Ising Hamiltonian).

        Args:
            qubo (np.ndarray): Symmetric QUBO matrix.

        Returns:
            SparsePauliOp: The mapped Cost Hamiltonian.
        """
        num_qubits = qubo.shape[0]
        pauli_list = []
        
        # x_i = (I - Z_i)/2
        # x_i * x_j = (I - Z_i - Z_j + Z_i * Z_j) / 4
        
        for i in range(num_qubits):
            for j in range(i, num_qubits):
                weight = qubo[i, j]
                if weight == 0:
                    continue
                    
                if i == j:
                    # Linear term
                    z_str = ['I'] * num_qubits
                    z_str[i] = 'Z'
                    pauli_list.append(("".join(z_str)[::-1], -weight / 2.0))
                else:
                    # Quadratic term. Note QUBO is symmetric, so we take 2 * Q[i,j] for off-diagonals
                    weight *= 2.0
                    
                    z_str_i = ['I'] * num_qubits
                    z_str_i[i] = 'Z'
                    pauli_list.append(("".join(z_str_i)[::-1], -weight / 4.0))
                    
                    z_str_j = ['I'] * num_qubits
                    z_str_j[j] = 'Z'
                    pauli_list.append(("".join(z_str_j)[::-1], -weight / 4.0))
                    
                    z_str_ij = ['I'] * num_qubits
                    z_str_ij[i] = 'Z'
                    z_str_ij[j] = 'Z'
                    pauli_list.append(("".join(z_str_ij)[::-1], weight / 4.0))
                    
        return SparsePauliOp.from_list(pauli_list).simplify()

    def _decode_bitstring(self, bitstring: str, n_assets: int, binary_bits: int) -> np.ndarray:
        """
        Converts the measured bitstring back into continuous weights.

        Args:
            bitstring (str): Binary string measured from the quantum circuit.
            n_assets (int): Total number of assets.
            binary_bits (int): Qubits per asset.

        Returns:
            np.ndarray: Decoded weight vector.
        """
        # Note: Qiskit returns bitstrings in reverse order (little-endian)
        # Assuming the Hamiltonian was built with standard ordering, reverse the string back
        bitstring = bitstring[::-1]
        
        weights = np.zeros(n_assets)
        max_val = (2 ** binary_bits) - 1
        
        for i in range(n_assets):
            chunk = bitstring[i * binary_bits : (i + 1) * binary_bits]
            integer_val = int(chunk, 2)
            weights[i] = integer_val / max_val
            
        return weights

    def solve(self, qubo: np.ndarray, n_assets: int, binary_bits: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes the QAOA solver on the given QUBO.

        Args:
            qubo (np.ndarray): The objective matrix.
            n_assets (int): Total assets in the portfolio.
            binary_bits (int): Qubits utilized per asset.

        Returns:
            Tuple[np.ndarray, Dict]: Decoded weights and execution metadata.
        """
        logger.info(f"Starting QAOA execution for {n_assets} assets ({qubo.shape[0]} total qubits).")
        
        # 1. Map to Hamiltonian
        hamiltonian = self._qubo_to_ising(qubo)
        
        # 2. Setup QAOA
        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=self.optimizer,
            reps=self.reps
        )
        
        # 3. Optimize (Compute Minimum Eigenvalue)
        # For qiskit_algorithms >= 0.3.0, QAOA is a MinimumEigensolver
        result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)
        
        # 4. Extract most probable state
        best_bitstring = result.best_measurement['bitstring']
        final_energy = result.eigenvalue.real
        
        # 5. Decode weights
        weights = self._decode_bitstring(best_bitstring, n_assets, binary_bits)
        
        metadata = {
            "optimal_parameters": result.optimal_point,
            "final_energy": final_energy,
            "best_bitstring": best_bitstring,
            "optimizer_evals": result.optimizer_evals
        }
        
        logger.info("QAOA execution completed successfully.")
        return weights, metadata