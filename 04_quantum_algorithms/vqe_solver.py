"""
Module: 04_quantum_algorithms.vqe_solver
Executes the Variational Quantum Eigensolver for portfolio optimization.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE

# Using v1 primitives to ensure compatibility with qiskit_algorithms eigenvalue solvers
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class VQEPortfolioSolver:
    """
    Solves the portfolio QUBO using the VQE algorithm.
    """

    def __init__(self, reps: int = 3, entanglement: str = 'linear', optimizer_iter: int = 150):
        """
        Initializes the VQE solver.

        Args:
            reps (int): Number of ansatz repetition layers.
            entanglement (str): Entanglement pattern ('linear', 'full', 'circular').
            optimizer_iter (int): Maximum classical optimization iterations.
        """
        self.reps = reps
        self.entanglement = entanglement
        self.optimizer = COBYLA(maxiter=optimizer_iter)
        self.estimator = Estimator()
        self.sampler = Sampler()

    def _qubo_to_ising(self, qubo: np.ndarray) -> SparsePauliOp:
        """
        Converts a QUBO matrix to a Qiskit SparsePauliOp (Ising Hamiltonian).
        """
        num_qubits = qubo.shape[0]
        pauli_list = []
        
        for i in range(num_qubits):
            for j in range(i, num_qubits):
                weight = qubo[i, j]
                if weight == 0:
                    continue
                    
                if i == j:
                    z_str = ['I'] * num_qubits
                    z_str[i] = 'Z'
                    pauli_list.append(("".join(z_str)[::-1], -weight / 2.0))
                else:
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
        """
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
        Executes the VQE solver on the given QUBO.
        """
        num_qubits = qubo.shape[0]
        logger.info(f"Starting VQE execution for {n_assets} assets ({num_qubits} total qubits).")
        
        # 1. Map to Hamiltonian
        hamiltonian = self._qubo_to_ising(qubo)
        
        # 2. Setup Hardware-Efficient Ansatz
        ansatz = RealAmplitudes(num_qubits, entanglement=self.entanglement, reps=self.reps)
        
        # 3. Setup VQE
        vqe = VQE(
            estimator=self.estimator,
            ansatz=ansatz,
            optimizer=self.optimizer
        )
        
        # 4. Optimize (Compute Minimum Eigenvalue)
        result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        optimal_parameters = result.optimal_point
        final_energy = result.eigenvalue.real
        
        # 5. Sample the optimal circuit to extract the bitstring
        bound_circuit = ansatz.bind_parameters(optimal_parameters)
        bound_circuit.measure_all()
        
        job = self.sampler.run(bound_circuit)
        quasi_dists = job.result().quasi_dists[0]
        
        # Find the state with the highest probability
        best_state_int = max(quasi_dists, key=quasi_dists.get)
        best_bitstring = format(best_state_int, f'0{num_qubits}b')
        
        # 6. Decode weights
        weights = self._decode_bitstring(best_bitstring, n_assets, binary_bits)
        
        metadata = {
            "optimal_parameters": optimal_parameters,
            "final_energy": final_energy,
            "best_bitstring": best_bitstring,
            "optimizer_evals": result.optimizer_evals
        }
        
        logger.info("VQE execution completed successfully.")
        return weights, metadata