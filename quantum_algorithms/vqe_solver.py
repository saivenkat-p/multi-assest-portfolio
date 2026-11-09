"""
Module: quantum_algorithms.vqe_solver

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Native PennyLane implementation of the Variational Quantum Eigensolver (VQE)
    for solving portfolio optimization problems formulated as Quadratic Unconstrained
    Binary Optimization (QUBO) models.

Workflow:
    Expected Returns & Covariance
            │
            ▼
    QUBO Formulation (HamiltonianBuilder)
            │
            ▼
    QUBO -> Ising Translation (quantum_algorithms/hamiltonian.py)
            │
            ▼
    PennyLane Parameterized HEA Circuit (quantum_algorithms/ansatz.py)
            │
            ▼
    Classical / Differentiable Optimization (quantum_algorithms/optimizer.py)
            │
            ▼
    Exact Statevector Probabilities & Ground State Sampling
            │
            ▼
    Bitstring Decoding -> Continuous Portfolio Weights (quantum_algorithms/decoding.py)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml

from quantum_algorithms.ansatz import (
    calculate_parameter_count,
    generate_initial_parameters,
    portfolio_ansatz,
)
from quantum_algorithms.backend import create_device, validate_device_name
from quantum_algorithms.decoding import (
    decode_bitstring_to_weights,
    qubo_energy_of_bitstring,
    validate_encoding_dimensions,
)
from quantum_algorithms.hamiltonian import (
    qubo_to_pennylane_hamiltonian,
    validate_qubo_matrix,
)
from quantum_algorithms.optimizer import (
    DEFAULT_OPTIMIZER,
    OptimizationResult,
    get_optimizer,
)

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_REPS = 2
DEFAULT_ENTANGLEMENT = "linear"
DEFAULT_OPTIMIZER_ITER = 120
DEFAULT_DEVICE_NAME = "default.qubit"
FLOAT_TOLERANCE = 1e-10

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =============================================================================
# VQE Solver Class
# =============================================================================

class VQEPortfolioSolver:
    """
    Production-grade PennyLane Variational Quantum Eigensolver for portfolio optimization.

    Parameters:
        device_name: Name of PennyLane device (default: 'default.qubit').
        reps: Number of ansatz layer repetitions (default: 2).
        entanglement: Entanglement topology ('linear', 'circular', 'full').
        optimizer_name: Classical optimization algorithm ('COBYLA', 'SLSQP', 'ADAM', etc.).
        optimizer_iter: Maximum optimizer iterations.
        shots: Measurement shots (None for exact analytic statevector simulation).
        seed: Random seed for deterministic simulation.
    """

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        reps: int = DEFAULT_REPS,
        entanglement: str = DEFAULT_ENTANGLEMENT,
        optimizer_name: str = DEFAULT_OPTIMIZER,
        optimizer_iter: int = DEFAULT_OPTIMIZER_ITER,
        shots: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        if not isinstance(reps, int) or reps <= 0:
            raise ValueError(f"reps must be a positive integer, got {reps}.")

        if not isinstance(optimizer_iter, int) or optimizer_iter <= 0:
            raise ValueError(f"optimizer_iter must be a positive integer, got {optimizer_iter}.")

        self.device_name = validate_device_name(device_name)
        self.reps = reps
        self.entanglement = entanglement.strip().lower()
        self.optimizer_name = optimizer_name
        self.optimizer_iter = optimizer_iter
        self.shots = shots
        self.seed = seed

        logger.info(
            "VQEPortfolioSolver initialized (device='%s', reps=%d, entanglement='%s', optimizer='%s', maxiter=%d)",
            self.device_name,
            self.reps,
            self.entanglement,
            self.optimizer_name,
            self.optimizer_iter,
        )

    def _validate_qubo(self, qubo: np.ndarray) -> np.ndarray:
        return validate_qubo_matrix(qubo)

    def _validate_encoding(self, n_assets: int, binary_bits: int, total_qubits: int) -> None:
        validate_encoding_dimensions(n_assets, binary_bits, total_qubits)

    def _decode_bitstring(self, bitstring: str, n_assets: int, binary_bits: int) -> np.ndarray:
        return decode_bitstring_to_weights(bitstring, n_assets, binary_bits, normalize=True)

    def solve(
        self,
        qubo: np.ndarray,
        n_assets: int,
        binary_bits: int,
        offset: float = 0.0,
        initial_params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute VQE to find ground state of the portfolio Ising Hamiltonian.

        Args:
            qubo: Symmetric QUBO matrix of shape (M, M).
            n_assets: Number of assets.
            binary_bits: Bits per asset.
            offset: Constant energy offset.
            initial_params: Optional initial variational parameters.

        Returns:
            Tuple of:
                - weights: Continuous optimal portfolio weights.
                - metadata: Comprehensive optimization trace and performance metrics.
        """
        start_time = time.perf_counter()
        Q = self._validate_qubo(qubo)
        num_qubits = Q.shape[0]
        self._validate_encoding(n_assets, binary_bits, num_qubits)

        logger.info(
            "Starting PennyLane VQE optimization (assets=%d, qubits=%d, reps=%d, backend='%s')",
            n_assets,
            num_qubits,
            self.reps,
            self.device_name,
        )

        try:
            # 1. Build PennyLane Cost Hamiltonian
            hamiltonian = qubo_to_pennylane_hamiltonian(Q, offset=offset, include_offset=True)

            # 2. Provision PennyLane Device
            dev = create_device(
                device_name=self.device_name,
                wires=num_qubits,
                shots=self.shots,
                seed=self.seed,
            )

            # 3. Define Variational Cost QNode
            @qml.qnode(dev)
            def vqe_cost_circuit(params: np.ndarray) -> float:
                portfolio_ansatz(
                    params=params,
                    wires=num_qubits,
                    reps=self.reps,
                    entanglement=self.entanglement,
                )
                return qml.expval(hamiltonian)

            # 4. Define Probability Distribution QNode
            @qml.qnode(dev)
            def vqe_prob_circuit(params: np.ndarray) -> np.ndarray:
                portfolio_ansatz(
                    params=params,
                    wires=num_qubits,
                    reps=self.reps,
                    entanglement=self.entanglement,
                )
                return qml.probs(wires=range(num_qubits))

            # 5. Parameter Initialization
            n_params = calculate_parameter_count(num_qubits, self.reps)
            if initial_params is not None:
                params0 = np.asarray(initial_params, dtype=float)
                if len(params0) != n_params:
                    raise ValueError(f"initial_params length must be {n_params}, got {len(params0)}.")
            else:
                params0 = generate_initial_parameters(num_qubits, self.reps, seed=self.seed)

            # 6. Execute Variational Optimization
            optimizer = get_optimizer(
                optimizer_name=self.optimizer_name,
                maxiter=self.optimizer_iter,
            )

            opt_res: OptimizationResult = optimizer.minimize(
                cost_fn=vqe_cost_circuit,
                initial_params=params0,
            )

            optimal_params = opt_res.optimal_params
            final_energy = float(opt_res.optimal_value)

            # 7. Extract Ground State Probabilities & Identify Best Bitstring
            probs = np.asarray(vqe_prob_circuit(optimal_params), dtype=float)
            best_state_idx = int(np.argmax(probs))
            best_bitstring = format(best_state_idx, f"0{num_qubits}b")

            top_indices = np.argsort(probs)[::-1][: min(16, len(probs))]
            bitstrings_dict: Dict[str, float] = {
                format(idx, f"0{num_qubits}b"): float(probs[idx]) for idx in top_indices
            }

            # 8. Decode Bitstring into Portfolio Weights
            weights = self._decode_bitstring(best_bitstring, n_assets, binary_bits)

            elapsed_time = time.perf_counter() - start_time
            best_qubo_val = qubo_energy_of_bitstring(Q, offset, best_bitstring)

            metadata: Dict[str, Any] = {
                "optimal_parameters": optimal_params,
                "final_energy": final_energy,
                "best_bitstring": best_bitstring,
                "best_qubo_value": best_qubo_val,
                "top_bitstrings": bitstrings_dict,
                "optimizer_evals": opt_res.iterations,
                "convergence_trace": opt_res.cost_history,
                "reps": self.reps,
                "entanglement": self.entanglement,
                "num_qubits": num_qubits,
                "num_assets": n_assets,
                "binary_bits": binary_bits,
                "execution_time_sec": elapsed_time,
                "solver": "VQE",
                "backend": f"PennyLane:{self.device_name}",
                "success": True if (opt_res.success and np.isfinite(final_energy)) else False,
            }

            logger.info(
                "PennyLane VQE finished successfully: Energy=%.6f, Bitstring=%s, Runtime=%.4fs",
                final_energy,
                best_bitstring,
                elapsed_time,
            )
            return weights, metadata

        except Exception as exc:
            logger.exception("VQE optimization execution failed: %s", exc)
            raise RuntimeError(f"PennyLane VQE optimization failed: {exc}") from exc
