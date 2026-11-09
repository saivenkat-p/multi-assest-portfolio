"""
Module: quantum_algorithms.qaoa_solver

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Native PennyLane implementation of the Quantum Approximate Optimization
    Algorithm (QAOA) for portfolio optimization problems formulated as
    Quadratic Unconstrained Binary Optimization (QUBO) models.

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
    PennyLane Parameterized QAOA Circuit (QNode)
            │
            ▼
    Classical / Differentiable Optimization (quantum_algorithms/optimizer.py)
            │
            ▼
    Optimal State Probabilities & Measurement Sampling
            │
            ▼
    Bitstring Decoding -> Optimal Portfolio Weights (quantum_algorithms/decoding.py)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml

from quantum_algorithms.backend import create_device, validate_device_name
from quantum_algorithms.decoding import (
    decode_bitstring_to_weights,
    qubo_energy_of_bitstring,
    validate_encoding_dimensions,
)
from quantum_algorithms.hamiltonian import (
    create_qaoa_mixer_hamiltonian,
    qubo_to_ising_coefficients,
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
DEFAULT_OPTIMIZER_ITER = 100
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
# QAOA Circuit Layer Primitives
# =============================================================================

def apply_cost_layer(gamma: float, h: np.ndarray, J: np.ndarray, num_wires: int) -> None:
    r"""
    Apply QAOA phase-separator unitary U_C(gamma) = exp(-i * gamma * H_C).

    Decomposes diagonal Ising Hamiltonian into single-qubit RZ and two-qubit IsingZZ gates:
        RZ(theta) = exp(-i * theta/2 * Z)  ==> theta = 2 * gamma * h_i
        IsingZZ(theta) = exp(-i * theta/2 * Z \otimes Z) ==> theta = 2 * gamma * J_ij

    Args:
        gamma: Rotation parameter for cost layer.
        h: Linear coefficients of shape (num_wires,).
        J: Quadratic interaction matrix of shape (num_wires, num_wires).
        num_wires: Number of quantum wires.
    """
    # Linear Z rotations
    for i in range(num_wires):
        coeff = float(h[i])
        if abs(coeff) > FLOAT_TOLERANCE:
            qml.RZ(2.0 * gamma * coeff, wires=i)

    # Quadratic ZZ interactions
    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            coupling = float(J[i, j])
            if abs(coupling) > FLOAT_TOLERANCE:
                qml.IsingZZ(2.0 * gamma * coupling, wires=[i, j])


def apply_mixer_layer(beta: float, num_wires: int) -> None:
    """
    Apply QAOA transverse-field mixer unitary U_M(beta) = exp(-i * beta * sum X_i).

    Args:
        beta: Rotation parameter for mixer layer.
        num_wires: Number of quantum wires.
    """
    for i in range(num_wires):
        qml.RX(2.0 * beta, wires=i)


# =============================================================================
# QAOA Solver Class
# =============================================================================

class QAOAPortfolioSolver:
    """
    Production-grade PennyLane QAOA Solver for portfolio optimization.

    Executes parameterized quantum approximate optimization over discretized
    portfolio selection QUBO models, evaluates exact expectation values,
    samples optimal basis states, and decodes the result into asset weights.
    """

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        reps: int = DEFAULT_REPS,
        optimizer_name: str = DEFAULT_OPTIMIZER,
        optimizer_iter: int = DEFAULT_OPTIMIZER_ITER,
        shots: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the QAOA portfolio solver.

        Args:
            device_name: PennyLane device identifier (default: 'default.qubit').
            reps: Number of alternating QAOA layers (p).
            optimizer_name: Classical optimization algorithm ('COBYLA', 'SLSQP', 'ADAM', etc.).
            optimizer_iter: Maximum optimizer iterations.
            shots: Measurement shots (None for exact analytic statevector simulation).
            seed: Random seed for simulation determinism.
        """
        if not isinstance(reps, int) or reps <= 0:
            raise ValueError(f"reps must be a positive integer, got {reps}.")

        if not isinstance(optimizer_iter, int) or optimizer_iter <= 0:
            raise ValueError(f"optimizer_iter must be a positive integer, got {optimizer_iter}.")

        self.device_name = validate_device_name(device_name)
        self.reps = reps
        self.optimizer_name = optimizer_name
        self.optimizer_iter = optimizer_iter
        self.shots = shots
        self.seed = seed

        logger.info(
            "QAOAPortfolioSolver initialized (device='%s', reps=%d, optimizer='%s', maxiter=%d)",
            self.device_name,
            self.reps,
            self.optimizer_name,
            self.optimizer_iter,
        )

    def _validate_qubo(self, qubo: np.ndarray) -> np.ndarray:
        """Validate input QUBO matrix."""
        return validate_qubo_matrix(qubo)

    def _validate_encoding(
        self,
        n_assets: int,
        binary_bits: int,
        total_qubits: int,
    ) -> None:
        """Validate encoding dimensions."""
        validate_encoding_dimensions(n_assets, binary_bits, total_qubits)

    def _decode_bitstring(
        self,
        bitstring: str,
        n_assets: int,
        binary_bits: int,
    ) -> np.ndarray:
        """Decode bitstring to portfolio weights."""
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
        Execute QAOA to solve the portfolio optimization problem.

        Args:
            qubo: Symmetric QUBO interaction matrix of shape (M, M).
            n_assets: Number of portfolio assets.
            binary_bits: Number of bits per asset.
            offset: Scalar constant energy offset.
            initial_params: Optional initial parameter array of shape (2 * reps,).

        Returns:
            Tuple of:
                - weights: np.ndarray of continuous optimal portfolio weights.
                - metadata: Dict[str, Any] detailing optimization trace and metrics.
        """
        start_time = time.perf_counter()
        Q = self._validate_qubo(qubo)
        num_qubits = Q.shape[0]
        self._validate_encoding(n_assets, binary_bits, num_qubits)

        logger.info(
            "Starting PennyLane QAOA optimization (assets=%d, qubits=%d, reps=%d, backend='%s')",
            n_assets,
            num_qubits,
            self.reps,
            self.device_name,
        )

        try:
            # 1. Algebraic coefficients and PennyLane Hamiltonian
            c0, h, J = qubo_to_ising_coefficients(Q, offset)
            cost_h = qubo_to_pennylane_hamiltonian(Q, offset=offset, include_offset=True)

            # 2. Provision PennyLane Device
            dev = create_device(
                device_name=self.device_name,
                wires=num_qubits,
                shots=self.shots,
                seed=self.seed,
            )

            # 3. Build Cost Expectation QNode
            @qml.qnode(dev)
            def qaoa_cost_circuit(params: np.ndarray) -> float:
                # Prepare uniform superposition |+>^{\otimes M}
                for w in range(num_qubits):
                    qml.Hadamard(wires=w)

                gammas = params[: self.reps]
                betas = params[self.reps :]

                for layer in range(self.reps):
                    apply_cost_layer(gammas[layer], h, J, num_qubits)
                    apply_mixer_layer(betas[layer], num_qubits)

                return qml.expval(cost_h)

            # 4. Build Probability Distribution QNode
            @qml.qnode(dev)
            def qaoa_prob_circuit(params: np.ndarray) -> np.ndarray:
                for w in range(num_qubits):
                    qml.Hadamard(wires=w)

                gammas = params[: self.reps]
                betas = params[self.reps :]

                for layer in range(self.reps):
                    apply_cost_layer(gammas[layer], h, J, num_qubits)
                    apply_mixer_layer(betas[layer], num_qubits)

                return qml.probs(wires=range(num_qubits))

            # 5. Initialize Parameters
            if initial_params is not None:
                params0 = np.asarray(initial_params, dtype=float)
                if len(params0) != 2 * self.reps:
                    raise ValueError(f"initial_params length must be {2 * self.reps}, got {len(params0)}.")
            else:
                # Standard QAOA heuristics: small positive gammas and betas
                rng = np.random.default_rng(self.seed if self.seed is not None else 42)
                gammas0 = rng.uniform(0.01, np.pi / 2, size=self.reps)
                betas0 = rng.uniform(0.01, np.pi / 4, size=self.reps)
                params0 = np.concatenate([gammas0, betas0])

            # 6. Execute Optimization Loop
            optimizer = get_optimizer(
                optimizer_name=self.optimizer_name,
                maxiter=self.optimizer_iter,
            )

            opt_res: OptimizationResult = optimizer.minimize(
                cost_fn=qaoa_cost_circuit,
                initial_params=params0,
            )

            optimal_params = opt_res.optimal_params
            final_energy = float(opt_res.optimal_value)

            # 7. Extract State Probabilities & Identify Best Bitstring
            probs = np.asarray(qaoa_prob_circuit(optimal_params), dtype=float)
            best_state_idx = int(np.argmax(probs))
            best_bitstring = format(best_state_idx, f"0{num_qubits}b")

            # Extract top candidate bitstrings for UI
            top_indices = np.argsort(probs)[::-1][: min(16, len(probs))]
            bitstrings_dict: Dict[str, float] = {
                format(idx, f"0{num_qubits}b"): float(probs[idx]) for idx in top_indices
            }

            # 8. Decode Bitstring to Weights
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
                "num_qubits": num_qubits,
                "num_assets": n_assets,
                "binary_bits": binary_bits,
                "execution_time_sec": elapsed_time,
                "solver": "QAOA",
                "backend": f"PennyLane:{self.device_name}",
                "success": True if (opt_res.success and np.isfinite(final_energy)) else False,
            }

            logger.info(
                "PennyLane QAOA finished successfully: Energy=%.6f, Bitstring=%s, Runtime=%.4fs",
                final_energy,
                best_bitstring,
                elapsed_time,
            )
            return weights, metadata

        except Exception as exc:
            logger.exception("QAOA optimization execution failed: %s", exc)
            raise RuntimeError(f"PennyLane QAOA optimization failed: {exc}") from exc
