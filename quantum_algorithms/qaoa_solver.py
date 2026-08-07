"""
Module: 04_quantum_algorithms.qaoa_solver

Executes the Quantum Approximate Optimization Algorithm (QAOA)
for adaptive portfolio optimization.

This module converts a Quadratic Unconstrained Binary Optimization (QUBO)
problem into an Ising Hamiltonian and solves it using QAOA. The resulting
optimal bitstring is decoded into continuous portfolio weights for downstream
financial analysis.

Workflow
--------
Optimization Module
        │
        ▼
Hamiltonian (QUBO)
        │
        ▼
QUBO → Ising Mapping
        │
        ▼
QAOA Optimization
        │
        ▼
Optimal Bitstring
        │
        ▼
Portfolio Weights

Mathematical Formulation
------------------------
Binary variables are mapped using

    x_i = (I - Z_i) / 2

which transforms the QUBO objective into an equivalent Ising Hamiltonian
compatible with quantum optimization.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Module constants
DEFAULT_REPS = 2
DEFAULT_OPTIMIZER_ITER = 100
FLOAT_TOLERANCE = 1e-10


class QAOAPortfolioSolver:
    
    def __init__(
        self,
        sampler: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        reps: int = DEFAULT_REPS,
        optimizer_iter: int = DEFAULT_OPTIMIZER_ITER,
    ):
        """
        Initialize the QAOA Solver for portfolio optimization.
        """

        if not isinstance(reps, int):
            raise TypeError("reps must be an integer.")

        if not isinstance(optimizer_iter, int):
            raise TypeError("optimizer_iter must be an integer.")

        if reps <= 0:
            raise ValueError("reps must be greater than zero.")

        if optimizer_iter <= 0:
            raise ValueError("optimizer_iter must be greater than zero.")

        self.sampler = (
            sampler if sampler is not None else Sampler()
        )

        self.optimizer = (
            optimizer
            if optimizer is not None
            else COBYLA(maxiter=optimizer_iter)
        )

        self.reps = reps

    def _validate_qubo(self, qubo: np.ndarray) -> None:
        """
        Validate the supplied QUBO matrix.

        Args:
            qubo:
                Candidate QUBO matrix.

        Raises:
            TypeError:
                If the input is not a NumPy array.

            ValueError:
                If the matrix is invalid.
        """
        if not isinstance(qubo, np.ndarray):
            raise TypeError("QUBO must be a numpy array.")
        if qubo.size == 0:
            raise ValueError("QUBO cannot be empty.")

        if qubo.ndim != 2:
            raise ValueError("QUBO must be a 2D matrix.")

        rows, cols = qubo.shape

        if rows != cols:
            raise ValueError("QUBO must be a square matrix.")

        # Check for NaN or Inf BEFORE symmetry check
        if not np.all(np.isfinite(qubo)):
            raise ValueError("QUBO contains NaN or infinite values.")

        if not np.allclose(
            qubo,
            qubo.T,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError("QUBO matrix must be symmetric.")
                 
    def _validate_encoding(
        self,
        n_assets: int,
        binary_bits: int,
        total_qubits: int,
    ) -> None:
        """
        Ensure the bitstring encoding matches the asset allocation parameters.
        """
        if not isinstance(n_assets, int) or n_assets <= 0:
            raise ValueError("n_assets must be a positive integer.")
            
        if not isinstance(binary_bits, int) or binary_bits <= 0:
            raise ValueError("binary_bits must be a positive integer.")
            
        if n_assets * binary_bits != total_qubits:
            raise ValueError(
                f"Encoding mismatch: {n_assets} assets * {binary_bits} bits != {total_qubits} qubits."
            )

    def _qubo_to_ising(self, qubo: np.ndarray) -> SparsePauliOp:
        """
        Convert a QUBO matrix into an Ising Hamiltonian.
        """
        self._validate_qubo(qubo)

        logger.info(
            "Converting %d x %d QUBO matrix to Ising Hamiltonian.",
            qubo.shape[0],
            qubo.shape[1],
        )

        try:
            qubo = np.where(
                np.abs(qubo) < FLOAT_TOLERANCE,
                0.0,
                qubo,
            )
            num_qubits = qubo.shape[0]
            pauli_list = []

            for i in range(num_qubits):
                for j in range(i, num_qubits):
                    weight = float(qubo[i, j])

                    if np.isclose(weight, 0.0, atol=FLOAT_TOLERANCE):
                        continue

                    if i == j:
                        z_string = ["I"] * num_qubits
                        z_string[i] = "Z"
                        pauli_list.append(("".join(z_string)[::-1], -weight / 2.0))
                    else:
                        weight *= 2.0

                        # Z_i
                        z_i = ["I"] * num_qubits
                        z_i[i] = "Z"
                        pauli_list.append(("".join(z_i)[::-1], -weight / 4.0))

                        # Z_j
                        z_j = ["I"] * num_qubits
                        z_j[j] = "Z"
                        pauli_list.append(("".join(z_j)[::-1], -weight / 4.0))

                        # Z_i Z_j
                        z_ij = ["I"] * num_qubits
                        z_ij[i] = "Z"
                        z_ij[j] = "Z"
                        pauli_list.append(("".join(z_ij)[::-1], weight / 4.0))

            if not pauli_list:
                raise RuntimeError("Hamiltonian construction produced no Pauli terms.")

            hamiltonian = SparsePauliOp.from_list(pauli_list).simplify()
            coefficients = np.asarray(hamiltonian.coeffs, dtype=np.complex128)

            if not np.all(np.isfinite(coefficients)):
                raise RuntimeError("Hamiltonian contains non-finite coefficients.")

            logger.info("Hamiltonian created successfully with %d Pauli terms.", len(hamiltonian))
            return hamiltonian

        except Exception as exc:
            logger.exception("Failed to construct Ising Hamiltonian.")
            raise RuntimeError("Unable to convert QUBO into an Ising Hamiltonian.") from exc

    def _decode_bitstring(
        self,
        bitstring: str,
        n_assets: int,
        binary_bits: int,
    ) -> np.ndarray:
        """
        Decode a measured quantum bitstring into portfolio weights.
        """
        if not isinstance(bitstring, str):
            raise TypeError("bitstring must be a string.")

        if not set(bitstring).issubset({"0", "1"}):
            raise ValueError("bitstring must contain only '0' and '1'.")

        self._validate_encoding(
            n_assets=n_assets,
            binary_bits=binary_bits,
            total_qubits=len(bitstring),
        )

        logger.info("Decoding measured bitstring into portfolio weights.")

        try:
            bitstring = bitstring[::-1]
            weights = np.zeros(n_assets, dtype=np.float64)
            max_value = (2 ** binary_bits) - 1

            if max_value <= 0:
                raise ValueError("Invalid binary encoding.")

            for asset_idx in range(n_assets):
                start = asset_idx * binary_bits
                end = start + binary_bits
                chunk = bitstring[start:end]

                if len(chunk) != binary_bits:
                    raise ValueError("Bitstring length does not match encoding configuration.")

                integer_value = int(chunk, 2)
                weights[asset_idx] = integer_value / max_value

            weights[np.abs(weights) < FLOAT_TOLERANCE] = 0.0

            if not np.all(np.isfinite(weights)):
                raise RuntimeError("Decoded portfolio contains non-finite values.")

            logger.info("Successfully decoded %d portfolio weights.", len(weights))
            return weights

        except Exception as exc:
            logger.exception("Failed to decode QAOA measurement bitstring.")
            raise RuntimeError("Unable to decode quantum measurement.") from exc

    def solve(
        self,
        qubo: np.ndarray,
        n_assets: int,
        binary_bits: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute the Quantum Approximate Optimization Algorithm (QAOA)
        to solve the portfolio optimization problem.
        """
        self._validate_qubo(qubo)
        self._validate_encoding(
            n_assets=n_assets,
            binary_bits=binary_bits,
            total_qubits=qubo.shape[0],
        )

        logger.info(
            "Starting QAOA optimization (assets=%d, qubits=%d, reps=%d).",
            n_assets,
            qubo.shape[0],
            self.reps,
        )

        try:
            hamiltonian = self._qubo_to_ising(qubo)

            qaoa = QAOA(
                sampler=self.sampler,
                optimizer=self.optimizer,
                reps=self.reps,
            )

            logger.info("Running QAOA optimizer...")
            result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)

            if result is None:
                raise RuntimeError("QAOA returned no optimization result.")

            measurement = getattr(result, "best_measurement", None)
            if measurement is None:
                raise RuntimeError("Best measurement was not returned.")

            best_bitstring = measurement.get("bitstring")
            if best_bitstring is None:
                raise RuntimeError("Best measurement does not contain a bitstring.")

            eigenvalue = getattr(result, "eigenvalue", None)
            if eigenvalue is None:
                raise RuntimeError("Eigenvalue was not returned.")

            final_energy = float(eigenvalue.real)
            if not np.isfinite(final_energy):
                raise RuntimeError("Computed energy is not finite.")

            weights = self._decode_bitstring(best_bitstring, n_assets, binary_bits)
            if not np.all(np.isfinite(weights)):
                raise RuntimeError("Decoded portfolio contains invalid values.")

            weights[np.abs(weights) < FLOAT_TOLERANCE] = 0.0

            metadata = {
                "optimal_parameters": getattr(result, "optimal_point", None),
                "final_energy": final_energy,
                "best_bitstring": best_bitstring,
                "optimizer_evals": getattr(result, "optimizer_evals", None),
                "reps": self.reps,
                "num_qubits": qubo.shape[0],
                "num_assets": n_assets,
            }

            logger.info("QAOA optimization completed successfully.")
            logger.info("Final energy: %.8f", final_energy)
            logger.info("Best bitstring: %s", best_bitstring)

            return weights, metadata

        except Exception as exc:
            logger.exception("QAOA optimization failed.")
            raise RuntimeError("Failed to solve portfolio using QAOA.") from exc