"""
Module: quantum_algorithms.hamiltonian

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Converts Quadratic Unconstrained Binary Optimization (QUBO) matrices
    into native PennyLane Ising spin Hamiltonians.

Mathematical Formulation:
    Binary decision variables x_i in {0, 1} are mapped to spin operators Z_i in {+1, -1}:
        x_i = (I - Z_i) / 2

    Given the QUBO objective:
        E(x) = x^T Q x + offset = sum_i Q_ii x_i + sum_{i < j} 2 Q_ij x_i x_j + offset

    The equivalent Ising spin Hamiltonian is:
        H_C = c_0 I + sum_{i=0}^{M-1} h_i Z_i + sum_{0 <= i < j < M} J_ij Z_i Z_j

    where:
        J_ij = Q_ij / 2
        h_i  = -0.5 * sum_{j=0}^{M-1} Q_ij
        c_0  = offset + 0.25 * sum_{i,j} Q_ij + 0.25 * sum_i Q_ii

    This formulation guarantees exact computational basis energy equivalence:
        <b | H_C | b> == b^T Q b + offset,  for all b in {0, 1}^M.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE = 1e-12

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
# Validation
# =============================================================================

def validate_qubo_matrix(qubo: np.ndarray) -> np.ndarray:
    """
    Validate the QUBO matrix for dimensional and numerical consistency.

    Args:
        qubo: Candidate QUBO matrix.

    Returns:
        Validated symmetric NumPy array.

    Raises:
        TypeError: If qubo is not a NumPy array.
        ValueError: If qubo is empty, non-square, non-finite, or asymmetric.
    """
    if not isinstance(qubo, np.ndarray):
        raise TypeError(f"QUBO must be a numpy.ndarray, got {type(qubo).__name__}.")

    if qubo.size == 0:
        raise ValueError("QUBO matrix cannot be empty.")

    if qubo.ndim != 2:
        raise ValueError(f"QUBO must be a 2D matrix, got shape {qubo.shape}.")

    rows, cols = qubo.shape
    if rows != cols:
        raise ValueError(f"QUBO must be square, got {rows}x{cols}.")

    if not np.all(np.isfinite(qubo)):
        raise ValueError("QUBO matrix contains NaN or infinite values.")

    if not np.allclose(qubo, qubo.T, atol=1e-8):
        raise ValueError("QUBO matrix must be symmetric.")

    return qubo


# =============================================================================
# Hamiltonian Construction
# =============================================================================

def qubo_to_ising_coefficients(
    qubo: np.ndarray,
    offset: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute algebraic Ising Hamiltonian parameters (c0, h, J) from a QUBO matrix.

    Args:
        qubo: Symmetric QUBO matrix of shape (M, M).
        offset: Scalar offset from QUBO formulation.

    Returns:
        Tuple of (c0, h, J):
            - c0: float scalar constant energy offset
            - h: np.ndarray of shape (M,) linear Z coefficients
            - J: np.ndarray of shape (M, M) quadratic ZZ interaction couplings
    """
    Q = validate_qubo_matrix(qubo)
    M = Q.shape[0]

    # Clean near-zero numerical noise
    Q = np.where(np.abs(Q) < FLOAT_TOLERANCE, 0.0, Q)

    # Quadratic couplings: J_ij = Q_ij / 2 for i < j
    J = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            J[i, j] = Q[i, j] / 2.0

    # Linear terms: h_i = -0.5 * sum_{j=0}^{M-1} Q_ij
    h = -0.5 * np.sum(Q, axis=1)

    # Constant offset: c_0 = offset + 0.25 * sum(Q) + 0.25 * trace(Q)
    c0 = float(offset + 0.25 * np.sum(Q) + 0.25 * np.trace(Q))

    return c0, h, J


def qubo_to_pennylane_hamiltonian(
    qubo: np.ndarray,
    offset: float = 0.0,
    include_offset: bool = True,
) -> qml.Hamiltonian:
    """
    Construct a PennyLane Hamiltonian instance representing the QUBO problem.

    Args:
        qubo: Symmetric QUBO matrix of shape (M, M).
        offset: Constant offset from QUBO construction.
        include_offset: Whether to include the identity constant offset operator.

    Returns:
        Configured qml.Hamiltonian.
    """
    Q = validate_qubo_matrix(qubo)
    M = Q.shape[0]
    c0, h, J = qubo_to_ising_coefficients(Q, offset)

    coeffs: List[float] = []
    ops: List[qml.operation.Operator] = []

    # Constant identity term
    if include_offset and abs(c0) > FLOAT_TOLERANCE:
        coeffs.append(c0)
        ops.append(qml.Identity(0))

    # Linear PauliZ terms
    for i in range(M):
        if abs(h[i]) > FLOAT_TOLERANCE:
            coeffs.append(float(h[i]))
            ops.append(qml.PauliZ(i))

    # Quadratic PauliZ @ PauliZ terms
    for i in range(M):
        for j in range(i + 1, M):
            if abs(J[i, j]) > FLOAT_TOLERANCE:
                coeffs.append(float(J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if not ops:
        logger.warning("All Ising coefficients were near zero. Adding zero Identity operator.")
        coeffs.append(0.0)
        ops.append(qml.Identity(0))

    hamiltonian = qml.Hamiltonian(coeffs, ops)
    logger.info(
        "Constructed PennyLane Ising Hamiltonian with %d terms (qubits=%d, c0=%.6f)",
        len(coeffs),
        M,
        c0,
    )
    return hamiltonian


def create_qaoa_mixer_hamiltonian(num_qubits: int) -> qml.Hamiltonian:
    """
    Construct standard transverse-field mixer Hamiltonian for QAOA:
        H_M = sum_{i=0}^{M-1} PauliX(i)

    Args:
        num_qubits: Number of quantum wires.

    Returns:
        PennyLane Hamiltonian representing the mixer.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}.")

    coeffs = [1.0] * num_qubits
    ops = [qml.PauliX(i) for i in range(num_qubits)]
    return qml.Hamiltonian(coeffs, ops)
