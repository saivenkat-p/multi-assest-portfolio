"""
Mathematical verification tests for QUBO-to-Ising energy equivalence.

Validates that for all 2^M computational basis states |b>:
    <b | H_Ising | b> == b^T Q b + offset
within floating point tolerance (<= 1e-10).
"""

import itertools
from typing import Tuple
import numpy as np
import pennylane as qml
import pytest

from quantum_algorithms.hamiltonian import (
    qubo_to_pennylane_hamiltonian,
    qubo_to_ising_coefficients,
    validate_qubo_matrix,
)


def evaluate_pennylane_energy_on_basis_state(
    hamiltonian: qml.Hamiltonian,
    bitstring: Tuple[int, ...],
) -> float:
    """
    Compute <b | H | b> exactly using a PennyLane QNode on default.qubit.
    """
    num_wires = len(bitstring)
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev)
    def circuit():
        for wire, bit in enumerate(bitstring):
            if bit == 1:
                qml.PauliX(wires=wire)
        return qml.expval(hamiltonian)

    return float(circuit())


@pytest.mark.parametrize("dim,offset", [(2, 0.0), (3, 50.0), (4, -12.5), (5, 100.0)])
def test_all_basis_states_energy_equivalence(dim: int, offset: float):
    """
    Exhaustively enumerate all 2^M basis states and verify QUBO energy matches
    PennyLane Hamiltonian expectation value.
    """
    np.random.seed(dim * 42)
    # Generate random symmetric QUBO matrix
    raw = np.random.uniform(-5.0, 5.0, size=(dim, dim))
    Q = (raw + raw.T) / 2.0

    # Build PennyLane Hamiltonian with exact constant offset
    H = qubo_to_pennylane_hamiltonian(Q, offset=offset, include_offset=True)

    # Enumerate all 2^dim binary configurations
    for bit_tuple in itertools.product([0, 1], repeat=dim):
        b = np.array(bit_tuple, dtype=float)

        # Classical QUBO energy: b^T Q b + offset
        expected_qubo_energy = float(b.T @ Q @ b + offset)

        # Quantum Ising expectation: <b | H | b>
        quantum_energy = evaluate_pennylane_energy_on_basis_state(H, bit_tuple)

        assert np.isclose(
            expected_qubo_energy,
            quantum_energy,
            atol=1e-10,
        ), f"Mismatch for state {bit_tuple}: QUBO={expected_qubo_energy}, Quantum={quantum_energy}"


def test_ground_state_matches_optimal_qubo_solution():
    """
    Verify that exact Hamiltonian ground state matches brute-force QUBO minimum.
    """
    np.random.seed(123)
    dim = 4
    raw = np.random.uniform(-2.0, 2.0, size=(dim, dim))
    Q = (raw + raw.T) / 2.0
    offset = 25.0

    H = qubo_to_pennylane_hamiltonian(Q, offset=offset)

    # Brute force classical enumeration
    best_qubo_energy = float("inf")
    best_bitstring = None

    for bit_tuple in itertools.product([0, 1], repeat=dim):
        b = np.array(bit_tuple, dtype=float)
        energy = float(b.T @ Q @ b + offset)
        if energy < best_qubo_energy:
            best_qubo_energy = energy
            best_bitstring = bit_tuple

    # Matrix representation of PennyLane Hamiltonian
    H_matrix = qml.matrix(H)
    eigenvalues, _ = np.linalg.eigh(H_matrix)
    min_eigenvalue = float(eigenvalues[0])

    assert np.isclose(
        best_qubo_energy,
        min_eigenvalue,
        atol=1e-10,
    ), f"Ground state eigenvalue {min_eigenvalue} does not match QUBO min {best_qubo_energy}"
