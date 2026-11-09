"""
Comprehensive test suite for PennyLane VQEPortfolioSolver.
"""

import numpy as np
import pennylane as qml
import pytest

from quantum_algorithms.hamiltonian import qubo_to_pennylane_hamiltonian
from quantum_algorithms.vqe_solver import VQEPortfolioSolver


def test_vqe_solver_initialization():
    solver = VQEPortfolioSolver(
        reps=2,
        entanglement="circular",
        optimizer_iter=40,
        device_name="default.qubit",
    )
    assert solver.reps == 2
    assert solver.entanglement == "circular"
    assert solver.optimizer_iter == 40
    assert solver.device_name == "default.qubit"

    with pytest.raises(ValueError):
        VQEPortfolioSolver(reps=-1)

    with pytest.raises(ValueError):
        VQEPortfolioSolver(optimizer_iter=0)


def test_vqe_solve_and_exact_comparison():
    solver = VQEPortfolioSolver(
        reps=2,
        entanglement="linear",
        optimizer_name="COBYLA",
        optimizer_iter=50,
        seed=42,
    )

    # 2 assets, 1 bit each -> 2 qubits
    Q = np.array([
        [-1.5, 0.4],
        [0.4, -0.8],
    ])
    offset = 10.0

    weights, metadata = solver.solve(
        qubo=Q,
        n_assets=2,
        binary_bits=1,
        offset=offset,
    )

    assert isinstance(weights, np.ndarray)
    assert len(weights) == 2
    assert np.isclose(np.sum(weights), 1.0)
    assert (weights >= 0.0).all()

    # Exact diagonalization check
    H = qubo_to_pennylane_hamiltonian(Q, offset=offset)
    H_mat = qml.matrix(H)
    exact_ground_energy = float(np.min(np.linalg.eigvalsh(H_mat)))

    # VQE variational energy must satisfy variational principle: E_vqe >= E_exact
    assert metadata["final_energy"] >= exact_ground_energy - 1e-6
    # And should be reasonably close to ground state
    assert metadata["final_energy"] < exact_ground_energy + 1.0

    # Metadata checks
    assert metadata["solver"] == "VQE"
    assert metadata["num_qubits"] == 2
    assert metadata["num_assets"] == 2
    assert metadata["success"] == True
    assert len(metadata["convergence_trace"]) > 0


def test_vqe_reproducibility():
    solver1 = VQEPortfolioSolver(reps=1, optimizer_iter=25, seed=99)
    solver2 = VQEPortfolioSolver(reps=1, optimizer_iter=25, seed=99)

    Q = np.array([[2.0, -1.0], [-1.0, 1.5]])
    w1, m1 = solver1.solve(Q, n_assets=2, binary_bits=1)
    w2, m2 = solver2.solve(Q, n_assets=2, binary_bits=1)

    assert np.allclose(w1, w2)
    assert m1["best_bitstring"] == m2["best_bitstring"]
    assert np.isclose(m1["final_energy"], m2["final_energy"])
