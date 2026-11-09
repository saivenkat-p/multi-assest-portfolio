"""
Comprehensive test suite for PennyLane QAOAPortfolioSolver.
"""

import numpy as np
import pytest

from quantum_algorithms.qaoa_solver import QAOAPortfolioSolver


def test_qaoa_solver_initialization():
    solver = QAOAPortfolioSolver(reps=2, optimizer_iter=50, device_name="default.qubit")
    assert solver.reps == 2
    assert solver.optimizer_iter == 50
    assert solver.device_name == "default.qubit"

    with pytest.raises(ValueError):
        QAOAPortfolioSolver(reps=0)

    with pytest.raises(ValueError):
        QAOAPortfolioSolver(optimizer_iter=-10)


def test_qaoa_validation_errors():
    solver = QAOAPortfolioSolver()

    # Non-square
    with pytest.raises(ValueError):
        solver._validate_qubo(np.zeros((2, 3)))

    # NaN
    with pytest.raises(ValueError):
        solver._validate_qubo(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    # Dimension mismatch (2 assets * 2 bits = 4 != 3)
    with pytest.raises(ValueError):
        solver._validate_encoding(n_assets=2, binary_bits=2, total_qubits=3)


def test_qaoa_solve_small_system():
    solver = QAOAPortfolioSolver(reps=1, optimizer_name="COBYLA", optimizer_iter=30, seed=42)

    # 2 assets, 1 bit each -> 2 qubits
    Q = np.array([
        [-2.0, 0.5],
        [0.5, -1.0],
    ])
    offset = 5.0

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

    # Validate metadata
    assert "optimal_parameters" in metadata
    assert "final_energy" in metadata
    assert "best_bitstring" in metadata
    assert "top_bitstrings" in metadata
    assert "convergence_trace" in metadata
    assert metadata["solver"] == "QAOA"
    assert metadata["num_qubits"] == 2
    assert metadata["num_assets"] == 2
    assert metadata["success"] == True
    assert len(metadata["convergence_trace"]) > 0


def test_qaoa_reproducibility():
    solver1 = QAOAPortfolioSolver(reps=1, optimizer_iter=20, seed=123)
    solver2 = QAOAPortfolioSolver(reps=1, optimizer_iter=20, seed=123)

    Q = np.array([[1.0, -0.5], [-0.5, 2.0]])
    w1, m1 = solver1.solve(Q, n_assets=2, binary_bits=1)
    w2, m2 = solver2.solve(Q, n_assets=2, binary_bits=1)

    assert np.allclose(w1, w2)
    assert m1["best_bitstring"] == m2["best_bitstring"]
    assert np.isclose(m1["final_energy"], m2["final_energy"])
