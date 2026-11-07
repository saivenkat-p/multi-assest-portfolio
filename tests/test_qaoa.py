"""
Baseline test suite for QAOA portfolio solver interface and input validation.
"""

import numpy as np
import pytest

from quantum_algorithms.qaoa_solver import QAOAPortfolioSolver


def test_qaoa_solver_initialization():
    solver = QAOAPortfolioSolver(reps=2, optimizer_iter=50)
    assert solver.reps == 2

    with pytest.raises(TypeError):
        QAOAPortfolioSolver(reps="invalid")

    with pytest.raises(ValueError):
        QAOAPortfolioSolver(reps=0)

    with pytest.raises(ValueError):
        QAOAPortfolioSolver(optimizer_iter=-5)


def test_qaoa_validate_qubo():
    solver = QAOAPortfolioSolver()

    # Non-numpy
    with pytest.raises(TypeError):
        solver._validate_qubo([[1, 0], [0, 1]])

    # Non-square
    with pytest.raises(ValueError):
        solver._validate_qubo(np.zeros((2, 3)))

    # Empty
    with pytest.raises(ValueError):
        solver._validate_qubo(np.array([]))

    # NaN / Inf
    with pytest.raises(ValueError):
        solver._validate_qubo(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    # Asymmetric
    with pytest.raises(ValueError):
        solver._validate_qubo(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_qaoa_validate_encoding():
    solver = QAOAPortfolioSolver()

    # Valid: 2 assets * 3 bits = 6 qubits
    solver._validate_encoding(n_assets=2, binary_bits=3, total_qubits=6)

    # Mismatch
    with pytest.raises(ValueError):
        solver._validate_encoding(n_assets=2, binary_bits=3, total_qubits=5)

    with pytest.raises(ValueError):
        solver._validate_encoding(n_assets=0, binary_bits=3, total_qubits=6)
