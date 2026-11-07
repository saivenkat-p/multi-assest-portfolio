"""
Baseline tests for HamiltonianBuilder and QUBO construction.
"""

import numpy as np
import pytest

from optimization.hamiltonian_builder import HamiltonianBuilder


def test_hamiltonian_builder_initialization():
    builder = HamiltonianBuilder(risk_aversion=1.5, penalty_factor=500.0, binary_bits=2)
    assert builder.risk_aversion == 1.5
    assert builder.penalty_factor == 500.0
    assert builder.binary_bits == 2
    assert builder.max_val == 3


def test_hamiltonian_builder_qubo_shape_and_symmetry():
    builder = HamiltonianBuilder(risk_aversion=1.0, penalty_factor=100.0, binary_bits=2)
    n_assets = 3
    expected_returns = np.array([0.10, 0.15, 0.08])
    cov_matrix = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.05, 0.02],
        [0.00, 0.02, 0.03],
    ])

    Q, offset = builder.build_qubo(expected_returns, cov_matrix)

    # Total qubits = n_assets * binary_bits = 3 * 2 = 6
    assert Q.shape == (6, 6)
    # Must be symmetric
    assert np.allclose(Q, Q.T, atol=1e-10)
    assert offset == 100.0
    assert np.all(np.isfinite(Q))


def test_hamiltonian_builder_validation():
    builder = HamiltonianBuilder()

    # Mismatched dimensions
    with pytest.raises(Exception):
        builder.build_qubo(np.array([0.1, 0.2]), np.eye(3))
