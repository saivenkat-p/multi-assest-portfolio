"""
End-to-End integration tests for ARQPO PennyLane quantum optimization pipeline.

Validates complete workflow:
    Expected Returns & Covariance
        -> HamiltonianBuilder (QUBO)
        -> PennyLane QAOA / VQE Solvers
        -> Bitstring Decoding
        -> Portfolio Metrics Evaluation
"""

import numpy as np
import pytest

from optimization.hamiltonian_builder import HamiltonianBuilder
from optimization.portfolio_metrics import evaluate_portfolio
from quantum_algorithms.qaoa_solver import QAOAPortfolioSolver
from quantum_algorithms.vqe_solver import VQEPortfolioSolver


@pytest.fixture
def sample_market_data():
    """Generates synthetic 3-asset market return and covariance data."""
    np.random.seed(42)
    n_assets = 3
    expected_returns = np.array([0.12, 0.18, 0.09])
    # Positive semi-definite covariance
    A = np.random.uniform(0.05, 0.15, size=(n_assets, n_assets))
    cov_matrix = (A @ A.T) + np.eye(n_assets) * 0.01
    return expected_returns, cov_matrix


def test_end_to_end_qaoa_pipeline(sample_market_data):
    expected_returns, cov_matrix = sample_market_data
    n_assets = len(expected_returns)
    binary_bits = 1

    # 1. Build QUBO
    builder = HamiltonianBuilder(risk_aversion=1.0, penalty_factor=50.0, binary_bits=binary_bits)
    qubo, offset = builder.build_qubo(expected_returns, cov_matrix)

    # 2. Run PennyLane QAOA
    qaoa_solver = QAOAPortfolioSolver(
        device_name="default.qubit",
        reps=1,
        optimizer_name="COBYLA",
        optimizer_iter=30,
        seed=42,
    )
    weights, metadata = qaoa_solver.solve(
        qubo=qubo,
        n_assets=n_assets,
        binary_bits=binary_bits,
        offset=offset,
    )

    # 3. Validate Solution Integrity
    assert len(weights) == n_assets
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
    assert (weights >= 0.0).all()
    assert metadata["solver"] == "QAOA"
    assert metadata["success"] == True

    # 4. Evaluate Financial Portfolio Metrics
    metrics = evaluate_portfolio(weights, expected_returns, cov_matrix, risk_free_rate=0.02)
    assert metrics["Expected_Return"] > 0.0
    assert metrics["Volatility"] > 0.0
    assert np.isfinite(metrics["Sharpe_Ratio"])


def test_end_to_end_vqe_pipeline(sample_market_data):
    expected_returns, cov_matrix = sample_market_data
    n_assets = len(expected_returns)
    binary_bits = 1

    # 1. Build QUBO
    builder = HamiltonianBuilder(risk_aversion=1.5, penalty_factor=50.0, binary_bits=binary_bits)
    qubo, offset = builder.build_qubo(expected_returns, cov_matrix)

    # 2. Run PennyLane VQE
    vqe_solver = VQEPortfolioSolver(
        device_name="default.qubit",
        reps=1,
        entanglement="linear",
        optimizer_name="COBYLA",
        optimizer_iter=30,
        seed=42,
    )
    weights, metadata = vqe_solver.solve(
        qubo=qubo,
        n_assets=n_assets,
        binary_bits=binary_bits,
        offset=offset,
    )

    # 3. Validate Solution Integrity
    assert len(weights) == n_assets
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
    assert (weights >= 0.0).all()
    assert metadata["solver"] == "VQE"
    assert metadata["success"] == True

    # 4. Evaluate Financial Portfolio Metrics
    metrics = evaluate_portfolio(weights, expected_returns, cov_matrix, risk_free_rate=0.02)
    assert metrics["Expected_Return"] > 0.0
    assert metrics["Volatility"] > 0.0
    assert np.isfinite(metrics["Sharpe_Ratio"])
