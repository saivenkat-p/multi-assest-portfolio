"""
Unit tests for the quantum optimizer factory and adapters.
"""

import numpy as np
import pytest

from quantum_algorithms.optimizer import (
    get_optimizer,
    get_supported_optimizers,
    is_supported_optimizer,
    OptimizerConfig,
    QuantumOptimizerFactory,
    OptimizationResult,
)


def test_supported_optimizers():
    supported = get_supported_optimizers()
    assert "COBYLA" in supported
    assert "SLSQP" in supported
    assert "ADAM" in supported
    assert is_supported_optimizer("COBYLA")
    assert is_supported_optimizer("cobyla")
    assert not is_supported_optimizer("UNKNOWN_OPTIMIZER")


def test_optimizer_validation():
    with pytest.raises(ValueError):
        get_optimizer(optimizer_name="INVALID_OPT")

    with pytest.raises(ValueError):
        get_optimizer(maxiter=0)

    with pytest.raises(ValueError):
        get_optimizer(tol=-1.0)


def test_cobyla_minimization():
    # Simple quadratic objective: (x - 2)^2 + (y + 1)^2
    def cost_fn(params):
        return (params[0] - 2.0) ** 2 + (params[1] + 1.0) ** 2

    opt = get_optimizer("COBYLA", maxiter=60)
    res = opt.minimize(cost_fn, np.array([0.0, 0.0]))

    assert isinstance(res, OptimizationResult)
    assert np.isclose(res.optimal_params[0], 2.0, atol=1e-2)
    assert np.isclose(res.optimal_params[1], -1.0, atol=1e-2)
    assert res.optimal_value < 1e-3
    assert len(res.cost_history) > 0


def test_adam_minimization():
    def cost_fn(params):
        return (params[0] - 1.0) ** 2

    opt = get_optimizer("ADAM", maxiter=70, stepsize=0.1)
    res = opt.minimize(cost_fn, np.array([0.0]))

    assert isinstance(res, OptimizationResult)
    assert np.isclose(res.optimal_params[0], 1.0, atol=0.05)


def test_quantum_optimizer_factory():
    config = OptimizerConfig(optimizer_name="SLSQP", maxiter=50)
    factory = QuantumOptimizerFactory(config)
    adapter = factory.build()
    assert adapter is not None
