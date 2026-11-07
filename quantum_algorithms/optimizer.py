"""
Module: quantum_algorithms.optimizer

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Centralized factory and abstraction for classical and gradient-based
    optimizers used across PennyLane QAOA and VQE hybrid optimization pipelines.

Workflow Position:
    optimization/hamiltonian_builder.py
            │
            ▼
       backend.py
            │
            ▼
    quantum_algorithms/optimizer.py  <-- (Current Module)
            ├──► qaoa_solver.py
            └──► vqe_solver.py

Responsibilities:
    • Validate optimizer configurations.
    • Support SciPy gradient-free and gradient-based optimizers (COBYLA, SLSQP, Nelder-Mead).
    • Support PennyLane native optimizers (Adam, GradientDescent, SPSA).
    • Provide a unified .minimize() interface returning structured OptimizationResult.
    • Eliminate all legacy Qiskit algorithm dependencies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pennylane as qml
from scipy.optimize import minimize as scipy_minimize

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_OPTIMIZER = "COBYLA"
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 1e-6
DEFAULT_STEPSIZE = 0.05
FLOAT_TOLERANCE = 1e-12

SUPPORTED_OPTIMIZERS = (
    "COBYLA",
    "SLSQP",
    "SPSA",
    "ADAM",
    "GRADIENT_DESCENT",
    "NELDER-MEAD",
)

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
# Optimizer Enumeration
# =============================================================================

class OptimizerType(str, Enum):
    """Supported classical and quantum optimizers."""
    COBYLA = "COBYLA"
    SLSQP = "SLSQP"
    SPSA = "SPSA"
    ADAM = "ADAM"
    GRADIENT_DESCENT = "GRADIENT_DESCENT"
    NELDER_MEAD = "NELDER-MEAD"


# =============================================================================
# Result and Configuration Data Models
# =============================================================================

@dataclass
class OptimizationResult:
    """
    Standardized result structure returned by all optimizer executions.

    Attributes:
        optimal_params: Final optimized parameter array.
        optimal_value: Final objective cost value.
        iterations: Number of optimization steps completed.
        cost_history: Sequence of objective values recorded during optimization.
        success: Whether optimization terminated successfully.
        message: Status or termination message.
    """
    optimal_params: np.ndarray
    optimal_value: float
    iterations: int
    cost_history: List[float] = field(default_factory=list)
    success: bool = True
    message: str = "Optimization completed successfully."


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Configuration describing an optimizer.

    Parameters:
        optimizer_name: Name of the optimizer algorithm.
        maxiter: Maximum optimization iterations or function evaluations.
        tol: Convergence tolerance (for SciPy methods).
        stepsize: Learning rate (for gradient-based methods like Adam).
    """
    optimizer_name: str = DEFAULT_OPTIMIZER
    maxiter: int = DEFAULT_MAX_ITERATIONS
    tol: Optional[float] = DEFAULT_TOLERANCE
    stepsize: float = DEFAULT_STEPSIZE


# =============================================================================
# Validation Helpers
# =============================================================================

def _validate_optimizer_name(name: str) -> str:
    """Validate and normalize optimizer name."""
    if not isinstance(name, str):
        raise TypeError(f"Optimizer name must be a string, got {type(name).__name__}.")

    normalized = name.strip().upper()
    if normalized not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{name}'. Supported optimizers: {SUPPORTED_OPTIMIZERS}"
        )
    return normalized


def _validate_max_iterations(maxiter: int) -> int:
    """Validate iteration limit."""
    if not isinstance(maxiter, int):
        raise TypeError(f"maxiter must be an integer, got {type(maxiter).__name__}.")
    if maxiter <= 0:
        raise ValueError(f"maxiter must be greater than zero, got {maxiter}.")
    return maxiter


def _validate_tolerance(tol: Optional[float]) -> Optional[float]:
    """Validate convergence tolerance."""
    if tol is None:
        return None
    if not isinstance(tol, (int, float)):
        raise TypeError(f"tol must be a float or int, got {type(tol).__name__}.")
    if tol <= 0:
        raise ValueError(f"tol must be greater than zero, got {tol}.")
    return float(tol)


# =============================================================================
# Optimizer Execution Adapters
# =============================================================================

class BaseOptimizerAdapter:
    """Base interface for optimizer adapters."""
    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        raise NotImplementedError


class ScipyOptimizerAdapter(BaseOptimizerAdapter):
    """Adapter for SciPy classical optimization algorithms."""
    def __init__(self, method: str, maxiter: int, tol: Optional[float] = None) -> None:
        self.method = method
        self.maxiter = maxiter
        self.tol = tol

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        cost_history: List[float] = []

        def tracked_cost_fn(params: np.ndarray) -> float:
            val = float(cost_fn(params))
            cost_history.append(val)
            return val

        options: Dict[str, Any] = {"maxiter": self.maxiter}
        if self.method == "COBYLA":
            options["tol"] = self.tol or 1e-4

        res = scipy_minimize(
            tracked_cost_fn,
            np.asarray(initial_params, dtype=float),
            method=self.method,
            tol=self.tol,
            options=options,
        )

        final_val = float(res.fun) if np.isfinite(res.fun) else (cost_history[-1] if cost_history else 0.0)
        return OptimizationResult(
            optimal_params=np.asarray(res.x, dtype=float),
            optimal_value=final_val,
            iterations=len(cost_history),
            cost_history=cost_history,
            success=bool(res.success),
            message=str(res.message),
        )


class PennyLaneGradientOptimizerAdapter(BaseOptimizerAdapter):
    """Adapter for PennyLane native optimizers (Adam, GradientDescent, SPSA)."""
    def __init__(self, name: str, maxiter: int, stepsize: float) -> None:
        self.name = name
        self.maxiter = maxiter
        self.stepsize = stepsize

        if name == "ADAM":
            self.opt = qml.AdamOptimizer(stepsize=stepsize)
        elif name == "GRADIENT_DESCENT":
            self.opt = qml.GradientDescentOptimizer(stepsize=stepsize)
        elif name == "SPSA":
            self.opt = qml.SPSAOptimizer(maxiter=maxiter)
        else:
            raise ValueError(f"Unknown PennyLane optimizer: {name}")

    def minimize(
        self,
        cost_fn: Callable[[Any], float],
        initial_params: np.ndarray,
    ) -> OptimizationResult:
        # PennyLane requires qml.numpy with requires_grad=True for autograd differentiation
        params = qml.numpy.array(initial_params, requires_grad=True)
        cost_history: List[float] = []

        for step in range(self.maxiter):
            params, cost = self.opt.step_and_cost(cost_fn, params)
            cost_history.append(float(cost))

        final_cost = float(cost_fn(params))
        cost_history.append(final_cost)

        return OptimizationResult(
            optimal_params=np.asarray(params, dtype=float),
            optimal_value=final_cost,
            iterations=self.maxiter,
            cost_history=cost_history,
            success=True,
            message="PennyLane optimization finished max iterations.",
        )


# =============================================================================
# Factory API
# =============================================================================

def get_optimizer(
    optimizer_name: str = DEFAULT_OPTIMIZER,
    maxiter: int = DEFAULT_MAX_ITERATIONS,
    tol: Optional[float] = DEFAULT_TOLERANCE,
    stepsize: float = DEFAULT_STEPSIZE,
) -> BaseOptimizerAdapter:
    """
    Construct and return a unified optimizer adapter.

    Args:
        optimizer_name: Optimizer algorithm identifier ('COBYLA', 'SLSQP', 'ADAM', etc.).
        maxiter: Maximum optimization iterations.
        tol: Convergence tolerance for SciPy optimizers.
        stepsize: Step size for gradient-based optimizers.

    Returns:
        Configured BaseOptimizerAdapter with .minimize() method.
    """
    normalized_name = _validate_optimizer_name(optimizer_name)
    validated_maxiter = _validate_max_iterations(maxiter)
    validated_tol = _validate_tolerance(tol)

    if normalized_name in {"COBYLA", "SLSQP", "NELDER-MEAD"}:
        method = "Nelder-Mead" if normalized_name == "NELDER-MEAD" else normalized_name
        return ScipyOptimizerAdapter(
            method=method,
            maxiter=validated_maxiter,
            tol=validated_tol,
        )
    else:
        return PennyLaneGradientOptimizerAdapter(
            name=normalized_name,
            maxiter=validated_maxiter,
            stepsize=stepsize,
        )


def get_supported_optimizers() -> Tuple[str, ...]:
    """Return tuple of supported optimizer names."""
    return SUPPORTED_OPTIMIZERS


def is_supported_optimizer(name: str) -> bool:
    """Check if an optimizer name is supported."""
    if not isinstance(name, str):
        return False
    return name.strip().upper() in SUPPORTED_OPTIMIZERS


class QuantumOptimizerFactory:
    """Backward-compatible factory class for optimizer construction."""
    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        self.config = config or OptimizerConfig()

    def build(self) -> BaseOptimizerAdapter:
        return get_optimizer(
            optimizer_name=self.config.optimizer_name,
            maxiter=self.config.maxiter,
            tol=self.config.tol,
            stepsize=self.config.stepsize,
        )
