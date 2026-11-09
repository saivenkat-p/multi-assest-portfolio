"""
Module: quantum_algorithms.ansatz

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Provides production-quality parameterized quantum circuit (ansatz)
    construction utilities for variational quantum algorithms (VQE) used within
    the ARQPO framework using Xanadu PennyLane.

Workflow Position:
    Historical Data -> Data Preprocessing -> Market Regime Detection
    -> Adaptive Optimization -> Hamiltonian Construction
    -> Ansatz Construction  <-- Current Module
    -> VQE / QAOA Solver -> Portfolio Evaluation -> Explainability

Design Philosophy:
    • Production-quality PennyLane implementation
    • Hardware-Efficient Ansatz (HEA) with Ry and Rz rotation layers
    • Linear and circular CNOT entanglement topologies
    • Differentiable parameter management
    • Full backward compatibility with ARQPO solver pipelines
    • Zero Qiskit runtime dependencies
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Final, List, Optional, Set, Tuple, Union

import numpy as np
import pennylane as qml

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_REPS: Final[int] = 2
DEFAULT_ENTANGLEMENT: Final[str] = "linear"

SUPPORTED_ENTANGLEMENTS: Final[frozenset[str]] = frozenset(
    {
        "linear",
        "circular",
        "full",
    }
)

MIN_QUBITS: Final[int] = 1
MAX_QUBITS: Final[int] = 30
FLOAT_TOLERANCE: Final[float] = 1e-12

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
# Validation Helpers
# =============================================================================

def validate_ansatz_inputs(
    num_qubits: int,
    reps: int,
    entanglement: str,
) -> Tuple[int, int, str]:
    """
    Validate public API inputs before ansatz construction.

    Args:
        num_qubits: Number of quantum wires.
        reps: Number of ansatz layer repetitions.
        entanglement: Entanglement topology ('linear', 'circular', 'full').

    Returns:
        Tuple of validated (num_qubits, reps, normalized_entanglement).
    """
    if not isinstance(num_qubits, int):
        raise TypeError(f"num_qubits must be an integer, got {type(num_qubits).__name__}.")

    if num_qubits < MIN_QUBITS:
        raise ValueError(f"num_qubits must be at least {MIN_QUBITS}, got {num_qubits}.")

    if not isinstance(reps, int):
        raise TypeError(f"reps must be an integer, got {type(reps).__name__}.")

    if reps < 0:
        raise ValueError(f"reps cannot be negative, got {reps}.")

    if not isinstance(entanglement, str):
        raise TypeError(f"entanglement must be a string, got {type(entanglement).__name__}.")

    normalized_entanglement = entanglement.strip().lower()
    if normalized_entanglement not in SUPPORTED_ENTANGLEMENTS:
        raise ValueError(
            f"Unsupported entanglement '{entanglement}'. "
            f"Supported values: {sorted(SUPPORTED_ENTANGLEMENTS)}"
        )

    return num_qubits, reps, normalized_entanglement


# =============================================================================
# Parameter Count Calculation
# =============================================================================

def calculate_parameter_count(num_qubits: int, reps: int) -> int:
    """
    Compute total number of variational parameters for the hardware-efficient ansatz.

    Architecture has (reps + 1) rotation layers, each with Ry and Rz on all qubits:
        parameter_count = 2 * num_qubits * (reps + 1)

    Args:
        num_qubits: Number of quantum wires.
        reps: Number of repeating variational layers.

    Returns:
        Total integer parameter count.
    """
    return 2 * num_qubits * (reps + 1)


def generate_initial_parameters(
    num_qubits: int,
    reps: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate initial variational parameter array sampled from uniform distribution [0, 2*pi).

    Args:
        num_qubits: Number of quantum wires.
        reps: Layer count.
        seed: Random seed.

    Returns:
        1D NumPy array of initial angles.
    """
    n_params = calculate_parameter_count(num_qubits, reps)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=n_params)


# =============================================================================
# PennyLane Quantum Circuit Operations
# =============================================================================

def apply_rotation_layer(
    params_slice: Union[np.ndarray, List[float]],
    wires: List[int],
) -> None:
    """
    Apply single-qubit Ry and Rz rotations to all specified wires.

    Args:
        params_slice: 1D array of length 2 * len(wires).
        wires: List of wire indices.
    """
    for idx, wire in enumerate(wires):
        theta_y = params_slice[2 * idx]
        theta_z = params_slice[2 * idx + 1]
        qml.RY(theta_y, wires=wire)
        qml.RZ(theta_z, wires=wire)


def apply_entanglement_layer(
    entanglement: str,
    wires: List[int],
) -> None:
    """
    Apply two-qubit CNOT entangling gates according to specified topology.

    Args:
        entanglement: 'linear', 'circular', or 'full'.
        wires: List of wire indices.
    """
    n = len(wires)
    if n <= 1:
        return

    if entanglement == "linear":
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    elif entanglement == "circular":
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[n - 1], wires[0]])

    elif entanglement == "full":
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[wires[i], wires[j]])


def portfolio_ansatz(
    params: Union[np.ndarray, List[float]],
    wires: Union[int, List[int]],
    reps: int = DEFAULT_REPS,
    entanglement: str = DEFAULT_ENTANGLEMENT,
) -> None:
    """
    PennyLane Quantum Circuit Template for Portfolio Optimization.

    Constructs a hardware-efficient ansatz (HEA) with alternating rotation
    and entanglement layers:
        U(theta) = [Prod_{l=1}^p (Rot(theta_l) Entangle)] Rot(theta_0)

    Args:
        params: 1D array of variational parameters (length 2 * num_qubits * (reps + 1)).
        wires: Integer count of wires or list of wire labels.
        reps: Layer repetition depth (p).
        entanglement: Entanglement topology ('linear', 'circular', 'full').
    """
    wire_list = list(range(wires)) if isinstance(wires, int) else list(wires)
    num_qubits = len(wire_list)
    validate_ansatz_inputs(num_qubits, reps, entanglement)

    expected_params = calculate_parameter_count(num_qubits, reps)
    if len(params) != expected_params:
        raise ValueError(
            f"Parameter length mismatch: expected {expected_params} parameters, "
            f"got {len(params)}."
        )

    stride = 2 * num_qubits
    param_idx = 0

    # Initial rotation layer
    apply_rotation_layer(params[param_idx : param_idx + stride], wire_list)
    param_idx += stride

    # Repetition layers (Entanglement + Rotation)
    for layer in range(reps):
        apply_entanglement_layer(entanglement, wire_list)
        apply_rotation_layer(params[param_idx : param_idx + stride], wire_list)
        param_idx += stride


# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

class PortfolioAnsatz:
    """
    Object-oriented wrapper encapsulating portfolio ansatz configuration.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = DEFAULT_REPS,
        entanglement: str = DEFAULT_ENTANGLEMENT,
    ) -> None:
        self.num_qubits, self.reps, self.entanglement = validate_ansatz_inputs(
            num_qubits, reps, entanglement
        )
        self.parameter_count = calculate_parameter_count(self.num_qubits, self.reps)

    def generate_initial_parameters(self, seed: Optional[int] = 42) -> np.ndarray:
        return generate_initial_parameters(self.num_qubits, self.reps, seed=seed)

    def __call__(self, params: np.ndarray, wires: Optional[List[int]] = None) -> None:
        target_wires = wires if wires is not None else list(range(self.num_qubits))
        portfolio_ansatz(
            params=params,
            wires=target_wires,
            reps=self.reps,
            entanglement=self.entanglement,
        )


def build_portfolio_ansatz(
    num_qubits: int,
    reps: int = DEFAULT_REPS,
    entanglement: str = DEFAULT_ENTANGLEMENT,
) -> PortfolioAnsatz:
    """
    Factory function returning a configured PortfolioAnsatz instance.

    Args:
        num_qubits: Total number of quantum wires.
        reps: Variational repetition layers.
        entanglement: Entanglement topology.

    Returns:
        Configured PortfolioAnsatz instance.
    """
    return PortfolioAnsatz(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
    )
