"""
Module: 04_quantum_algorithms.ansatz

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Provides production-quality parameterized quantum circuit (ansatz)
    construction utilities for variational quantum algorithms used within
    the ARQPO framework.

Workflow Position:
    Historical Data
        ↓
    Data Preprocessing
        ↓
    Market Regime Detection
        ↓
    Adaptive Optimization
        ↓
    Hamiltonian Construction
        ↓
    Ansatz Construction  ← Current Module
        ↓
    VQE / QAOA Solver
        ↓
    Portfolio Evaluation
        ↓
    Explainability

Mathematical Objective:
    Construct hardware-efficient parameterized quantum circuits that map a
    classical parameter vector θ into a quantum state

        |ψ(θ)⟩ = U(θ)|0...0⟩

    suitable for variational optimization.

Design Philosophy:
    • Production-quality implementation
    • Defensive programming
    • Deterministic parameter ordering
    • Modular helper functions
    • Reusable across VQE and future algorithms
    • Backward compatible with existing solver APIs
"""

from __future__ import annotations

import logging
import time
from typing import Final

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_REPS: Final[int] = 2

DEFAULT_ENTANGLEMENT: Final[str] = "linear"

SUPPORTED_ENTANGLEMENTS: Final[frozenset[str]] = frozenset(
    {
        "linear",
        "circular",
    }
)

PARAMETER_PREFIX: Final[str] = "θ"

MIN_QUBITS: Final[int] = 1

FLOAT_TOLERANCE: Final[float] = 1e-12

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# =============================================================================
# Validation Helpers
# =============================================================================

def _validate_inputs(
    num_qubits: int,
    reps: int,
    entanglement: str,
    insert_barriers: bool,
) -> None:
    """
    Validate public API inputs before circuit construction.
    """
    if not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer.")

    if num_qubits < MIN_QUBITS:
        raise ValueError(f"num_qubits must be at least {MIN_QUBITS}.")

    if not isinstance(reps, int):
        raise TypeError("reps must be an integer.")

    if reps < 0:
        raise ValueError("reps cannot be negative.")

    if not isinstance(entanglement, str):
        raise TypeError("entanglement must be a string.")

    entanglement = entanglement.lower()

    if entanglement not in SUPPORTED_ENTANGLEMENTS:
        raise ValueError(
            f"Unsupported entanglement strategy '{entanglement}'. "
            f"Supported values are: {sorted(SUPPORTED_ENTANGLEMENTS)}."
        )

    if not isinstance(insert_barriers, bool):
        raise TypeError("insert_barriers must be a boolean.")


# =============================================================================
# Numerical Helpers
# =============================================================================

def _calculate_parameter_count(
    num_qubits: int,
    reps: int,
) -> int:
    """
    Compute the total number of variational parameters.
    """
    return (reps + 1) * num_qubits * 2


# =============================================================================
# Circuit Construction Helpers
# =============================================================================

def _add_rotation_layer(
    circuit: QuantumCircuit,
    parameters: ParameterVector,
    parameter_index: int,
) -> int:
    """
    Append one variational rotation layer.
    """
    for qubit in range(circuit.num_qubits):
        circuit.ry(
            parameters[parameter_index],
            qubit,
        )
        parameter_index += 1

        circuit.rz(
            parameters[parameter_index],
            qubit,
        )
        parameter_index += 1

    return parameter_index


def _add_entanglement_layer(
    circuit: QuantumCircuit,
    entanglement: str,
) -> None:
    """
    Add an entanglement layer.
    """
    if entanglement == "linear":
        for qubit in range(circuit.num_qubits - 1):
            circuit.cx(
                qubit,
                qubit + 1,
            )
    elif entanglement == "circular":
        for qubit in range(circuit.num_qubits):
            circuit.cx(
                qubit,
                (qubit + 1) % circuit.num_qubits,
            )


# =============================================================================
# Main API
# =============================================================================

def build_portfolio_ansatz(
    num_qubits: int,
    reps: int = DEFAULT_REPS,
    entanglement: str = DEFAULT_ENTANGLEMENT,
    insert_barriers: bool = True,
) -> QuantumCircuit:
    """
    Construct a production-quality hardware-efficient portfolio ansatz.
    """
    start_time = time.perf_counter()

    try:
        logger.info(
            "Building portfolio ansatz (qubits=%d, reps=%d, entanglement=%s)",
            num_qubits,
            reps,
            entanglement,
        )

        # -------------------------------------------------------------
        # Validate inputs
        # -------------------------------------------------------------
        _validate_inputs(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=insert_barriers,
        )

        # -------------------------------------------------------------
        # Initialize circuit
        # -------------------------------------------------------------
        circuit = QuantumCircuit(
            num_qubits,
            name="PortfolioAnsatz",
        )

        parameter_count = _calculate_parameter_count(
            num_qubits,
            reps,
        )

        parameters = ParameterVector(
            PARAMETER_PREFIX,
            parameter_count,
        )

        parameter_index = 0

        # -------------------------------------------------------------
        # Initial rotation layer
        # -------------------------------------------------------------
        parameter_index = _add_rotation_layer(
            circuit=circuit,
            parameters=parameters,
            parameter_index=parameter_index,
        )

        if insert_barriers:
            circuit.barrier()

        # -------------------------------------------------------------
        # Repeated ansatz blocks
        # -------------------------------------------------------------
        for layer in range(reps):
            logger.info(
                "Constructing ansatz layer %d/%d",
                layer + 1,
                reps,
            )

            _add_entanglement_layer(
                circuit=circuit,
                entanglement=entanglement,
            )

            if insert_barriers:
                circuit.barrier()

            parameter_index = _add_rotation_layer(
                circuit=circuit,
                parameters=parameters,
                parameter_index=parameter_index,
            )

            if insert_barriers:
                circuit.barrier()

        # -------------------------------------------------------------
        # Internal consistency checks
        # -------------------------------------------------------------
        if parameter_index != parameter_count:
            raise RuntimeError(
                "Parameter indexing mismatch detected. "
                f"Expected {parameter_count}, "
                f"generated {parameter_index}."
            )

        elapsed_time = time.perf_counter() - start_time

        logger.info("Portfolio ansatz successfully constructed.")
        logger.info(
            "Summary | Qubits=%d | Repetitions=%d | Parameters=%d | "
            "Depth=%d | Size=%d | Runtime=%.6f sec",
            circuit.num_qubits,
            reps,
            parameter_count,
            circuit.depth(),
            circuit.size(),
            elapsed_time,
        )

        return circuit

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Portfolio ansatz construction failed.")
        raise RuntimeError("Failed to build portfolio ansatz.") from exc