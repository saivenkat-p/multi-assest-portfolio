"""
Module: quantum_algorithms.backend

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Centralized provisioning and management of PennyLane quantum execution devices.
    Abstracts device initialization, shot configuration, seed determinism,
    and simulator backend selection for QAOA and VQE solvers.

Workflow Position:
    optimization/hamiltonian_builder.py
            │
            ▼
    quantum_algorithms/backend.py  <-- (Current Module)
            ├──► optimizer.py
            ├──► ansatz.py
            ├──► qaoa_solver.py
            └──► vqe_solver.py

Design Philosophy:
    - Pure PennyLane device provisioning (default.qubit, lightning.qubit).
    - Separation of exact statevector and shot-based simulation.
    - Explicit reproducibility via seed control.
    - Defensive validation and graceful fallback.
    - Zero Qiskit runtime dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import pennylane as qml

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_DEVICE: str = "default.qubit"
LIGHTNING_DEVICE: str = "lightning.qubit"

SUPPORTED_DEVICES: Set[str] = {
    "default.qubit",
    "lightning.qubit",
    "local_statevector",  # Legacy compatibility alias -> default.qubit
    "statevector",        # Legacy compatibility alias -> default.qubit
}

MIN_WIRES: int = 1
MAX_WIRES: int = 30  # Practical limit for local full statevector simulation

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
# Configuration Data Model
# =============================================================================

@dataclass(frozen=True)
class DeviceConfig:
    """
    Immutable configuration for a PennyLane quantum execution device.

    Attributes:
        device_name: Name of the PennyLane device plugin (default: 'default.qubit').
        wires: Number of quantum wires (qubits) or sequence of wire labels.
        shots: Number of measurement shots (None for exact analytic statevector).
        seed: Random seed for deterministic simulation.
        extra_options: Additional device-specific keyword arguments.
    """
    device_name: str = DEFAULT_DEVICE
    wires: int = 1
    shots: Optional[int] = None
    seed: Optional[int] = 42
    extra_options: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_device_name(name: str) -> str:
    """
    Validate and normalize the requested quantum device name.

    Args:
        name: Device identifier string.

    Returns:
        Normalized canonical device name.

    Raises:
        TypeError: If name is not a string.
        ValueError: If device is not supported.
    """
    if not isinstance(name, str):
        raise TypeError(f"Device name must be a string, got {type(name).__name__}.")

    normalized = name.strip().lower()

    # Map legacy aliases
    if normalized in {"local_statevector", "statevector", "default", "aer_simulator"}:
        return DEFAULT_DEVICE

    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported quantum device '{name}'. "
            f"Supported devices: {sorted(SUPPORTED_DEVICES)}"
        )

    return normalized


def validate_wires(wires: Union[int, List[Any]]) -> int:
    """
    Validate wire count for quantum simulation.

    Args:
        wires: Integer count of wires or list of wire identifiers.

    Returns:
        Total number of wires.

    Raises:
        TypeError: If wires is neither integer nor list.
        ValueError: If wire count is outside allowable range.
    """
    if isinstance(wires, int):
        num_wires = wires
    elif isinstance(wires, (list, tuple)):
        num_wires = len(wires)
    else:
        raise TypeError(f"wires must be int or sequence, got {type(wires).__name__}.")

    if num_wires < MIN_WIRES:
        raise ValueError(f"Wire count must be at least {MIN_WIRES}, got {num_wires}.")

    if num_wires > MAX_WIRES:
        raise ValueError(
            f"Requested {num_wires} wires exceeds local simulation safety limit of {MAX_WIRES}."
        )

    return num_wires


def validate_shots(shots: Optional[int]) -> Optional[int]:
    """
    Validate shot count for quantum measurement.

    Args:
        shots: Integer number of shots, or None for analytic statevector.

    Returns:
        Validated shots value.

    Raises:
        TypeError: If shots is not int or None.
        ValueError: If shots is <= 0.
    """
    if shots is None:
        return None

    if not isinstance(shots, int):
        raise TypeError(f"shots must be an integer or None, got {type(shots).__name__}.")

    if shots <= 0:
        raise ValueError(f"shots must be greater than zero, got {shots}.")

    return shots


# =============================================================================
# Factory Functions
# =============================================================================

def create_device(
    device_name: str = DEFAULT_DEVICE,
    wires: int = 1,
    shots: Optional[int] = None,
    seed: Optional[int] = 42,
    **kwargs: Any,
) -> qml.devices.Device:
    """
    Instantiate and return a configured PennyLane quantum device.

    Args:
        device_name: Device identifier ('default.qubit', 'lightning.qubit').
        wires: Number of quantum wires.
        shots: Measurement shot count (None for analytic expectation values).
        seed: Random seed for reproducible simulation.
        **kwargs: Additional device parameters.

    Returns:
        Configured PennyLane Device instance.

    Raises:
        RuntimeError: If device construction fails and fallback fails.
    """
    normalized_name = validate_device_name(device_name)
    validated_wires = validate_wires(wires)
    validated_shots = validate_shots(shots)

    logger.info(
        "Initializing PennyLane device '%s' (wires=%d, shots=%s, seed=%s)",
        normalized_name,
        validated_wires,
        validated_shots,
        seed,
    )

    device_kwargs: Dict[str, Any] = dict(kwargs)
    if seed is not None:
        device_kwargs["seed"] = seed
    if validated_shots is not None:
        device_kwargs["shots"] = validated_shots

    try:
        device = qml.device(
            normalized_name,
            wires=validated_wires,
            **device_kwargs,
        )
        logger.info("PennyLane device '%s' initialized successfully.", normalized_name)
        return device

    except Exception as exc:
        if normalized_name != DEFAULT_DEVICE:
            logger.warning(
                "Failed to initialize '%s': %s. Falling back to '%s'.",
                normalized_name,
                exc,
                DEFAULT_DEVICE,
            )
            try:
                device = qml.device(
                    DEFAULT_DEVICE,
                    wires=validated_wires,
                    **device_kwargs,
                )
                logger.info("Fallback to '%s' successful.", DEFAULT_DEVICE)
                return device
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Fallback device initialization failed: {fallback_exc}"
                ) from fallback_exc

        raise RuntimeError(
            f"Failed to initialize PennyLane device '{normalized_name}': {exc}"
        ) from exc


def is_analytic(device: Any) -> bool:
    """
    Check if a device operates in exact analytic (statevector) mode.

    Args:
        device: PennyLane Device instance.

    Returns:
        True if exact analytic simulation without shot noise, False otherwise.
    """
    shots = getattr(device, "shots", None)
    if shots is None:
        return True
    if hasattr(shots, "total_shots"):
        return shots.total_shots is None
    return shots == 0 or shots is None
