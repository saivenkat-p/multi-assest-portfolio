"""
Module: 04_quantum_algorithms.backend

Project:
Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
Centralized provisioning of quantum execution primitives used by
QAOA and VQE solvers. This module abstracts backend-specific
initialization, validates configuration, and provides a stable,
production-ready interface for obtaining Qiskit primitives.

Workflow Position:
03_optimization
│
▼
hamiltonian_builder.py
│
▼
backend.py
├────────► optimizer.py
├────────► ansatz.py
├────────► qaoa_solver.py
└────────► vqe_solver.py
│
▼
05_classical_baseline
06_validation
07_evaluation

Design Philosophy:
- Centralized backend provisioning.
- Fail-fast validation.
- Registry-based backend selection.
- Backward compatible public API.
- Compatible with future simulator and hardware backends.
- Minimal coupling with quantum algorithms.

Author:
ARQPO Development Team
"""

from __future__ import annotations

import logging
import time 
from typing import Any, Dict, Optional, Set, Tuple

from qiskit.primitives import (
    StatevectorEstimator,
    StatevectorSampler,
)

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_BACKEND: str = "local_statevector"

LOCAL_BACKENDS: Set[str] = {
    "local",
    "local_statevector",
    "statevector",
}

HARDWARE_BACKENDS: Set[str] = {
    "hardware",
    "ibmq",
    "qpu",
}

SUPPORTED_BACKENDS: Set[str] = (
    LOCAL_BACKENDS | HARDWARE_BACKENDS
)

DEFAULT_OPTIONS: Dict[str, Any] = {
    "shots": None,
}

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


def _validate_backend_name(backend_type: str) -> str:
    """
    Validate and normalize the requested backend name.

    Args:
        backend_type:
            Backend identifier supplied by the caller.

    Returns:
        Normalized backend identifier.

    Raises:
        TypeError:
            If backend_type is not a string.

        ValueError:
            If backend_type is empty or unsupported.
    """
    if not isinstance(backend_type, str):
        raise TypeError(
            "backend_type must be a string."
        )

    normalized = backend_type.strip().lower()

    if not normalized:
        raise ValueError(
            "backend_type cannot be empty."
        )

    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend_type}'. "
            f"Supported backends: "
            f"{sorted(SUPPORTED_BACKENDS)}"
        )

    return normalized


def _validate_options(
    options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate backend configuration options.

    Args:
        options:
            Optional backend configuration dictionary.

    Returns:
        Validated configuration dictionary.

    Raises:
        TypeError:
            If options is not a dictionary.

        ValueError:
            If configuration values are invalid.
    """
    if options is None:
        return DEFAULT_OPTIONS.copy()

    if not isinstance(options, dict):
        raise TypeError(
            "options must be a dictionary."
        )

    validated = DEFAULT_OPTIONS.copy()
    validated.update(options)

    shots = validated.get("shots")

    if shots is not None:
        if not isinstance(shots, int):
            raise TypeError(
                "'shots' must be an integer."
            )

        if shots <= 0:
            raise ValueError(
                "'shots' must be greater than zero."
            )

    return validated

# =============================================================================
# Primitive Construction Helpers
# =============================================================================


def _create_statevector_sampler(
    options: Dict[str, Any],
) -> StatevectorSampler:
    """
    Create a StatevectorSampler instance.

    Args:
        options:
            Validated backend configuration.

    Returns:
        Configured StatevectorSampler instance.
    """
    logger.debug("Creating StatevectorSampler.")

    return StatevectorSampler(
        default_shots=options.get("shots")
    )


def _create_statevector_estimator(
    options: Dict[str, Any],
) -> StatevectorEstimator:
    """
    Create a StatevectorEstimator instance.

    Args:
        options:
            Validated backend configuration.

    Returns:
        Configured StatevectorEstimator instance.
    """
    logger.debug("Creating StatevectorEstimator.")

    return StatevectorEstimator(
        default_precision=0.0 
    )


# =============================================================================
# Backend Builders
# =============================================================================


def _build_local_backend(
    options: Dict[str, Any],
) -> Tuple[StatevectorSampler, StatevectorEstimator]:
    """
    Construct primitives for exact local statevector simulation.

    Args:
        options:
            Validated backend configuration.

    Returns:
        Tuple containing the sampler and estimator primitives.
    """
    logger.info(
        "Initializing local Statevector backend."
    )

    sampler = _create_statevector_sampler(options)
    estimator = _create_statevector_estimator(options)

    logger.info(
        "Local Statevector backend initialized successfully."
    )

    return sampler, estimator


def _build_hardware_backend(
    options: Dict[str, Any],
) -> Tuple[Any, Any]:
    """
    Placeholder for IBM Runtime / QPU execution.

    Args:
        options:
            Validated backend configuration.

    Raises:
        NotImplementedError:
            Hardware execution is not yet implemented.
    """
    logger.error(
        "Hardware backend requested but is not available."
    )

    raise NotImplementedError(
        "Hardware execution requires "
        "'qiskit-ibm-runtime' and active credentials. "
        "Support will be added in a future ARQPO release."
    )


# =============================================================================
# Backend Registry
# =============================================================================

_BACKEND_BUILDERS = {
    "local": _build_local_backend,
    "local_statevector": _build_local_backend,
    "statevector": _build_local_backend,
    "hardware": _build_hardware_backend,
    "ibmq": _build_hardware_backend,
    "qpu": _build_hardware_backend,
}

# =============================================================================
# Public API
# =============================================================================

def get_primitives(
    backend_type: str = DEFAULT_BACKEND,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Provision Qiskit primitives for the requested execution backend.

    This function serves as the public entry point for obtaining
    Sampler and Estimator primitives used by the ARQPO quantum
    optimization algorithms.

    Args:
        backend_type:
            Backend identifier.

        options:
            Optional backend configuration.

    Returns:
        Tuple containing:

        - Sampler primitive
        - Estimator primitive

    Raises:
        TypeError:
            If the supplied arguments have invalid types.

        ValueError:
            If the backend configuration is invalid.

        RuntimeError:
            If backend initialization fails.

        NotImplementedError:
            If the requested backend is not implemented.
    """
    

    start_time = time.perf_counter()

    try:
        logger.info(
            "Initializing backend '%s'.",
            backend_type,
        )

        backend = _validate_backend_name(
            backend_type
        )

        validated_options = _validate_options(
            options
        )

        builder = _BACKEND_BUILDERS[backend]

        sampler, estimator = builder(
            validated_options
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Backend '%s' initialized successfully "
            "(%.6f seconds).",
            backend,
            elapsed,
        )

        return sampler, estimator
    
    except NotImplementedError:
        raise

    except Exception as exc:
        logger.exception(
            "Backend initialization failed."
        )
        raise RuntimeError(
            f"Failed to initialize backend "
            f"'{backend_type}'."
        ) from exc


# =============================================================================
# Runtime Tests
# =============================================================================


def _runtime_test_local_backend() -> None:
    """Verify successful local backend initialization."""
    sampler, estimator = get_primitives()

    assert sampler is not None
    assert estimator is not None

    print("✓ Runtime Test 1 Passed")


def _runtime_test_backend_aliases() -> None:
    """Verify backend aliases resolve correctly."""
    aliases = [
        "local",
        "statevector",
        "local_statevector",
    ]

    for alias in aliases:
        sampler, estimator = get_primitives(alias)
        assert sampler is not None
        assert estimator is not None

    print("✓ Runtime Test 2 Passed")


def _runtime_test_invalid_backend() -> None:
    """Verify invalid backend rejection."""
    try:
        get_primitives("invalid_backend")
    except RuntimeError:
        print("✓ Runtime Test 3 Passed")
        return

    raise AssertionError(
        "Invalid backend accepted."
    )


def _runtime_test_invalid_options() -> None:
    """Verify option validation."""
    try:
        get_primitives(
            options="invalid"  # type: ignore
        )
    except RuntimeError:
        print("✓ Runtime Test 4 Passed")
        return

    raise AssertionError(
        "Invalid options accepted."
    )


def _runtime_test_invalid_shots() -> None:
    """Verify invalid shot values."""
    try:
        get_primitives(
            options={
                "shots": -5
            }
        )
    except RuntimeError:
        print("✓ Runtime Test 5 Passed")
        return

    raise AssertionError(
        "Negative shots accepted."
    )


def _runtime_test_hardware_backend() -> None:
    """Verify hardware backend placeholder."""
    try:
        get_primitives("hardware")
    except RuntimeError:
        print("✓ Runtime Test 6 Passed")
        return

    raise AssertionError(
        "Hardware backend unexpectedly initialized."
    )


def _runtime_test_reproducibility() -> None:
    """Verify repeated initialization succeeds."""
    sampler1, estimator1 = get_primitives()
    sampler2, estimator2 = get_primitives()

    assert type(sampler1) is type(sampler2)
    assert type(estimator1) is type(estimator2)

    print("✓ Runtime Test 7 Passed")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ARQPO Backend Runtime Tests")
    print("=" * 60)

    _runtime_test_local_backend()
    _runtime_test_backend_aliases()
    _runtime_test_invalid_backend()
    _runtime_test_invalid_options()
    _runtime_test_invalid_shots()
    _runtime_test_hardware_backend()
    _runtime_test_reproducibility()

    print("=" * 60)
    print("All backend runtime tests passed successfully.")
    print("=" * 60)