"""
===============================================================================
Module: 04_quantum_algorithms.optimizer
Project: Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Centralized factory for constructing classical optimizers used by hybrid
    quantum optimization algorithms within ARQPO.

    This module provides a production-quality abstraction over Qiskit's
    classical optimizers, ensuring consistent configuration, validation,
    logging, deterministic behavior, and backward compatibility across
    QAOA and VQE workflows.

Workflow Position:
    Hamiltonian Builder
            │
            ▼
       Ansatz Builder
            │
            ▼
    Quantum Optimizer Factory   <-- (Current Module)
            │
            ├──────────────► QAOA Solver
            │
            └──────────────► VQE Solver

Responsibilities:
    • Validate optimizer configuration.
    • Construct supported Qiskit optimizers.
    • Centralize optimizer defaults.
    • Provide compatibility-aware optimizer creation.
    • Improve maintainability by avoiding duplicated optimizer logic.
    • Preserve deterministic configuration across quantum solvers.

Supported Optimizers:
    • COBYLA
    • SPSA
    • SLSQP

Design Philosophy:
    This module acts as the single source of truth for classical optimizer
    creation inside the ARQPO quantum stack. Individual quantum solvers
    should never instantiate Qiskit optimizers directly.

Author:
    ARQPO Development Team

===============================================================================
"""

from __future__ import annotations

import logging
import time 
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from qiskit_algorithms.optimizers import (
    COBYLA,
    SPSA,
    SLSQP,
    Optimizer,
)

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_OPTIMIZER = "COBYLA"

DEFAULT_MAX_ITERATIONS = 100

DEFAULT_TOLERANCE: Optional[float] = None

FLOAT_TOLERANCE = 1e-12

SUPPORTED_OPTIMIZERS = (
    "COBYLA",
    "SPSA",
    "SLSQP",
)

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
# Optimizer Enumeration
# =============================================================================


class OptimizerType(str, Enum):
    """
    Enumeration of supported classical optimizers.

    Using an enumeration prevents invalid string usage throughout
    the quantum optimization pipeline.
    """

    COBYLA = "COBYLA"
    SPSA = "SPSA"
    SLSQP = "SLSQP"


# =============================================================================
# Optimizer Configuration
# =============================================================================


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Immutable configuration describing a classical optimizer.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer.

    maxiter : int
        Maximum optimization iterations.

    tol : Optional[float]
        Numerical convergence tolerance.
    """

    optimizer_name: str = DEFAULT_OPTIMIZER
    maxiter: int = DEFAULT_MAX_ITERATIONS
    tol: Optional[float] = DEFAULT_TOLERANCE


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_optimizer_name(name: str) -> str:
    """
    Validate the optimizer name.

    Args:
        name:
            Optimizer identifier.

    Returns:
        Normalized optimizer name.

    Raises:
        TypeError:
            If the optimizer name is not a string.

        ValueError:
            If the optimizer is unsupported.
    """

    if not isinstance(name, str):
        raise TypeError(
            "optimizer_name must be a string."
        )

    normalized = name.strip().upper()

    if normalized not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{name}'. "
            f"Supported optimizers: {SUPPORTED_OPTIMIZERS}"
        )

    return normalized


def _validate_max_iterations(maxiter: int) -> int:
    """
    Validate the maximum iteration count.

    Args:
        maxiter:
            Requested iteration limit.

    Returns:
        Validated iteration count.

    Raises:
        TypeError:
            If maxiter is not an integer.

        ValueError:
            If maxiter is less than one.
    """

    if not isinstance(maxiter, int):
        raise TypeError(
            "maxiter must be an integer."
        )

    if maxiter < 1:
        raise ValueError(
            "maxiter must be greater than zero."
        )

    return maxiter


def _validate_tolerance(
    tol: Optional[float],
) -> Optional[float]:
    """
    Validate numerical convergence tolerance.

    Args:
        tol:
            Optimization tolerance.

    Returns:
        Validated tolerance.

    Raises:
        TypeError:
            If tolerance has an invalid datatype.

        ValueError:
            If tolerance is negative, NaN, or infinite.
    """

    if tol is None:
        return None

    if not isinstance(tol, (int, float)):
        raise TypeError(
            "tol must be a numeric value or None."
        )

    tol = float(tol)

    if math.isnan(tol):
        raise ValueError("tol cannot be NaN.")

    if math.isinf(tol):
        raise ValueError("tol cannot be infinite.")

    if tol < 0:
        raise ValueError(
            "tol must be non-negative."
        )

    return tol


def _validate_configuration(
    config: OptimizerConfig,
) -> OptimizerConfig:
    """
    Validate an optimizer configuration object.

    Args:
        config:
            Optimizer configuration.

    Returns:
        Validated configuration.

    Raises:
        TypeError:
            If configuration object is invalid.
    """

    if not isinstance(config, OptimizerConfig):
        raise TypeError(
            "config must be an OptimizerConfig instance."
        )

    validated_name = _validate_optimizer_name(
        config.optimizer_name
    )

    validated_iter = _validate_max_iterations(
        config.maxiter
    )

    validated_tol = _validate_tolerance(
        config.tol
    )

    return OptimizerConfig(
        optimizer_name=validated_name,
        maxiter=validated_iter,
        tol=validated_tol,
    )

# =============================================================================
# Quantum Optimizer Factory
# =============================================================================


class QuantumOptimizerFactory:
    """
    Factory responsible for constructing validated Qiskit optimizers.

    This class centralizes optimizer creation to ensure that all quantum
    algorithms within ARQPO use consistent optimizer configuration and
    validation.

    Notes:
        • This class does not perform optimization itself.
        • It only constructs configured optimizer instances.
        • All optimizer-specific implementation details are isolated here.
    """

    def __init__(
        self,
        config: OptimizerConfig,
    ) -> None:
        """
        Initialize the optimizer factory.

        Args:
            config:
                Optimizer configuration.

        Raises:
            TypeError:
                If configuration is invalid.
        """

        self._config = _validate_configuration(config)

        # Builder registry (initialized once)
        self._builders = {
            OptimizerType.COBYLA.value: self._build_cobyla,
            OptimizerType.SPSA.value: self._build_spsa,
            OptimizerType.SLSQP.value: self._build_slsqp,
        }




        logger.info(
            "QuantumOptimizerFactory initialized "
            "(optimizer=%s, maxiter=%d, tol=%s)",
            self._config.optimizer_name,
            self._config.maxiter,
            self._config.tol,
        )

    # -------------------------------------------------------------------------
    # Internal Builder Methods
    # -------------------------------------------------------------------------

    def _build_cobyla(self) -> Optimizer:
        """
        Construct a COBYLA optimizer.

        Returns:
            Configured COBYLA optimizer.
        """

        logger.debug("Creating COBYLA optimizer.")

        return COBYLA(
            maxiter=self._config.maxiter,
            tol=self._config.tol,
        )

    def _build_spsa(self) -> Optimizer:
        """
        Construct an SPSA optimizer.

        Notes:
            SPSA ignores convergence tolerance because its stopping
            behavior differs from gradient-based optimizers.

        Returns:
            Configured SPSA optimizer.
        """

        logger.debug("Creating SPSA optimizer.")

        return SPSA(
            maxiter=self._config.maxiter,
        )

    def _build_slsqp(self) -> Optimizer:
        """
        Construct an SLSQP optimizer.

        Returns:
            Configured SLSQP optimizer.
        """

        logger.debug("Creating SLSQP optimizer.")

        return SLSQP(
            maxiter=self._config.maxiter,
            tol=self._config.tol,
        )

    # -------------------------------------------------------------------------
    # Optimizer Dispatcher
    # -------------------------------------------------------------------------

    def build(self) -> Optimizer:
        """
        Build the requested optimizer.

        Returns:
            Configured Qiskit optimizer.

        Raises:
            RuntimeError:
                If optimizer creation fails unexpectedly.
        """

        try:
            start_time = time.perf_counter()
            optimizer_name = self._config.optimizer_name

            logger.info(
                "Building optimizer '%s'.",
                optimizer_name,
            )

            builder = self._builders.get(optimizer_name)

            if builder is None:
                raise ValueError(
                    f"Unsupported optimizer: {optimizer_name}"
                )

            optimizer = builder()

            logger.info(
                "%s optimizer created successfully.",
                optimizer_name,
            )
            
            elapsed = time.perf_counter() - start_time

            logger.info(
                "Optimizer built in %.6f seconds.",
                elapsed,
            )
            
            return optimizer

        except Exception as exc:

            logger.exception(
                "Optimizer construction failed."
            )

            raise RuntimeError(
                "Failed to construct optimizer."
            ) from exc

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def supported_optimizers() -> tuple[str, ...]:
        """
        Return all supported optimizer names.

        Returns:
            Tuple containing supported optimizer identifiers.
        """

        return  tuple(
            optimizer.value
            for optimizer in OptimizerType
        )   

    @staticmethod
    def is_supported(
        optimizer_name: str,
    ) -> bool:
        """
        Check whether an optimizer is supported.

        Args:
            optimizer_name:
                Name of optimizer.

        Returns:
            True if supported, otherwise False.
        """

        if not isinstance(optimizer_name, str):
            return False

        return (
            optimizer_name.strip().upper()
            in SUPPORTED_OPTIMIZERS
        )

# =============================================================================
# Public API
# =============================================================================


def get_optimizer(
    optimizer_name: str = DEFAULT_OPTIMIZER,
    maxiter: int = DEFAULT_MAX_ITERATIONS,
    tol: Optional[float] = DEFAULT_TOLERANCE,
) -> Optimizer:
    """
    Create and return a configured Qiskit classical optimizer.

    This function serves as the primary public interface for optimizer
    construction within the ARQPO quantum optimization framework.
    Both QAOA and VQE solvers should use this function instead of
    directly instantiating Qiskit optimizer classes.

    Args:
        optimizer_name:
            Name of the optimizer.

        maxiter:
            Maximum optimization iterations.

        tol:
            Numerical convergence tolerance.

    Returns:
        Configured Qiskit Optimizer instance.

    Raises:
        RuntimeError:
            If optimizer construction fails.

    Example:
        >>> optimizer = get_optimizer(
        ...     optimizer_name="COBYLA",
        ...     maxiter=200,
        ...     tol=1e-6,
        ... )
    """

    logger.info(
        "Received optimizer request "
        "(optimizer=%s, maxiter=%d, tol=%s).",
        optimizer_name,
        maxiter,
        tol,
    )

    config = OptimizerConfig(
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        tol=tol,
    )

    factory = QuantumOptimizerFactory(config)

    optimizer = factory.build()

    logger.info(
        "Optimizer '%s' successfully returned.",
        optimizer_name.upper(),
    )

    return optimizer


# =============================================================================
# Compatibility Wrappers
# =============================================================================


def create_optimizer(
    optimizer_name: str = DEFAULT_OPTIMIZER,
    maxiter: int = DEFAULT_MAX_ITERATIONS,
    tol: Optional[float] = DEFAULT_TOLERANCE,
) -> Optimizer:
    """
    Compatibility wrapper around ``get_optimizer()``.

    Notes:
        This wrapper exists to preserve backward compatibility with
        earlier ARQPO modules and external notebooks that may still
        reference ``create_optimizer()``.

    Args:
        optimizer_name:
            Optimizer name.

        maxiter:
            Maximum optimization iterations.

        tol:
            Convergence tolerance.

    Returns:
        Configured optimizer.
    """

    logger.debug(
        "create_optimizer() called."
    )

    return get_optimizer(
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        tol=tol,
    )


# =============================================================================
# Module Utility Functions
# =============================================================================


def get_supported_optimizers() -> tuple[str, ...]:
    """
    Return all supported optimizer names.

    Returns:
        Tuple containing supported optimizer identifiers.
    """

    return QuantumOptimizerFactory.supported_optimizers()


def is_supported_optimizer(
    optimizer_name: str,
) -> bool:
    """
    Check whether an optimizer is supported.

    Args:
        optimizer_name:
            Name of optimizer.

    Returns:
        True if supported, otherwise False.
    """

    return QuantumOptimizerFactory.is_supported(
        optimizer_name
    )


# =============================================================================
# Module Metadata
# =============================================================================

__all__ = [
    "OptimizerType",
    "OptimizerConfig",
    "QuantumOptimizerFactory",
    "get_optimizer",   
    "create_optimizer",
    "get_supported_optimizers",
    "is_supported_optimizer",
]

# =============================================================================
# Runtime Tests
# =============================================================================


def _runtime_test_supported_optimizers() -> None:
    """
    Verify that every supported optimizer can be constructed.
    """

    logger.info("Running supported optimizer tests...")

    for optimizer_name in get_supported_optimizers():

        optimizer = get_optimizer(
            optimizer_name=optimizer_name,
            maxiter=100,
            tol=1e-6,
        )

        assert isinstance(
            optimizer,
            Optimizer,
        )

        logger.info(
            "✓ %s created successfully.",
            optimizer_name,
        )


def _runtime_test_default_optimizer() -> None:
    """
    Verify default optimizer creation.
    """

    logger.info("Running default optimizer test...")

    optimizer = get_optimizer()

    assert isinstance(
        optimizer,
        Optimizer,
    )

    logger.info("✓ Default optimizer test passed.")


def _runtime_test_invalid_optimizer() -> None:
    """
    Verify unsupported optimizer rejection.
    """

    logger.info("Running invalid optimizer test...")

    try:

        get_optimizer(
            optimizer_name="INVALID_OPTIMIZER",
        )

    except ValueError:

        logger.info(
            "✓ Invalid optimizer rejected correctly."
        )

    else:

        raise AssertionError(
            "Invalid optimizer should raise RuntimeError."
        )


def _runtime_test_invalid_iterations() -> None:
    """
    Verify iteration validation.
    """

    logger.info("Running invalid iteration test...")

    try:

        get_optimizer(
            maxiter=0,
        )

    except ValueError:

        logger.info(
            "✓ Invalid iteration count rejected."
        )

    else:

        raise AssertionError(
            "Iteration validation failed."
        )


def _runtime_test_invalid_tolerance() -> None:
    """
    Verify tolerance validation.
    """

    logger.info("Running invalid tolerance test...")

    try:

        get_optimizer(
            tol=-1.0,
        )

    except ValueError:

        logger.info(
            "✓ Invalid tolerance rejected."
        )

    else:

        raise AssertionError(
            "Tolerance validation failed."
        )


def _runtime_test_supported_query() -> None:
    """
    Verify supported optimizer utilities.
    """

    logger.info("Running utility tests...")

    supported = get_supported_optimizers()

    assert isinstance(
        supported,
        tuple,
    )

    assert len(supported) > 0

    assert is_supported_optimizer("COBYLA")

    assert is_supported_optimizer("SPSA")

    assert is_supported_optimizer("SLSQP")

    assert not is_supported_optimizer("ABC")

    logger.info(
        "✓ Utility tests passed."
    )


def _runtime_test_factory() -> None:
    """
    Verify factory construction.
    """

    logger.info("Running factory tests...")

    config = OptimizerConfig(
        optimizer_name="COBYLA",
        maxiter=50,
        tol=1e-6,
    )

    factory = QuantumOptimizerFactory(config)

    optimizer = factory.build()

    assert isinstance(
        optimizer,
        Optimizer,
    )

    logger.info(
        "✓ Factory test passed."
    )


def run_runtime_tests() -> None:
    """
    Execute all runtime tests.

    Raises:
        RuntimeError:
            If any runtime test fails.
    """

    logger.info(
        "=" * 70
    )
    logger.info(
        "Starting optimizer runtime tests..."
    )
    logger.info(
        "=" * 70
    )

    try:

        _runtime_test_supported_optimizers()

        _runtime_test_default_optimizer()

        _runtime_test_invalid_optimizer()

        _runtime_test_invalid_iterations()

        _runtime_test_invalid_tolerance()

        _runtime_test_supported_query()

        _runtime_test_factory()

        logger.info(
            "=" * 70
        )
        logger.info(
            "All optimizer runtime tests passed successfully."
        )
        logger.info(
            "=" * 70
        )

    except Exception as exc:

        logger.exception(
            "Optimizer runtime tests failed."
        )

        raise RuntimeError(
            "Runtime tests failed."
        ) from exc


# =============================================================================
# Standalone Execution
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 72)
    print("ARQPO Quantum Optimizer Factory")
    print("=" * 72)

    run_runtime_tests()

    print("\nSupported Optimizers:")

    for optimizer_name in get_supported_optimizers():

        print(f"  • {optimizer_name}")

    print("\nCreating demonstration optimizer...\n")

    optimizer = get_optimizer(
        optimizer_name="COBYLA",
        maxiter=200,
        tol=1e-6,
    )

    print("Created Optimizer:")
    print(type(optimizer).__name__)

    print("\nAll tests completed successfully.")