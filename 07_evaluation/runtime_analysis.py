"""
Module: 07_evaluation.runtime_analysis

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)
    WISER × Vanguard Quantum Challenge 2026

Purpose:
    Provides production-grade runtime profiling utilities for measuring
    the computational performance of portfolio optimization algorithms.

Workflow:
    Solver Function
          │
          ▼
    Input Validation
          │
          ▼
    Runtime Measurement
          │
          ▼
    Statistical Analysis
          │
          ▼
    Metadata Generation
          │
          ▼
    Runtime Report

Design Philosophy:
    • Deterministic profiling
    • High-resolution timing
    • Defensive programming
    • Production logging
    • Reusable profiling framework
    • Backward compatibility

Author:
    ARQPO Research Project
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_REPETITIONS: int = 10

FLOAT_TOLERANCE: float = 1e-10

REPORT_VERSION: str = "1.0"

LOGGER_NAME: str = __name__

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

# =============================================================================
# Runtime Profiler
# =============================================================================

class RuntimeProfiler:
    """
    Profiles execution time for classical and quantum optimization
    algorithms.

    This profiler provides deterministic runtime measurements using
    Python's high-resolution performance counter.

    Attributes:
        repetitions:
            Number of repeated executions used for statistical profiling.
    """

    def __init__(
        self,
        repetitions: int = DEFAULT_REPETITIONS,
    ) -> None:
        """
        Initialize the runtime profiler.

        Args:
            repetitions:
                Number of repeated executions for statistical analysis.

        Raises:
            ValueError:
                If repetitions is invalid.
        """

        logger.info("Initializing RuntimeProfiler.")

        self._validate_repetitions(repetitions)

        self.repetitions = repetitions

        logger.info(
            "RuntimeProfiler initialized successfully."
        )

    # -------------------------------------------------------------------------
    # Validation Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_callable(
        solver_function: Callable[..., Any],
    ) -> None:
        """
        Validate the supplied solver function.

        Args:
            solver_function:
                Callable solver.

        Raises:
            TypeError:
                If the object is not callable.
        """

        if solver_function is None:
            raise TypeError(
                "Solver function cannot be None."
            )

        if not callable(solver_function):
            raise TypeError(
                "solver_function must be callable."
            )

    @staticmethod
    def _validate_repetitions(
        repetitions: int,
    ) -> None:
        """
        Validate repetition count.

        Args:
            repetitions:
                Number of executions.

        Raises:
            ValueError:
                If repetitions are invalid.
        """

        if not isinstance(repetitions, int):
            raise TypeError(
                "repetitions must be an integer."
            )

        if repetitions <= 0:
            raise ValueError(
                "repetitions must be greater than zero."
            )

    @staticmethod
    def _validate_runtime(
        runtime: float,
    ) -> float:
        """
        Validate measured runtime.

        Args:
            runtime:
                Execution time.

        Returns:
            Validated runtime.

        Raises:
            ValueError:
                If runtime is invalid.
        """

        if not np.isfinite(runtime):
            raise ValueError(
                "Runtime must be finite."
            )

        if runtime < 0.0:
            raise ValueError(
                "Runtime cannot be negative."
            )

        if np.isclose(
            runtime,
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            return 0.0

        return float(runtime)

    # -------------------------------------------------------------------------
    # Runtime Measurement Helpers
    # -------------------------------------------------------------------------

    def _measure_once(
        self,
        solver_function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[float, Any]:
        """
        Execute a solver exactly once and measure its runtime.

        Args:
            solver_function:
                Solver function to execute.
            *args:
                Positional arguments passed to the solver.
            **kwargs:
                Keyword arguments passed to the solver.

        Returns:
            Tuple containing:
                runtime (seconds)
                solver output

        Raises:
            RuntimeError:
                If solver execution fails.
        """

        self._validate_callable(solver_function)

        solver_name = getattr(
            solver_function,
            "__qualname__",
            str(solver_function),
        )

        logger.debug(
            "Profiling single execution for '%s'.",
            solver_name,
        )

        start_time = time.perf_counter()

        try:
            result = solver_function(*args, **kwargs)

        except Exception as exc:
            logger.exception(
                "Solver execution failed for '%s'.",
                solver_name,
            )
            raise RuntimeError(
                f"Execution failed for '{solver_name}'."
            ) from exc

        elapsed_time = time.perf_counter() - start_time

        elapsed_time = self._validate_runtime(
            elapsed_time
        )

        logger.info(
            "Execution completed in %.8f seconds.",
            elapsed_time,
        )

        return elapsed_time, result

    def _measure_multiple(
        self,
        solver_function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[float], Any]:
        """
        Execute a solver repeatedly.

        Args:
            solver_function:
                Solver to execute.

        Returns:
            List of runtimes and the final solver output.
        """

        logger.info(
            "Running %d profiling repetitions.",
            self.repetitions,
        )

        runtimes: List[float] = []

        final_output: Optional[Any] = None

        for run in range(self.repetitions):

            logger.debug(
                "Repetition %d/%d",
                run + 1,
                self.repetitions,
            )

            runtime, final_output = self._measure_once(
                solver_function,
                *args,
                **kwargs,
            )

            runtimes.append(runtime)

        logger.info(
            "Repeated profiling completed."
        )

        return runtimes, final_output

    # -------------------------------------------------------------------------
    # Numerical Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _cleanup_runtime(
        runtime: float,
    ) -> float:
        """
        Remove floating-point artifacts from measured runtime.

        Args:
            runtime:
                Measured execution time.

        Returns:
            Clean runtime.
        """

        if np.isclose(
            runtime,
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            return 0.0

        return float(runtime)

    def _compute_statistics(
        self,
        runtimes: List[float],
    ) -> Dict[str, float]:
        """
        Compute runtime statistics.

        Args:
            runtimes:
                List of measured runtimes.

        Returns:
            Dictionary containing summary statistics.
        """

        runtime_array = np.asarray(
            runtimes,
            dtype=float,
        )
        if runtime_array.size == 0:
            raise ValueError(
                "Runtime list cannot be empty."
            )
        

        statistics = {
            "mean_time": self._cleanup_runtime(
                float(np.mean(runtime_array))
            ),
            "std_time": self._cleanup_runtime(
                float(np.std(runtime_array))
            ),
            "min_time": self._cleanup_runtime(
                float(np.min(runtime_array))
            ),
            "max_time": self._cleanup_runtime(
                float(np.max(runtime_array))
            ),
            "median_time": self._cleanup_runtime(
                float(np.median(runtime_array))
            ),
        }

        logger.info(
            "Runtime statistics computed successfully."
        )

        return statistics

    def _generate_metadata(
        self,
        solver_name: str,
        repetitions: int,
    ) -> Dict[str, Any]:
        """
        Generate runtime profiling metadata.

        Args:
            solver_name:
                Name of the profiled solver.

            repetitions:
                Number of executions.

        Returns:
            Metadata dictionary.
        """

        return {
            "report_version": REPORT_VERSION,
            "solver_name": solver_name,
            "repetitions": repetitions,
            "profiler": self.__class__.__name__,
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def measure_execution_time(
        self,
        solver_function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[float, Any]:
        """
        Execute a solver once and measure its runtime.

        This method preserves the original public API for backward
        compatibility while internally using the production runtime
        measurement engine.

        Args:
            solver_function:
                Solver function to execute.
            *args:
                Positional arguments passed to the solver.
            **kwargs:
                Keyword arguments passed to the solver.

        Returns:
            Tuple containing:

            - execution time (seconds)
            - solver output

        Raises:
            RuntimeError:
                If solver execution fails.
        """

        logger.info("=" * 70)
        logger.info("Starting runtime profiling.")
        logger.info("=" * 70)

        try:

            runtime, output = self._measure_once(
                solver_function,
                *args,
                **kwargs,
            )

            logger.info(
                "Runtime profiling completed successfully."
            )

            logger.info("=" * 70)

            return runtime, output

        except Exception as exc:

            logger.exception(
                "Runtime profiling failed."
            )

            raise RuntimeError(
                "Failed to measure execution time."
            ) from exc

    def profile_multiple(
        self,
        solver_function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute repeated runtime profiling and generate a complete
        statistical report.

        Args:
            solver_function:
                Solver function to profile.
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.

        Returns:
            Production runtime profiling report.

        Raises:
            RuntimeError:
                If profiling fails.
        """

        logger.info("=" * 70)
        logger.info("Starting statistical runtime profiling.")
        logger.info("=" * 70)

        profiler_start = time.perf_counter()

        try:

            runtimes, output = self._measure_multiple(
                solver_function,
                *args,
                **kwargs,
            )

            statistics = self._compute_statistics(
                runtimes
            )

            solver_name = getattr(
                solver_function,
                "__qualname__",
                str(solver_function),
            )

            metadata = self._generate_metadata(
                solver_name=solver_name,
                repetitions=self.repetitions,
            )

            profiler_runtime = (
                time.perf_counter() - profiler_start
            )

            metadata["profiling_time_seconds"] = (
                self._cleanup_runtime(
                    profiler_runtime
                )  
            )
            metadata["successful_runs"] = len(runtimes)
            report = {
                "statistics": statistics,
                "individual_runtimes": runtimes,
                "solver_output": output,
                "metadata": metadata,
            }

            logger.info(
                "Statistical runtime profiling completed."
            )

            logger.info(
                "Average Runtime : %.8f sec",
                statistics["mean_time"],
            )

            logger.info(
                "Minimum Runtime : %.8f sec",
                statistics["min_time"],
            )

            logger.info(
                "Maximum Runtime : %.8f sec",
                statistics["max_time"],
            )

            logger.info(
                "Standard Deviation : %.8f sec",
                statistics["std_time"],
            )

            logger.info("=" * 70)

            return report

        except Exception as exc:

            logger.exception(
                "Statistical runtime profiling failed."
            )

            raise RuntimeError(
                "Runtime analysis failed."
            ) from exc

# =============================================================================
# Runtime Tests
# =============================================================================

def _sample_solver(values: np.ndarray) -> np.ndarray:
    """Simple deterministic solver used for runtime testing."""
    return np.sort(values)


def _runtime_test_happy_path() -> None:
    """Test normal runtime measurement."""

    profiler = RuntimeProfiler()

    runtime, output = profiler.measure_execution_time(
        _sample_solver,
        np.array([3, 1, 2]),
    )

    assert runtime >= 0.0
    assert np.array_equal(output, np.array([1, 2, 3]))

    logger.info("Happy Path Test PASSED.")


def _runtime_test_medium_input() -> None:
    """Test profiling using medium-sized input."""

    profiler = RuntimeProfiler()

    values = np.random.rand(1000)

    report = profiler.profile_multiple(
        _sample_solver,
        values,
    )

    assert report["statistics"]["mean_time"] >= 0.0

    logger.info("Medium Input Test PASSED.")


def _runtime_test_large_input() -> None:
    """Test profiling using a large input array."""

    profiler = RuntimeProfiler(repetitions=5)

    values = np.random.rand(100000)

    report = profiler.profile_multiple(
        _sample_solver,
        values,
    )

    assert report["statistics"]["max_time"] >= 0.0

    logger.info("Large Input Test PASSED.")


def _runtime_test_invalid_callable() -> None:
    """Ensure invalid callables raise an exception."""

    profiler = RuntimeProfiler()

    try:

        profiler.measure_execution_time(
            None,
        )

    except RuntimeError:
        logger.info("Invalid Callable Test PASSED.")
        return

    raise AssertionError(
        "Expected RuntimeError."
    )


def _runtime_test_invalid_arguments() -> None:
    """Ensure solver argument errors propagate correctly."""

    profiler = RuntimeProfiler()

    try:

        profiler.measure_execution_time(
            _sample_solver,
        )

    except RuntimeError:
        logger.info("Invalid Arguments Test PASSED.")
        return

    raise AssertionError(
        "Expected RuntimeError."
    )


def _runtime_test_edge_case() -> None:
    """Profile a single-element input."""

    profiler = RuntimeProfiler()

    runtime, output = profiler.measure_execution_time(
        _sample_solver,
        np.array([5]),
    )

    assert runtime >= 0.0
    assert output.size == 1

    logger.info("Edge Case Test PASSED.")


def _runtime_test_determinism() -> None:
    """Verify deterministic solver output."""

    profiler = RuntimeProfiler()

    report1 = profiler.profile_multiple(
        _sample_solver,
        np.array([5, 4, 3, 2, 1]),
    )

    report2 = profiler.profile_multiple(
        _sample_solver,
        np.array([5, 4, 3, 2, 1]),
    )

    assert np.array_equal(
        report1["solver_output"],
        report2["solver_output"],
    )

    logger.info("Determinism Test PASSED.")


def _runtime_test_stress() -> None:
    """Repeated profiling stress test."""

    profiler = RuntimeProfiler(
        repetitions=3,
    )

    for _ in range(20):

        profiler.profile_multiple(
            _sample_solver,
            np.random.rand(500),
        )

    logger.info("Stress Test PASSED.")


def _runtime_test_numerical_robustness() -> None:
    """Verify runtime statistics remain finite."""

    profiler = RuntimeProfiler()

    report = profiler.profile_multiple(
        _sample_solver,
        np.array([1.0]),
    )

    stats = report["statistics"]

    assert np.isfinite(stats["mean_time"])
    assert np.isfinite(stats["std_time"])
    assert np.isfinite(stats["min_time"])
    assert np.isfinite(stats["max_time"])
    assert np.isfinite(stats["median_time"])

    logger.info("Numerical Robustness Test PASSED.")


def _runtime_test_integration() -> None:
    """Verify report structure."""

    profiler = RuntimeProfiler()

    report = profiler.profile_multiple(
        _sample_solver,
        np.random.rand(10),
    )

    assert "statistics" in report
    assert "metadata" in report
    assert "solver_output" in report
    assert "individual_runtimes" in report

    logger.info("Integration Test PASSED.")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("RUNNING RUNTIME ANALYSIS TESTS")
    logger.info("=" * 70)

    _runtime_test_happy_path()
    _runtime_test_medium_input()
    _runtime_test_large_input()
    _runtime_test_invalid_callable()
    _runtime_test_invalid_arguments()
    _runtime_test_edge_case()
    _runtime_test_determinism()
    _runtime_test_stress()
    _runtime_test_numerical_robustness()
    _runtime_test_integration()

    logger.info("=" * 70)
    logger.info("ALL RUNTIME TESTS PASSED")
    logger.info("=" * 70)