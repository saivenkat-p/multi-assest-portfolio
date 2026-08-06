"""
Module: 04_quantum_algorithms.circuit_visualizer

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Production-quality visualization utilities for quantum circuits generated
    throughout the ARQPO optimization workflow.

Workflow Position:
    Hamiltonian Builder
            ↓
        Ansatz Builder
            ↓
      QAOA / VQE Solver
            ↓
    Backend Execution
            ↓
    Circuit Visualizer   <-- This module
            ↓
    Explainability
            ↓
    Streamlit Dashboard

Responsibilities:
    * Validate quantum circuits prior to visualization.
    * Render circuits using supported Qiskit drawing backends.
    * Save rendered circuit diagrams.
    * Extract circuit statistics.
    * Provide compatibility wrappers for Qiskit visualization APIs.
    * Support downstream explainability and dashboard modules.

Design Philosophy:
    This module is intentionally read-only. It never modifies a quantum
    circuit. Instead, it validates, visualizes, summarizes, and exports
    information from existing QuantumCircuit instances.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Union

from qiskit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError


# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_DRAW_STYLE = "mpl"
TEXT_DRAW_STYLE = "text"
MATPLOTLIB_DRAW_STYLE = "mpl"

SUPPORTED_DRAW_STYLES: Set[str] = {
    TEXT_DRAW_STYLE,
    MATPLOTLIB_DRAW_STYLE,
}

DEFAULT_FIGURE_SCALE = 1.0
FLOAT_TOLERANCE = 1e-12

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
# Data Models
# =============================================================================


@dataclass(slots=True)
class CircuitStatistics:
    """
    Stores metadata describing a quantum circuit.

    Attributes:
        depth:
            Logical circuit depth.

        width:
            Total circuit width (quantum + classical).

        num_qubits:
            Number of quantum bits.

        num_clbits:
            Number of classical bits.

        gate_count:
            Total number of operations.

        operation_counts:
            Dictionary containing counts for each gate type.

        parameter_count:
            Number of free parameters in the circuit.

        metadata:
            Additional implementation-specific metadata.
    """

    depth: int = 0
    width: int = 0
    num_qubits: int = 0
    num_clbits: int = 0
    gate_count: int = 0

    operation_counts: Dict[str, int] = field(default_factory=dict)

    parameter_count: int = 0

    metadata: Dict[str, Union[str, float, int]] = field(
        default_factory=dict
    )

# =============================================================================
# Circuit Visualizer
# =============================================================================


class CircuitVisualizer:
    """
    Production-quality visualization utility for Qiskit QuantumCircuit objects.

    The class provides a stable public interface for rendering,
    exporting, and summarizing circuits while isolating Qiskit-specific
    implementation details behind compatibility helpers.

    Notes:
        This class never mutates the supplied QuantumCircuit.
        All operations are read-only.
    """

    def __init__(
        self,
        default_style: str = DEFAULT_DRAW_STYLE,
        figure_scale: float = DEFAULT_FIGURE_SCALE,
    ) -> None:
        """
        Initialize the circuit visualizer.

        Args:
            default_style:
                Default visualization backend.

            figure_scale:
                Scaling factor applied to rendered figures.

        Raises:
            ValueError:
                If an invalid configuration is supplied.
        """
        self._validate_style(default_style)

        if figure_scale <= 0:
            raise ValueError(
                "figure_scale must be greater than zero."
            )

        self.default_style = default_style
        self.figure_scale = float(figure_scale)

        logger.info(
            "CircuitVisualizer initialized "
            "(style=%s, scale=%.2f)",
            self.default_style,
            self.figure_scale,
        )

    # =====================================================================
    # Validation Helpers
    # =====================================================================

    @staticmethod
    def _validate_circuit(circuit: QuantumCircuit) -> None:
        """
        Validate a QuantumCircuit instance.

        Args:
            circuit:
                Circuit to validate.

        Raises:
            TypeError:
                If the object is not a QuantumCircuit.

            ValueError:
                If the circuit is invalid or contains no qubits.
        """
        if circuit is None:
            logger.error("Received None instead of a QuantumCircuit.")
            raise TypeError("Circuit cannot be None.")

        if not isinstance(circuit, QuantumCircuit):
            logger.error(
                "Expected QuantumCircuit, received %s.",
                type(circuit).__name__,
            )
            raise TypeError(
                "Expected a qiskit.QuantumCircuit instance."
            )

        if circuit.num_qubits <= 0:
            logger.error("Circuit contains zero qubits.")
            raise ValueError(
                "Quantum circuit must contain at least one qubit."
            )

    @staticmethod
    def _validate_style(style: str) -> None:
        """
        Validate the requested drawing backend.

        Args:
            style:
                Drawing backend.

        Raises:
            TypeError:
                If style is not a string.

            ValueError:
                If the style is unsupported.
        """
        if not isinstance(style, str):
            raise TypeError("Drawing style must be a string.")

        style = style.strip().lower()

        if style not in SUPPORTED_DRAW_STYLES:
            raise ValueError(
                f"Unsupported drawing style '{style}'. "
                f"Supported styles: {sorted(SUPPORTED_DRAW_STYLES)}"
            )

    @staticmethod
    def _validate_output_path(
        output_path: Optional[Union[str, Path]]
    ) -> Optional[Path]:
        """
        Validate an output file path.

        Args:
            output_path:
                Destination file.

        Returns:
            Normalized pathlib.Path object or None.

        Raises:
            TypeError:
                If the supplied path has an invalid type.

            ValueError:
                If the destination directory does not exist.
        """
        if output_path is None:
            return None

        if not isinstance(output_path, (str, Path)):
            raise TypeError(
                "output_path must be a string, Path, or None."
            )

        output_path = Path(output_path)

        parent = output_path.parent

        if parent and not parent.exists():
            raise ValueError(
                f"Directory does not exist: {parent}"
            )

        return output_path

    # =====================================================================
    # Compatibility Helpers
    # =====================================================================

    @staticmethod
    def _normalize_style(style: Optional[str]) -> str:
        """
        Normalize a drawing style.

        Args:
            style:
                User supplied style.

        Returns:
            Normalized style string.
        """
        if style is None:
            return DEFAULT_DRAW_STYLE
        
        if not isinstance(style, str):
            raise TypeError("Drawing style must be a string.")


        return style.strip().lower()

    def _resolve_style(
        self,
        style: Optional[str],
    ) -> str:
        """
        Resolve the effective drawing backend.

        Args:
            style:
                Optional override.

        Returns:
            Validated drawing backend.
        """
        resolved_style = (
            self.default_style
            if style is None
            else self._normalize_style(style)
        )

        self._validate_style(resolved_style)

        return resolved_style

    @staticmethod
    def _safe_draw(
        circuit: QuantumCircuit,
        style: str,
        filename: Optional[Path] = None,
    ):
        """
        Compatibility wrapper around QuantumCircuit.draw().

        This isolates Qiskit-specific drawing behaviour from the
        public API, making future compatibility updates localized.

        Args:
            circuit:
                Quantum circuit.

            style:
                Drawing backend.

            filename:
                Optional destination file.

        Returns:
            Object returned by QuantumCircuit.draw().
        """
        kwargs = {"output": style}

        if filename is not None:
            kwargs["filename"] = str(filename)

        return circuit.draw(**kwargs)

    # =====================================================================
    # Public API
    # =====================================================================

    def draw(
        self,
        circuit: QuantumCircuit,
        style: Optional[str] = None,
    ):
        """
        Draw a quantum circuit.

        Args:
            circuit:
                Quantum circuit to visualize.

            style:
                Optional drawing backend.

        Returns:
            Rendered object returned by Qiskit's draw().

        Raises:
            RuntimeError:
                If rendering fails.
        """
        start_time = time.perf_counter()

        try:
            self._validate_circuit(circuit)

            resolved_style = self._resolve_style(style)

            logger.info(
                "Rendering quantum circuit using '%s' backend.",
                resolved_style,
            )

            drawing = self._safe_draw(
                circuit=circuit,
                style=resolved_style,
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "Circuit rendered successfully in %.6f seconds.",
                elapsed,
            )

            return drawing

        except Exception as exc:
            logger.exception("Circuit rendering failed.")
            raise RuntimeError(
                "Unable to render quantum circuit."
            ) from exc

    # -----------------------------------------------------------------

    def save(
        self,
        circuit: QuantumCircuit,
        output_path: Union[str, Path],
        style: Optional[str] = None,
    ) -> Path:
        """
        Save a rendered quantum circuit.

        Args:
            circuit:
                Quantum circuit.

            output_path:
                Destination image path.

            style:
                Drawing backend.

        Returns:
            Path of the saved file.

        Raises:
            RuntimeError:
                If saving fails.
        """
        start_time = time.perf_counter()

        try:
            self._validate_circuit(circuit)

            destination = self._validate_output_path(output_path)

            resolved_style = self._resolve_style(style)

            logger.info(
                "Saving circuit diagram to '%s'.",
                destination,
            )

            self._safe_draw(
                circuit=circuit,
                style=resolved_style,
                filename=destination,
            )

            elapsed = time.perf_counter() - start_time

            logger.info(
                "Circuit saved successfully in %.6f seconds.",
                elapsed,
            )

            return destination
        

        
        
        
        except MissingOptionalLibraryError:
            logger.warning(
                "Matplotlib visualization requires "
                "'pylatexenc'. Install with:\n"
                "pip install pylatexenc"
            )
            raise
        
        
        
        
        
        except Exception as exc:
            logger.exception("Saving circuit diagram failed.")
            raise RuntimeError(
                "Unable to save circuit visualization."
            ) from exc

    # -----------------------------------------------------------------

    def extract_statistics(
        self,
        circuit: QuantumCircuit,
    ) -> CircuitStatistics:
        """
        Extract structural statistics from a quantum circuit.

        Args:
            circuit:
                Quantum circuit.

        Returns:
            CircuitStatistics object.
        """
        start_time = time.perf_counter()

        try:
            self._validate_circuit(circuit)

            stats = CircuitStatistics()

            stats.depth = circuit.depth()

            stats.width = circuit.width()

            stats.num_qubits = circuit.num_qubits

            stats.num_clbits = circuit.num_clbits

            stats.gate_count = len(circuit.data)

            stats.operation_counts = dict(circuit.count_ops())

            stats.parameter_count = len(circuit.parameters)

            stats.metadata["draw_style"] = self.default_style

            stats.metadata["execution_time"] = (
                time.perf_counter() - start_time
            )

            logger.info(
                "Circuit statistics extracted successfully."
            )

            return stats

        except Exception as exc:
            logger.exception(
                "Failed to extract circuit statistics."
            )
            raise RuntimeError(
                "Unable to extract circuit statistics."
            ) from exc

    # -----------------------------------------------------------------

    def summarize(
        self,
        circuit: QuantumCircuit,
    ) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """
        Return circuit statistics as a serializable dictionary.

        Args:
            circuit:
                Quantum circuit.

        Returns:
            Dictionary containing circuit statistics.
        """
        stats = self.extract_statistics(circuit)

        return {
            "depth": stats.depth,
            "width": stats.width,
            "num_qubits": stats.num_qubits,
            "num_clbits": stats.num_clbits,
            "gate_count": stats.gate_count,
            "parameter_count": stats.parameter_count,
            "operation_counts": stats.operation_counts,
            "metadata": stats.metadata,
        }

# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":
    """
    Independent runtime tests for CircuitVisualizer.

    These tests verify:
        * Object construction
        * Text rendering
        * Statistics extraction
        * Saving circuit diagrams
        * Validation behavior

    Expected:
        All tests should complete without raising exceptions.
    """

    from qiskit import QuantumCircuit

    print("=" * 70)
    print("ARQPO Circuit Visualizer Runtime Tests")
    print("=" * 70)

    visualizer = CircuitVisualizer()

    # -----------------------------------------------------------------
    # Test 1: Simple Bell Circuit
    # -----------------------------------------------------------------

    print("\n[TEST 1] Bell Circuit")

    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])

    drawing = visualizer.draw(
        bell,
        style=TEXT_DRAW_STYLE,
    )

    print(drawing)

    stats = visualizer.extract_statistics(bell)

    print("\nStatistics")

    print(f"Depth           : {stats.depth}")
    print(f"Width           : {stats.width}")
    print(f"Qubits          : {stats.num_qubits}")
    print(f"Classical Bits  : {stats.num_clbits}")
    print(f"Gate Count      : {stats.gate_count}")
    print(f"Parameters      : {stats.parameter_count}")
    print(f"Operations      : {stats.operation_counts}")

    # -----------------------------------------------------------------
    # Test 2: Dictionary Summary
    # -----------------------------------------------------------------

    print("\n[TEST 2] Summary Dictionary")

    summary = visualizer.summarize(bell)

    for key, value in summary.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------------------
    # Test 3: Save Figure
    # -----------------------------------------------------------------

    print("\n[TEST 3] Saving Circuit")

    try:
        filename = visualizer.save(
            bell,
            output_path="bell_circuit.png",
        )

        print(f"Saved to: {filename}")

    except Exception as exc:
        print(
            "Skipping save test "
            "(Matplotlib may not be installed)."
        )
        print(exc)

    # -----------------------------------------------------------------
    # Test 4: Invalid Style
    # -----------------------------------------------------------------

    print("\n[TEST 4] Invalid Style")

    try:
        visualizer.draw(
            bell,
            style="invalid_style",
        )

    except Exception as exc:
        print("Expected exception:")
        print(exc)

    # -----------------------------------------------------------------
    # Test 5: Invalid Circuit
    # -----------------------------------------------------------------

    print("\n[TEST 5] Invalid Circuit")

    try:
        visualizer.draw(None)

    except Exception as exc:
        print("Expected exception:")
        print(exc)

    # -----------------------------------------------------------------
    # Test 6: Empty Circuit
    # -----------------------------------------------------------------

    print("\n[TEST 6] Empty Circuit")

    try:
        zero_operation = QuantumCircuit(1)

        stats = visualizer.extract_statistics(zero_operation)

        print(stats)

    except Exception as exc:
        print(exc)

    
        # -----------------------------------------------------------------
    # Test 7: Invalid Output Path
    # -----------------------------------------------------------------

    print("\n[TEST 7] Invalid Output Path")

    try:
        visualizer.save(
            bell,
            output_path="non_existent_folder/bell.png",
        )

    except Exception as exc:
        print("Expected exception:")
        print(exc)

    # -----------------------------------------------------------------
    # Test 8: Repeated Draw Calls
    # -----------------------------------------------------------------

    print("\n[TEST 8] Repeated Draw Calls")

    try:
        for i in range(3):
            visualizer.draw(
                bell,
                style=TEXT_DRAW_STYLE,
            )
            print(f"Draw {i + 1} completed.")

    except Exception as exc:
        print(exc)

    # -----------------------------------------------------------------
    # Test 9: Statistics Stability
    # -----------------------------------------------------------------

    print("\n[TEST 9] Statistics Stability")

    try:
        stats1 = visualizer.extract_statistics(bell)
        stats2 = visualizer.extract_statistics(bell)

        assert stats1.depth == stats2.depth
        assert stats1.width == stats2.width
        assert stats1.gate_count == stats2.gate_count
        assert (
            stats1.operation_counts
            == stats2.operation_counts
        )

        print("Statistics are deterministic.")

    except Exception as exc:
        print(exc)

    # -----------------------------------------------------------------
    # Test 10: Invalid Output Path Type
    # -----------------------------------------------------------------

    print("\n[TEST 10] Invalid Output Path Type")

    try:
        visualizer.save(
            bell,
            output_path=12345,
        )

    except Exception as exc:
        print("Expected exception:")
        print(exc)

    # -----------------------------------------------------------------
    # Test 11: Invalid Style Type
    # -----------------------------------------------------------------

    print("\n[TEST 11] Invalid Style Type")

    try:
        visualizer.draw(
            bell,
            style=123,
        )

    except Exception as exc:
        print("Expected exception:")
        print(exc)
        
        print("\n" + "=" * 70)
        print("All runtime tests completed.")
        print("=" * 70)