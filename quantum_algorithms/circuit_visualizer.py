"""
Module: quantum_algorithms.circuit_visualizer

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Production-quality visualization and inspection utilities for PennyLane
    quantum circuits (QNodes and QuantumTapes) generated throughout the
    ARQPO optimization workflow.

Workflow Position:
    Hamiltonian Builder -> Ansatz Builder -> QAOA / VQE Solver
    -> Circuit Visualizer  <-- Current Module
    -> Explainability -> Streamlit Dashboard

Responsibilities:
    * Validate PennyLane QNodes and tapes prior to visualization.
    * Render circuits using PennyLane text drawing (qml.draw) and Matplotlib (qml.draw_mpl).
    * Export rendered circuit diagrams to disk.
    * Extract circuit complexity metrics (gate count, operation breakdown, depth, wires).
    * Zero Qiskit runtime dependencies.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pennylane as qml

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_DRAW_STYLE = "text"
TEXT_DRAW_STYLE = "text"
MATPLOTLIB_DRAW_STYLE = "mpl"

SUPPORTED_DRAW_STYLES: Set[str] = {
    TEXT_DRAW_STYLE,
    MATPLOTLIB_DRAW_STYLE,
}

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

@dataclass
class CircuitStatistics:
    """
    Stores complexity and structural metadata describing a PennyLane quantum circuit.

    Attributes:
        depth: Logical circuit depth.
        width: Number of quantum wires.
        num_qubits: Equivalent to wire count.
        gate_count: Total number of quantum gates/operations.
        operation_counts: Breakdown of counts per gate type.
        parameter_count: Total free/trainable parameters.
    """
    depth: int = 0
    width: int = 0
    num_qubits: int = 0
    gate_count: int = 0
    operation_counts: Dict[str, int] = field(default_factory=dict)
    parameter_count: int = 0


# =============================================================================
# Circuit Visualizer Class
# =============================================================================

class CircuitVisualizer:
    """
    Production-quality visualization and structural inspection utility for
    PennyLane quantum circuits.
    """

    def __init__(self, default_style: str = DEFAULT_DRAW_STYLE) -> None:
        self.default_style = self._validate_style(default_style)

    @staticmethod
    def _validate_style(style: str) -> str:
        if not isinstance(style, str):
            raise TypeError(f"Visualization style must be a string, got {type(style).__name__}.")
        norm_style = style.strip().lower()
        if norm_style not in SUPPORTED_DRAW_STYLES:
            raise ValueError(
                f"Unsupported visualization style '{style}'. "
                f"Supported styles: {sorted(SUPPORTED_DRAW_STYLES)}"
            )
        return norm_style

    def _get_tape(self, circuit: Any, *args: Any, **kwargs: Any) -> qml.tape.QuantumScript:
        """
        Extract the underlying QuantumScript/QuantumTape from a QNode or tape.
        """
        if isinstance(circuit, qml.tape.QuantumScript):
            return circuit

        if isinstance(circuit, qml.QNode):
            # Construct tape by taping the QNode with supplied arguments
            tape = qml.workflow.construct_tape(circuit)(*args, **kwargs)
            return tape

        raise TypeError(
            f"Circuit must be a PennyLane QNode or QuantumTape, got {type(circuit).__name__}."
        )

    def extract_statistics(self, circuit: Any, *args: Any, **kwargs: Any) -> CircuitStatistics:
        """
        Extract structural complexity metrics from a PennyLane quantum circuit.
        """
        tape = self._get_tape(circuit, *args, **kwargs)

        operations = tape.operations
        gate_count = len(operations)
        wire_set = tape.wires
        num_qubits = len(wire_set)

        op_counts: Dict[str, int] = {}
        param_count = 0

        for op in operations:
            name = op.name
            op_counts[name] = op_counts.get(name, 0) + 1
            param_count += len(op.data)

        # Compute depth via tape graph or default to operations
        try:
            depth = tape.graph.get_depth() if hasattr(tape, "graph") else len(operations)
        except Exception:
            depth = len(operations)

        return CircuitStatistics(
            depth=depth,
            width=num_qubits,
            num_qubits=num_qubits,
            gate_count=gate_count,
            operation_counts=op_counts,
            parameter_count=param_count,
        )

    def draw(
        self,
        circuit: Any,
        *args: Any,
        style: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Draw a PennyLane circuit using text or Matplotlib rendering.

        Args:
            circuit: PennyLane QNode or QuantumTape.
            *args: Arguments required to evaluate the QNode.
            style: 'text' (default) or 'mpl'.
            **kwargs: Additional keyword arguments.

        Returns:
            String ASCII diagram for 'text' style, or Matplotlib Figure for 'mpl' style.
        """
        chosen_style = self._validate_style(style) if style is not None else self.default_style

        if not isinstance(circuit, (qml.QNode, qml.tape.QuantumScript)):
            raise TypeError(
                f"circuit must be a PennyLane QNode or QuantumTape, got {type(circuit).__name__}."
            )

        if chosen_style == TEXT_DRAW_STYLE:
            if isinstance(circuit, qml.QNode):
                return qml.draw(circuit)(*args, **kwargs)
            else:
                return qml.drawer.tape_text(circuit)

        elif chosen_style == MATPLOTLIB_DRAW_STYLE:
            if isinstance(circuit, qml.QNode):
                fig, ax = qml.draw_mpl(circuit)(*args, **kwargs)
                return fig
            else:
                fig, ax = qml.drawer.tape_mpl(circuit)
                return fig

    def summarize(self, circuit: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Return a summary dictionary of circuit complexity metrics.
        """
        stats = self.extract_statistics(circuit, *args, **kwargs)
        return {
            "num_qubits": stats.num_qubits,
            "width": stats.width,
            "depth": stats.depth,
            "gate_count": stats.gate_count,
            "operation_counts": stats.operation_counts,
            "parameter_count": stats.parameter_count,
        }

    def save(
        self,
        circuit: Any,
        output_path: Union[str, Path],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Render and save a circuit diagram to disk.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if out_p.suffix.lower() == ".txt":
            text_repr = self.draw(circuit, *args, style=TEXT_DRAW_STYLE, **kwargs)
            out_p.write_text(str(text_repr), encoding="utf-8")
        else:
            fig = self.draw(circuit, *args, style=MATPLOTLIB_DRAW_STYLE, **kwargs)
            fig.savefig(str(out_p), bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)

        logger.info("Circuit visualization saved to %s", out_p)
        return str(out_p)
