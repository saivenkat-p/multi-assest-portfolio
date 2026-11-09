"""
Unit tests for PennyLane circuit visualizer and statistics extractor.
"""

from pathlib import Path
import numpy as np
import pennylane as qml
import pytest

from quantum_algorithms.circuit_visualizer import (
    CircuitVisualizer,
    CircuitStatistics,
)


@pytest.fixture
def sample_qnode():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def bell_circuit(theta):
        qml.Hadamard(wires=0)
        qml.RY(theta, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    return bell_circuit


def test_visualizer_statistics(sample_qnode):
    viz = CircuitVisualizer()
    stats = viz.extract_statistics(sample_qnode, 0.5)

    assert isinstance(stats, CircuitStatistics)
    assert stats.num_qubits == 2
    assert stats.gate_count == 3
    assert stats.operation_counts["Hadamard"] == 1
    assert stats.operation_counts["RY"] == 1
    assert stats.operation_counts["CNOT"] == 1


def test_visualizer_text_draw(sample_qnode):
    viz = CircuitVisualizer()
    diagram = viz.draw(sample_qnode, 0.5, style="text")
    assert isinstance(diagram, str)
    assert "H" in diagram
    assert "CNOT" in diagram or "╭" in diagram or "C" in diagram


def test_visualizer_summarize(sample_qnode):
    viz = CircuitVisualizer()
    summary = viz.summarize(sample_qnode, 0.5)
    assert summary["num_qubits"] == 2
    assert summary["gate_count"] == 3
    assert "operation_counts" in summary


def test_visualizer_save_text(sample_qnode, tmp_path):
    viz = CircuitVisualizer()
    out_file = tmp_path / "circuit.txt"
    saved = viz.save(sample_qnode, out_file, 0.5)
    assert Path(saved).exists()
    content = Path(saved).read_text(encoding="utf-8")
    assert len(content) > 0
