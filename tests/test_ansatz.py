"""
Unit tests for the PennyLane portfolio ansatz module.
"""

import numpy as np
import pennylane as qml
import pytest

from quantum_algorithms.ansatz import (
    calculate_parameter_count,
    generate_initial_parameters,
    portfolio_ansatz,
    build_portfolio_ansatz,
    validate_ansatz_inputs,
)


def test_calculate_parameter_count():
    # 3 qubits, reps=2 -> (2 + 1) * 3 * 2 = 18 parameters
    assert calculate_parameter_count(3, 2) == 18
    # 4 qubits, reps=1 -> (1 + 1) * 4 * 2 = 16 parameters
    assert calculate_parameter_count(4, 1) == 16
    # 2 qubits, reps=0 -> (0 + 1) * 2 * 2 = 4 parameters
    assert calculate_parameter_count(2, 0) == 4


def test_validate_ansatz_inputs():
    with pytest.raises(ValueError):
        validate_ansatz_inputs(num_qubits=0, reps=2, entanglement="linear")

    with pytest.raises(ValueError):
        validate_ansatz_inputs(num_qubits=4, reps=-1, entanglement="linear")

    with pytest.raises(ValueError):
        validate_ansatz_inputs(num_qubits=4, reps=2, entanglement="unknown_topology")

    with pytest.raises(TypeError):
        validate_ansatz_inputs(num_qubits="4", reps=2, entanglement="linear")


def test_portfolio_ansatz_qnode_execution():
    num_qubits = 3
    reps = 2
    dev = qml.device("default.qubit", wires=num_qubits)
    params = generate_initial_parameters(num_qubits, reps, seed=42)

    @qml.qnode(dev)
    def test_circuit():
        portfolio_ansatz(params, wires=num_qubits, reps=reps, entanglement="linear")
        return qml.probs(wires=range(num_qubits))

    probs = test_circuit()
    assert len(probs) == 2 ** num_qubits
    assert np.isclose(np.sum(probs), 1.0)
    assert (probs >= 0.0).all()


def test_portfolio_ansatz_circular_entanglement():
    num_qubits = 3
    reps = 1
    dev = qml.device("default.qubit", wires=num_qubits)
    params = generate_initial_parameters(num_qubits, reps, seed=7)

    @qml.qnode(dev)
    def test_circuit():
        portfolio_ansatz(params, wires=num_qubits, reps=reps, entanglement="circular")
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    expvals = test_circuit()
    assert len(expvals) == 3
    for ev in expvals:
        assert -1.0 <= ev <= 1.0


def test_build_portfolio_ansatz_object():
    ansatz_obj = build_portfolio_ansatz(num_qubits=2, reps=1, entanglement="linear")
    assert ansatz_obj.parameter_count == 8
    params = ansatz_obj.generate_initial_parameters(seed=10)
    assert len(params) == 8
