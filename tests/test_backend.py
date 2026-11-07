"""
Unit tests for PennyLane backend device provisioning.
"""

import pytest
import pennylane as qml

from quantum_algorithms.backend import (
    DEFAULT_DEVICE,
    LIGHTNING_DEVICE,
    create_device,
    validate_device_name,
    validate_wires,
    validate_shots,
    is_analytic,
)


def test_validate_device_name():
    assert validate_device_name("default.qubit") == "default.qubit"
    assert validate_device_name("lightning.qubit") == "lightning.qubit"
    assert validate_device_name("local_statevector") == "default.qubit"
    assert validate_device_name("statevector") == "default.qubit"

    with pytest.raises(TypeError):
        validate_device_name(123)

    with pytest.raises(ValueError):
        validate_device_name("unsupported_supercomputer")


def test_validate_wires():
    assert validate_wires(4) == 4
    assert validate_wires([0, 1, 2]) == 3

    with pytest.raises(ValueError):
        validate_wires(0)

    with pytest.raises(ValueError):
        validate_wires(35)  # exceeds safety limit

    with pytest.raises(TypeError):
        validate_wires("invalid")


def test_validate_shots():
    assert validate_shots(None) is None
    assert validate_shots(1000) == 1000

    with pytest.raises(ValueError):
        validate_shots(-10)

    with pytest.raises(ValueError):
        validate_shots(0)

    with pytest.raises(TypeError):
        validate_shots("1000")


def test_create_device_default():
    dev = create_device("default.qubit", wires=3)
    assert dev is not None
    assert is_analytic(dev)


def test_create_device_with_shots():
    dev = create_device("default.qubit", wires=2, shots=500)
    assert dev is not None
    assert not is_analytic(dev)


def test_create_device_fallback():
    # Attempting to load lightning.qubit should either work or fall back cleanly to default.qubit
    dev = create_device(LIGHTNING_DEVICE, wires=2)
    assert dev is not None
