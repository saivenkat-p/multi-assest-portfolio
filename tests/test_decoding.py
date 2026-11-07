"""
Unit tests for bitstring decoding, dimension validation, and constraint satisfaction.
"""

import numpy as np
import pytest

from quantum_algorithms.decoding import (
    decode_bitstring_to_weights,
    qubo_energy_of_bitstring,
    validate_encoding_dimensions,
    validate_bitstring,
)


def test_validate_encoding_dimensions():
    # Valid: 3 assets * 2 bits = 6 qubits
    validate_encoding_dimensions(3, 2, 6)

    # Mismatch
    with pytest.raises(ValueError):
        validate_encoding_dimensions(3, 2, 5)

    with pytest.raises(ValueError):
        validate_encoding_dimensions(0, 2, 6)


def test_validate_bitstring():
    assert validate_bitstring("0101", 4) == "0101"

    with pytest.raises(ValueError):
        validate_bitstring("0102", 4)

    with pytest.raises(ValueError):
        validate_bitstring("01", 4)

    with pytest.raises(TypeError):
        validate_bitstring(1010, 4)


def test_decode_bitstring_normalized():
    # 2 assets, 2 bits each -> total 4 bits
    # Asset 0 bits: '11' -> int 3 -> raw 3/3 = 1.0
    # Asset 1 bits: '01' -> int 1 -> raw 1/3 = 0.333333
    # Total raw = 4/3. Normalized: w0 = 3/4 = 0.75, w1 = 1/4 = 0.25
    weights = decode_bitstring_to_weights("1101", n_assets=2, binary_bits=2, normalize=True)
    assert np.isclose(weights[0], 0.75)
    assert np.isclose(weights[1], 0.25)
    assert np.isclose(np.sum(weights), 1.0)


def test_decode_bitstring_raw_unnormalized():
    raw_weights = decode_bitstring_to_weights("1100", n_assets=2, binary_bits=2, normalize=False)
    assert np.isclose(raw_weights[0], 1.0)
    assert np.isclose(raw_weights[1], 0.0)


def test_decode_zero_bitstring_fallback():
    # '0000' -> all zeros, should fall back to uniform 1/N
    weights = decode_bitstring_to_weights("0000", n_assets=2, binary_bits=2, normalize=True)
    assert np.allclose(weights, [0.5, 0.5])


def test_qubo_energy_of_bitstring():
    Q = np.array([[2.0, -1.0], [-1.0, 3.0]])
    offset = 10.0
    # b = [1, 1] -> b^T Q b = 2 - 1 - 1 + 3 = 3 -> total = 13.0
    energy = qubo_energy_of_bitstring(Q, offset, "11")
    assert np.isclose(energy, 13.0)
