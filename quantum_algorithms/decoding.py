"""
Module: quantum_algorithms.decoding

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Description:
    Decodes measurement bitstrings from quantum circuit executions into
    valid financial portfolio weights.

Encoding Details:
    Each asset i in {0, ..., N-1} is represented by B binary variables (bits).
    Total qubits / bitstring length M = N * B.
    Asset i uses bit positions [i*B, (i+1)*B).
    Integer value:
        V_i = sum_{k=0}^{B-1} 2^k * bit_{i, k}
    Raw discrete weight:
        w_i = V_i / (2^B - 1)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FLOAT_TOLERANCE = 1e-10


def validate_encoding_dimensions(
    n_assets: int,
    binary_bits: int,
    total_qubits: int,
) -> None:
    """
    Ensure problem dimensions match qubit count.

    Args:
        n_assets: Number of portfolio assets.
        binary_bits: Bits allocated per asset.
        total_qubits: Total number of qubits in the circuit.

    Raises:
        ValueError: If parameters are non-positive or dimensions do not match.
    """
    if not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError(f"n_assets must be a positive integer, got {n_assets}.")

    if not isinstance(binary_bits, int) or binary_bits <= 0:
        raise ValueError(f"binary_bits must be a positive integer, got {binary_bits}.")

    expected_qubits = n_assets * binary_bits
    if total_qubits != expected_qubits:
        raise ValueError(
            f"Encoding dimension mismatch: {n_assets} assets * {binary_bits} bits "
            f"= {expected_qubits} qubits != {total_qubits} total qubits."
        )


def validate_bitstring(bitstring: str, expected_length: int) -> str:
    """
    Validate a measurement bitstring.

    Args:
        bitstring: String representation of measured bits ('0101...').
        expected_length: Expected character length.

    Returns:
        Clean validated bitstring.

    Raises:
        TypeError: If bitstring is not a string.
        ValueError: If length does not match or invalid characters exist.
    """
    if not isinstance(bitstring, str):
        raise TypeError(f"bitstring must be a string, got {type(bitstring).__name__}.")

    bitstring = bitstring.strip()
    if len(bitstring) != expected_length:
        raise ValueError(
            f"Bitstring length {len(bitstring)} does not match expected {expected_length}."
        )

    if not set(bitstring).issubset({"0", "1"}):
        raise ValueError(f"Bitstring must contain only '0' and '1', got '{bitstring}'.")

    return bitstring


def decode_bitstring_to_weights(
    bitstring: str,
    n_assets: int,
    binary_bits: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Decode a measured computational basis bitstring into asset portfolio weights.

    Args:
        bitstring: Canonical bitstring ('b_0 b_1 ... b_{M-1}').
        n_assets: Number of assets.
        binary_bits: Number of bits per asset.
        normalize: If True, normalize weights to sum exactly to 1.0 (budget constraint).

    Returns:
        NumPy array of shape (n_assets,) containing portfolio weights.
    """
    total_qubits = len(bitstring)
    validate_encoding_dimensions(n_assets, binary_bits, total_qubits)
    valid_bitstring = validate_bitstring(bitstring, total_qubits)

    max_integer = (2 ** binary_bits) - 1
    raw_weights = np.zeros(n_assets, dtype=np.float64)

    for asset_idx in range(n_assets):
        start = asset_idx * binary_bits
        end = start + binary_bits
        chunk = valid_bitstring[start:end]

        # In big-endian integer reading:
        # e.g., '10' with 2 bits -> integer 2
        # where chunk[k] has weight 2^(B - 1 - k)
        int_val = int(chunk, 2)
        raw_weights[asset_idx] = int_val / max_integer

    raw_weights[np.abs(raw_weights) < FLOAT_TOLERANCE] = 0.0

    if not normalize:
        return raw_weights

    # Normalize to satisfy sum(w) == 1.0
    total_weight = float(np.sum(raw_weights))
    if total_weight > FLOAT_TOLERANCE:
        weights = raw_weights / total_weight
    else:
        # If all bits are zero, fall back to uniform allocation
        logger.warning("All measured bits were zero. Falling back to uniform 1/N weights.")
        weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)

    return weights


def qubo_energy_of_bitstring(
    qubo: np.ndarray,
    offset: float,
    bitstring: str,
) -> float:
    """
    Compute classical QUBO energy for a candidate bitstring.

    Args:
        qubo: Symmetric QUBO matrix.
        offset: Scalar offset.
        bitstring: Binary string.

    Returns:
        Scalar energy value.
    """
    b = np.array([int(c) for c in bitstring], dtype=float)
    return float(b.T @ qubo @ b + offset)
