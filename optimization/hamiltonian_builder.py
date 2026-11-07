"""
Module: 03_optimization.hamiltonian_builder

Converts the classical Mean-Variance portfolio optimization problem into a
Quadratic Unconstrained Binary Optimization (QUBO) formulation suitable for
quantum optimization algorithms such as QAOA, VQE, and quantum annealers.

The module discretizes continuous portfolio weights into binary variables,
constructs the corresponding QUBO matrix, and incorporates:

    • Portfolio variance
    • Expected returns
    • Budget constraint penalty

Mathematical formulation:

    Minimize:

        0.5 * wᵀΣw
        - γ μᵀw
        + λ(Σw - 1)²

where

    μ : expected returns
    Σ : covariance matrix
    γ : risk aversion
    λ : budget penalty

The resulting objective is converted into a QUBO matrix that can be consumed
directly by downstream quantum optimization modules.

This implementation preserves backward compatibility while providing
production-grade validation, logging, exception handling, numerical
robustness, and complete documentation.
"""

import logging
from typing import Tuple

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

DEFAULT_RISK_AVERSION = 1.0
DEFAULT_PENALTY_FACTOR = 1000.0
DEFAULT_BINARY_BITS = 3

MIN_BINARY_BITS = 1
FLOAT_TOLERANCE = 1e-12

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HamiltonianBuilder:
    """
    Builds the Quadratic Unconstrained Binary Optimization (QUBO) matrix
    representing the classical portfolio optimization problem.

    The generated QUBO can be supplied directly to quantum optimization
    algorithms including QAOA and VQE.

    Attributes
    ----------
    risk_aversion : float
        Risk-return trade-off coefficient.

    penalty_factor : float
        Penalty applied to violations of the portfolio budget constraint.

    binary_bits : int
        Number of binary variables used to represent each asset weight.

    max_val : int
        Maximum integer representable by the binary encoding.
    """

    def __init__(
        self,
        risk_aversion: float = DEFAULT_RISK_AVERSION,
        penalty_factor: float = DEFAULT_PENALTY_FACTOR,
        binary_bits: int = DEFAULT_BINARY_BITS,
    ) -> None:
        """
        Initialize the Hamiltonian builder.

        Args:
            risk_aversion:
                Risk-return trade-off parameter.

            penalty_factor:
                Penalty coefficient applied to the budget constraint.

            binary_bits:
                Number of binary variables used to encode one asset weight.

        Raises:
            TypeError:
                If parameter types are invalid.

            ValueError:
                If parameter values are outside valid ranges.
        """

        self._validate_parameters(
            risk_aversion,
            penalty_factor,
            binary_bits,
        )

        self.risk_aversion = float(risk_aversion)
        self.penalty_factor = float(penalty_factor)
        self.binary_bits = int(binary_bits)

        self.max_val = (2 ** self.binary_bits) - 1

        logger.info(
            "HamiltonianBuilder initialized "
            "(risk_aversion=%.4f, penalty_factor=%.4f, binary_bits=%d)",
            self.risk_aversion,
            self.penalty_factor,
            self.binary_bits,
        )

    # ======================================================================
    # Validation Helpers
    # ======================================================================

    @staticmethod
    def _validate_parameters(
        risk_aversion: float,
        penalty_factor: float,
        binary_bits: int,
    ) -> None:
        """
        Validate constructor parameters.

        Args:
            risk_aversion:
                Risk-return trade-off coefficient.

            penalty_factor:
                Budget constraint penalty coefficient.

            binary_bits:
                Number of binary encoding bits.

        Raises:
            TypeError:
                If parameter types are invalid.

            ValueError:
                If parameter values are invalid.
        """

        if not isinstance(risk_aversion, (int, float)):
            raise TypeError("risk_aversion must be numeric.")

        if not isinstance(penalty_factor, (int, float)):
            raise TypeError("penalty_factor must be numeric.")

        if not isinstance(binary_bits, int):
            raise TypeError("binary_bits must be an integer.")

        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative.")

        if penalty_factor <= 0:
            raise ValueError("penalty_factor must be positive.")

        if binary_bits < MIN_BINARY_BITS:
            raise ValueError(
                f"binary_bits must be at least {MIN_BINARY_BITS}."
            )

    @staticmethod
    def _validate_expected_returns(
        expected_returns: np.ndarray,
    ) -> None:
        """
        Validate expected returns vector.

        Args:
            expected_returns:
                One-dimensional expected return vector.

        Raises:
            TypeError:
                If input type is invalid.

            ValueError:
                If array contents are invalid.
        """

        if not isinstance(expected_returns, np.ndarray):
            raise TypeError("expected_returns must be a NumPy ndarray.")

        if expected_returns.ndim != 1:
            raise ValueError("expected_returns must be one-dimensional.")

        if expected_returns.size == 0:
            raise ValueError("expected_returns cannot be empty.")

        if not np.all(np.isfinite(expected_returns)):
            raise ValueError(
                "expected_returns contains NaN or infinite values."
            )

    @staticmethod
    def _validate_covariance_matrix(
        covariance_matrix: np.ndarray,
        expected_size: int,
    ) -> None:
        """
        Validate covariance matrix.

        Args:
            covariance_matrix:
                Covariance matrix.

            expected_size:
                Expected matrix dimension.

        Raises:
            TypeError:
                If input type is invalid.

            ValueError:
                If covariance matrix is invalid.
        """

        if not isinstance(covariance_matrix, np.ndarray):
            raise TypeError("covariance_matrix must be a NumPy ndarray.")

        if covariance_matrix.ndim != 2:
            raise ValueError("covariance_matrix must be two-dimensional.")

        rows, cols = covariance_matrix.shape

        if rows != cols:
            raise ValueError("covariance_matrix must be square.")

        if rows != expected_size:
            raise ValueError(
                "covariance_matrix dimensions do not match "
                "expected_returns."
            )

        if not np.all(np.isfinite(covariance_matrix)):
            raise ValueError(
                "covariance_matrix contains NaN or infinite values."
            )

        if not np.allclose(
            covariance_matrix,
            covariance_matrix.T,
            atol=FLOAT_TOLERANCE,
        ):
            raise ValueError("covariance_matrix must be symmetric.")

    # ======================================================================
    # Public API
    # ======================================================================

    def build_qubo(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Construct the QUBO matrix and constant energy offset.

        Args:
            expected_returns:
                Expected return vector with shape (N,).

            covariance_matrix:
                Covariance matrix with shape (N, N).

        Returns:
            Tuple[np.ndarray, float]:
                QUBO matrix and constant offset.

        Raises:
            TypeError:
                If inputs are invalid.

            ValueError:
                If input values are invalid.

            RuntimeError:
                If numerical instability is detected.
        """

        try:
            self._validate_expected_returns(expected_returns)

            self._validate_covariance_matrix(
                covariance_matrix,
                expected_returns.size,
            )

            n_assets = expected_returns.size
            total_qubits = n_assets * self.binary_bits

            logger.info(
                "Building QUBO for %d assets (%d qubits).",
                n_assets,
                total_qubits,
            )

            scale = 1.0 / self.max_val

            logger.info(
                "Binary discretization scale factor = %.8f",
                scale,
            )

            Q = np.zeros(
                (total_qubits, total_qubits),
                dtype=np.float64,
            )

            offset = float(self.penalty_factor)

            # ==============================================================
            # 1. Variance Term
            # ==============================================================

            logger.info("Constructing variance term.")

            for i in range(n_assets):
                for j in range(n_assets):
                    for k1 in range(self.binary_bits):
                        for k2 in range(self.binary_bits):

                            q_idx1 = i * self.binary_bits + k1
                            q_idx2 = j * self.binary_bits + k2

                            coeff = (
                                0.5
                                * covariance_matrix[i, j]
                                * scale
                                * scale
                                * (2 ** k1)
                                * (2 ** k2)
                            )

                            Q[q_idx1, q_idx2] += coeff

            logger.info("Variance term completed.")

            # ==============================================================
            # 2. Expected Return Term
            # ==============================================================

            logger.info("Constructing expected return term.")

            for i in range(n_assets):
                for k in range(self.binary_bits):

                    q_idx = i * self.binary_bits + k

                    coeff = (
                        -self.risk_aversion
                        * expected_returns[i]
                        * scale
                        * (2 ** k)
                    )

                    Q[q_idx, q_idx] += coeff

            logger.info("Expected return term completed.")

            # ==============================================================
            # 3. Budget Constraint Penalty
            # ==============================================================

            logger.info("Constructing budget constraint penalty.")

            for i in range(n_assets):
                for j in range(n_assets):
                    for k1 in range(self.binary_bits):
                        for k2 in range(self.binary_bits):

                            q_idx1 = i * self.binary_bits + k1
                            q_idx2 = j * self.binary_bits + k2

                            coeff_quad = (
                                self.penalty_factor
                                * scale
                                * scale
                                * (2 ** k1)
                                * (2 ** k2)
                            )

                            Q[q_idx1, q_idx2] += coeff_quad

            for i in range(n_assets):
                for k in range(self.binary_bits):

                    q_idx = i * self.binary_bits + k

                    coeff_lin = (
                        -2.0
                        * self.penalty_factor
                        * scale
                        * (2 ** k)
                    )

                    Q[q_idx, q_idx] += coeff_lin

            logger.info("Budget constraint completed.")

            # ==============================================================
            # Symmetrize Matrix
            # ==============================================================

            logger.info("Symmetrizing QUBO matrix.")

            Q = (Q + Q.T) / 2.0

            # ==============================================================
            # Floating-Point Cleanup
            # ==============================================================

            Q[np.abs(Q) < FLOAT_TOLERANCE] = 0.0

            # ==============================================================
            # Numerical Validation
            # ==============================================================

            if not np.all(np.isfinite(Q)):
                raise RuntimeError(
                    "QUBO matrix contains non-finite numerical values."
                )

            if not np.isfinite(offset):
                raise RuntimeError(
                    "Computed QUBO offset is not finite."
                )

            max_coeff = np.max(np.abs(Q))

            if max_coeff > 1e12:
                logger.warning(
                    "Large QUBO coefficients detected "
                    "(maximum absolute value = %.4e).",
                    max_coeff,
                )

            if max_coeff < FLOAT_TOLERANCE:
                logger.warning(
                    "QUBO coefficients are extremely small."
                )

            logger.info(
                "QUBO matrix successfully constructed."
            )

            return Q, offset

        except Exception as exc:
            logger.exception(
                "Failed to construct QUBO matrix: %s",
                exc,
            )
            raise

    def build_pennylane_hamiltonian(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        include_offset: bool = True,
    ):
        """
        Build and return a native PennyLane Ising Hamiltonian directly from financial inputs.

        Args:
            expected_returns: 1D array of expected asset returns.
            covariance_matrix: 2D asset return covariance matrix.
            include_offset: Whether to include constant energy offset term.

        Returns:
            PennyLane Hamiltonian representing the portfolio optimization problem.
        """
        Q, offset = self.build_qubo(expected_returns, covariance_matrix)
        from quantum_algorithms.hamiltonian import qubo_to_pennylane_hamiltonian
        return qubo_to_pennylane_hamiltonian(Q, offset, include_offset=include_offset)