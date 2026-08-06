"""
Module: 03_optimization.adaptive_weights

Adjusts optimization parameters dynamically based on detected market regimes.

This module provides adaptive optimization parameters for the ARQPO
(Adaptive Regime-Aware Quantum Portfolio Optimizer). It modifies
optimization hyperparameters such as risk aversion, asset weight bounds,
and QUBO penalty coefficients according to the detected market regime.

The implementation is designed to remain fully backward compatible while
adding production-grade validation, logging, and numerical robustness.
"""

import logging
import math
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Module Constants
# -----------------------------------------------------------------------------

DEFAULT_RISK_AVERSION = 1.0
DEFAULT_PENALTY = 1000.0

DEFAULT_REGIME = "Stable"

RISK_MULTIPLIERS: Dict[str, float] = {
    "Bull": 0.5,
    "Stable": 1.0,
    "Bear": 2.5,
    "High Volatility": 4.0,
}

HIGH_VOLATILITY_DIVISOR = 3.0
BEAR_MARKET_DIVISOR = 5.0
STRESS_PENALTY_MULTIPLIER = 2.0

# -----------------------------------------------------------------------------
# Module Logger
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RegimeAdaptiveParameters:
    """
    Modifies optimization parameters according to the detected market regime.

    The class dynamically adjusts:

    - Risk aversion
    - Asset weight bounds
    - QUBO penalty coefficients

    while preserving compatibility with downstream optimization modules.
    """

    def __init__(
        self,
        base_risk_aversion: float = DEFAULT_RISK_AVERSION,
        base_penalty: float = DEFAULT_PENALTY,
    ) -> None:
        """
        Initialize adaptive optimization parameters.

        Args:
            base_risk_aversion (float):
                Base portfolio risk-aversion coefficient.

            base_penalty (float):
                Base QUBO constraint penalty.

        Raises:
            TypeError:
                If inputs are not numeric.

            ValueError:
                If inputs are non-positive, NaN, or infinite.
        """
        self.base_risk_aversion = self._validate_positive_number(
            base_risk_aversion,
            "base_risk_aversion",
        )

        self.base_penalty = self._validate_positive_number(
            base_penalty,
            "base_penalty",
        )

        self.risk_multipliers = RISK_MULTIPLIERS.copy()

        logger.info(
            "Initialized RegimeAdaptiveParameters "
            "(risk_aversion=%s, penalty=%s)",
            self.base_risk_aversion,
            self.base_penalty,
        )

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positive_number(value: float, name: str) -> float:
        """
        Validate a positive numeric parameter.

        Args:
            value (float):
                Value to validate.

            name (str):
                Parameter name.

        Returns:
            float:
                Validated numeric value.

        Raises:
            TypeError:
                If value is not numeric.

            ValueError:
                If value is invalid.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        value = float(value)

        if math.isnan(value):
            raise ValueError(f"{name} cannot be NaN.")

        if math.isinf(value):
            raise ValueError(f"{name} cannot be infinite.")

        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

        return value

    @staticmethod
    def _validate_regime(current_regime: str) -> str:
        """
        Validate regime label.

        Args:
            current_regime (str):
                Market regime.

        Returns:
            str:
                Cleaned regime string.

        Raises:
            TypeError:
                If regime is not a string.

            ValueError:
                If regime is empty.
        """
        if not isinstance(current_regime, str):
            raise TypeError("current_regime must be a string.")

        current_regime = current_regime.strip()

        if not current_regime:
            raise ValueError("current_regime cannot be empty.")

        return current_regime

    @staticmethod
    def _validate_asset_count(n_assets: int) -> int:
        """
        Validate asset count.

        Args:
            n_assets (int):
                Number of assets.

        Returns:
            int:
                Validated asset count.

        Raises:
            TypeError:
                If not integer.

            ValueError:
                If <= 0.
        """
        if not isinstance(n_assets, int):
            raise TypeError("n_assets must be an integer.")

        if n_assets <= 0:
            raise ValueError("n_assets must be greater than zero.")

        return n_assets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_adjusted_risk_aversion(self, current_regime: str) -> float:
        """
        Calculate regime-adjusted risk aversion.

        Args:
            current_regime (str):
                Active market regime.

        Returns:
            float:
                Adjusted risk-aversion coefficient.

        Raises:
            TypeError:
                If regime type is invalid.

            ValueError:
                If regime is empty.
        """
        regime = self._validate_regime(current_regime)

        if regime not in self.risk_multipliers:
            logger.warning(
                "Unknown regime '%s'. Using default multiplier.",
                regime,
            )

        multiplier = self.risk_multipliers.get(
            regime,
            self.risk_multipliers[DEFAULT_REGIME],
        )

        adjusted_gamma = self.base_risk_aversion * multiplier

        logger.info(
            "Regime '%s': adjusted risk aversion = %.4f",
            regime,
            adjusted_gamma,
        )

        return float(adjusted_gamma)

    def get_adjusted_bounds(
        self,
        current_regime: str,
        n_assets: int,
    ) -> Tuple[Tuple[float, float], ...]:
        """
        Calculate adaptive portfolio weight bounds.

        Args:
            current_regime (str):
                Active market regime.

            n_assets (int):
                Number of portfolio assets.

        Returns:
            Tuple[Tuple[float, float], ...]:
                Weight bounds for every asset.

        Raises:
            TypeError:
                If inputs are invalid.

            ValueError:
                If inputs are invalid.
        """
        regime = self._validate_regime(current_regime)
        n_assets = self._validate_asset_count(n_assets)

        max_weight = 1.0

        if regime == "High Volatility":
            max_weight = min(
                1.0,
                HIGH_VOLATILITY_DIVISOR / n_assets,
            )

            logger.info(
                "High Volatility detected. "
                "Maximum asset weight = %.4f",
                max_weight,
            )

        elif regime == "Bear":
            max_weight = min(
                1.0,
                BEAR_MARKET_DIVISOR / n_assets,
            )

            logger.info(
                "Bear market detected. "
                "Maximum asset weight = %.4f",
                max_weight,
            )

        return tuple((0.0, max_weight) for _ in range(n_assets))

    def get_adjusted_penalty(self, current_regime: str) -> float:
        """
        Calculate adaptive QUBO penalty coefficient.

        Args:
            current_regime (str):
                Active market regime.

        Returns:
            float:
                Regime-adjusted penalty.

        Raises:
            TypeError:
                If regime type is invalid.

            ValueError:
                If regime is empty.
        """
        regime = self._validate_regime(current_regime)

        penalty = self.base_penalty

        if regime in ("High Volatility", "Bear"):
            penalty *= STRESS_PENALTY_MULTIPLIER

        logger.info(
            "Regime '%s': adjusted penalty = %.4f",
            regime,
            penalty,
        )

        return float(penalty)