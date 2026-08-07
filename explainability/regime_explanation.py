"""
ARQPO - Market Regime Explainability Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Generates human-readable explanations describing how the
    detected market regime influences portfolio optimization,
    including the optimizer's risk preference and expected
    portfolio behavior.

Workflow Position:
    08_explainability/regime_explanation.py

        Market Regime
              │
              ▼
        Risk Preference
              │
        Optimization Objective
              │
        Portfolio Behaviour
              │
              ▼
        Human-Readable Explanation

Mathematical Objective:
    Translate detected market regimes and adaptive risk-aversion
    parameters into interpretable explanations suitable for
    researchers, portfolio managers, and dashboard users.

Design Philosophy:
    - Defensive Programming
    - Numerical Stability
    - Deterministic Behaviour
    - Production Logging
    - Backward Compatibility
    - Research-Grade Implementation
"""

import logging
import time
from typing import Dict

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE: float = 1e-10

DEFAULT_REGIME_DESCRIPTIONS = {
    "Bull": (
        "Characterized by upward trends and relatively low "
        "market volatility. The optimizer favors higher-return "
        "assets while accepting additional risk."
    ),
    "Bear": (
        "Characterized by prolonged downward market trends. "
        "The optimizer prioritizes capital preservation and "
        "allocates more conservatively."
    ),
    "High_Volatility": (
        "Characterized by significant market uncertainty and "
        "rapid price fluctuations. Diversification penalties "
        "and risk controls are strengthened."
    ),
    "Neutral": (
        "Characterized by balanced market conditions. The "
        "optimizer maintains an even trade-off between "
        "expected return and portfolio risk."
    ),
}

# =============================================================================
# Logger Configuration
# =============================================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

# =============================================================================
# Public Classes
# =============================================================================


class RegimeExplainer:
    """
    Generates human-readable explanations describing how
    market regimes influence portfolio optimization.
    """

    def __init__(self):
        """
        Initializes the market regime explainer.

        Raises:
            RuntimeError:
                If the explainer cannot be initialized.

        Notes:
            The constructor initializes the standardized market
            regime descriptions used throughout the explainability
            module while preserving backward compatibility.
        """
        logger.info("Initializing RegimeExplainer.")

        try:
            if not isinstance(
                DEFAULT_REGIME_DESCRIPTIONS,
                dict,
            ):
                logger.error(
                    "DEFAULT_REGIME_DESCRIPTIONS must be a dictionary."
                )
                raise TypeError(
                    "DEFAULT_REGIME_DESCRIPTIONS must be a dictionary."
                )

            if len(DEFAULT_REGIME_DESCRIPTIONS) == 0:
                logger.error(
                    "No market regime descriptions were configured."
                )
                raise ValueError(
                    "DEFAULT_REGIME_DESCRIPTIONS cannot be empty."
                )

            self.regime_descriptions = (
                DEFAULT_REGIME_DESCRIPTIONS.copy()
            )

            logger.info(
                "RegimeExplainer initialized successfully "
                "with %d supported market regimes.",
                len(self.regime_descriptions),
            )

        except Exception as exc:
            logger.exception(
                "Failed to initialize RegimeExplainer."
            )

            raise RuntimeError(
                "Unable to initialize RegimeExplainer."
            ) from exc

    def _validate_inputs(
        self,
        current_regime: str,
        risk_aversion: float,
    ) -> float:
        """
        Validates the inputs for regime explanation.

        Args:
            current_regime (str):
                Detected market regime.

            risk_aversion (float):
                Risk-aversion (gamma) parameter used during
                portfolio optimization.

        Returns:
            float:
                Validated risk-aversion parameter.

        Raises:
            TypeError:
                If an input has an invalid type.

            ValueError:
                If an input contains an invalid value.
        """
        if not isinstance(current_regime, str):
            logger.error(
                "current_regime must be a string."
            )
            raise TypeError(
                "current_regime must be a string."
            )

        current_regime = current_regime.strip()

        if not current_regime:
            logger.error(
                "current_regime cannot be empty."
            )
            raise ValueError(
                "current_regime cannot be empty."
            )

        if not isinstance(
            risk_aversion,
            (int, float, np.floating),
        ):
            logger.error(
                "risk_aversion must be numeric."
            )
            raise TypeError(
                "risk_aversion must be numeric."
            )

        risk_aversion = float(risk_aversion)

        if not np.isfinite(risk_aversion):
            logger.error(
                "risk_aversion contains NaN or infinite values."
            )
            raise ValueError(
                "risk_aversion must be finite."
            )

        return risk_aversion

    def _cleanup_scalar(
        self,
        value: float,
    ) -> float:
        """
        Removes floating-point numerical artifacts.

        Args:
            value (float):
                Input scalar.

        Returns:
            float:
                Cleaned scalar.
        """
        if np.isclose(
            value,
            0.0,
            atol=FLOAT_TOLERANCE,
        ):
            return 0.0

        return float(value)

    def _build_model_impact(
        self,
        risk_aversion: float,
    ) -> str:
        """
        Builds a human-readable explanation describing
        the optimizer's risk preference.

        Args:
            risk_aversion (float):
                Gamma parameter.

        Returns:
            str:
                Model impact explanation.
        """
        impact = (
            "With a risk aversion parameter "
            f"(gamma) of {risk_aversion:.4f}, "
        )

        if risk_aversion > (
            1.0 + FLOAT_TOLERANCE
        ):
            impact += (
                "the optimizer strongly penalized "
                "portfolio volatility, emphasizing "
                "capital preservation."
            )

        elif risk_aversion < (
            1.0 - FLOAT_TOLERANCE
        ):
            impact += (
                "the optimizer emphasized expected "
                "returns over variance, supporting "
                "a growth-oriented allocation."
            )

        else:
            impact += (
                "the optimizer maintained a balanced "
                "trade-off between expected return "
                "and portfolio risk."
            )

        return impact

    def _build_report(
        self,
        current_regime: str,
        description: str,
        impact: str,
    ) -> Dict[str, str]:
        """
        Builds the standardized explainability report.

        Args:
            current_regime (str):
                Detected market regime.

            description (str):
                Human-readable regime description.

            impact (str):
                Explanation of optimization behaviour.

        Returns:
            Dict[str, str]:
                Standardized explainability report.

        Notes:
            Report keys intentionally remain unchanged
            to preserve backward compatibility.
        """
        return {
            "regime_detected": current_regime,
            "regime_characteristics": description,
            "model_impact": impact,
        }

    def explain(
        self,
        current_regime: str,
        risk_aversion: float,
    ) -> Dict[str, str]:
        """
        Generates a human-readable explanation for the given market regime
        and risk aversion parameter.

        Args:
            current_regime (str):
                Detected market regime.
            risk_aversion (float):
                Risk-aversion (gamma) parameter.

        Returns:
            Dict[str, str]:
                Standardized explainability report.
        """
        validated_risk = self._validate_inputs(
            current_regime, risk_aversion
        )
        cleaned_risk = self._cleanup_scalar(validated_risk)

        regime_key = current_regime.strip()
        description = self.regime_descriptions.get(
            regime_key,
            (
                "Characterized by custom or unclassified market conditions. "
                "The optimizer applies adaptive risk controls dynamically."
            ),
        )

        impact = self._build_model_impact(cleaned_risk)
        return self._build_report(regime_key, description, impact)


# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":

    logger.info(
        "Starting RegimeExplainer runtime tests."
    )

    explainer = RegimeExplainer()

    print("\n" + "=" * 80)
    print("RegimeExplainer Runtime Tests")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1 : Happy Path
    # -------------------------------------------------------------------------
    print("\n[1] Happy Path")

    report = explainer.explain(
        current_regime="Bull",
        risk_aversion=0.80,
    )

    print(report)

    # -------------------------------------------------------------------------
    # Test 2 : Medium Input
    # -------------------------------------------------------------------------
    print("\n[2] Medium Input")

    regimes = [
        "Bull",
        "Bear",
        "Neutral",
        "High_Volatility",
    ]

    for regime in regimes:

        report = explainer.explain(
            current_regime=regime,
            risk_aversion=1.00,
        )

        print(
            regime,
            "->",
            report["regime_detected"],
        )

    # -------------------------------------------------------------------------
    # Test 3 : Large Input
    # -------------------------------------------------------------------------
    print("\n[3] Large Input")

    for _ in range(1000):

        explainer.explain(
            current_regime="Bull",
            risk_aversion=0.90,
        )

    print("Large input test passed.")

    # -------------------------------------------------------------------------
    # Test 4 : Invalid Type
    # -------------------------------------------------------------------------
    print("\n[4] Invalid Type")

    invalid_cases = [
        (123, 1.0),
        ("Bull", "high"),
        (None, 1.0),
    ]

    for regime, gamma in invalid_cases:

        try:

            explainer.explain(
                regime,
                gamma,
            )

        except Exception as exc:

            print(
                type(exc).__name__,
                exc,
            )

    # -------------------------------------------------------------------------
    # Test 5 : Invalid Values
    # -------------------------------------------------------------------------
    print("\n[5] Invalid Values")

    invalid_values = [
        ("", 1.0),
        ("Bull", np.nan),
        ("Bull", np.inf),
        ("Bull", -np.inf),
    ]

    for regime, gamma in invalid_values:

        try:

            explainer.explain(
                regime,
                gamma,
            )

        except Exception as exc:

            print(
                type(exc).__name__,
                exc,
            )

    # -------------------------------------------------------------------------
    # Test 6 : Edge Case
    # -------------------------------------------------------------------------
    print("\n[6] Edge Case")

    edge_cases = [
        ("Bull", 0.0),
        ("Bull", 1.0),
        ("Bull", 10.0),
        ("Unknown_Regime", 1.0),
    ]

    for regime, gamma in edge_cases:

        report = explainer.explain(
            regime,
            gamma,
        )

        print(report)

    # -------------------------------------------------------------------------
    # Test 7 : Determinism
    # -------------------------------------------------------------------------
    print("\n[7] Determinism")

    first = explainer.explain(
        "Bear",
        1.20,
    )

    second = explainer.explain(
        "Bear",
        1.20,
    )

    print(
        "Deterministic:",
        first == second,
    )

    # -------------------------------------------------------------------------
    # Test 8 : Stress Test
    # -------------------------------------------------------------------------
    print("\n[8] Stress Test")

    stress_regimes = [
        "Bull",
        "Bear",
        "Neutral",
        "High_Volatility",
    ]

    for _ in range(5000):

        for regime in stress_regimes:

            explainer.explain(
                regime,
                1.0,
            )

    print("Stress test passed.")

    # -------------------------------------------------------------------------
    # Test 9 : Numerical Robustness
    # -------------------------------------------------------------------------
    print("\n[9] Numerical Robustness")

    numerical_cases = [
        0.999999999999,
        1.000000000001,
        1.000000000000,
    ]

    for gamma in numerical_cases:

        report = explainer.explain(
            "Neutral",
            gamma,
        )

        print(report["model_impact"])

    # -------------------------------------------------------------------------
    # Test 10 : Integration Verification
    # -------------------------------------------------------------------------
    print("\n[10] Integration Verification")

    report = explainer.explain(
        "High_Volatility",
        1.50,
    )

    required_keys = {
        "regime_detected",
        "regime_characteristics",
        "model_impact",
    }

    integration_pass = required_keys.issubset(
        report.keys()
    )

    print(
        "Integration Passed:",
        integration_pass,
    )

    print("\n" + "=" * 80)
    print("All runtime tests completed successfully.")
    print("=" * 80)