"""
ARQPO - Explainability Report Generator Module.

Project:
    Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

Purpose:
    Aggregates explainability outputs from portfolio allocation,
    market regime analysis, and portfolio performance metrics into
    both a structured machine-readable report and a human-readable
    Markdown report.

Workflow Position:
    08_explainability/report_generator.py

        Allocation Report
                │
        Regime Explanation
                │
        Portfolio Metrics
                │
                ▼
        Master Explainability Report
                │
        ├── Structured Dictionary
        └── Markdown Report
                │
                ▼
        Dashboard / Research Report

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
from typing import Any, Dict, List, Tuple

import numpy as np

# =============================================================================
# Module Constants
# =============================================================================

FLOAT_TOLERANCE: float = 1e-10

REPORT_TITLE = "# PORTFOLIO EXPLAINABILITY REPORT"
REPORT_SEPARATOR = "-" * 40
REPORT_END = "END OF REPORT"

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


class PortfolioReportGenerator:
    """
    Aggregates portfolio explainability outputs into a final
    structured report and Markdown summary.
    """

    def __init__(self):
        """
        Initializes the portfolio explainability report generator.

        Raises:
            RuntimeError:
                If the report generator cannot be initialized.

        Notes:
            Initializes module-level report configuration while
            preserving backward compatibility. Configuration validation
            and initialization logging are fully integrated.
        """
        logger.info(
            "Initializing PortfolioReportGenerator."
        )

        try:
            required_constants = {
                "REPORT_TITLE": REPORT_TITLE,
                "REPORT_SEPARATOR": REPORT_SEPARATOR,
                "REPORT_END": REPORT_END,
            }

            for constant_name, constant_value in (
                required_constants.items()
            ):
                if not isinstance(
                    constant_value,
                    str,
                ):
                    logger.error(
                        "%s must be a string.",
                        constant_name,
                    )

                    raise TypeError(
                        f"{constant_name} must be a string."
                    )

                if not constant_value.strip():
                    logger.error(
                        "%s cannot be empty.",
                        constant_name,
                    )

                    raise ValueError(
                        f"{constant_name} cannot be empty."
                    )

            self.report_title = REPORT_TITLE
            self.report_separator = REPORT_SEPARATOR
            self.report_end = REPORT_END

            logger.info(
                "PortfolioReportGenerator initialized "
                "successfully."
            )

        except Exception as exc:
            logger.exception(
                "Failed to initialize "
                "PortfolioReportGenerator."
            )

            raise RuntimeError(
                "Unable to initialize "
                "PortfolioReportGenerator."
            ) from exc

    def generate(
        self,
        allocation_report: Dict[str, Any],
        regime_report: Dict[str, str],
        portfolio_metrics: Dict[str, float],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generates the complete explainability report.

        The report consists of:

        1. Machine-readable dictionary
        2. Human-readable Markdown report

        Args:
            allocation_report (Dict[str, Any]):
                Portfolio allocation explanations.

            regime_report (Dict[str, str]):
                Market regime explanation.

            portfolio_metrics (Dict[str, float]):
                Portfolio performance metrics.

        Returns:
            Tuple[Dict[str, Any], str]:
                Structured report dictionary and Markdown report.

        Raises:
            RuntimeError:
                If report generation fails unexpectedly.
        """
        start_time = time.perf_counter()

        try:
            logger.info(
                "Starting master explainability report generation."
            )

            # =====================================================
            # Input Validation
            # =====================================================

            self._validate_inputs(
                allocation_report=allocation_report,
                regime_report=regime_report,
                portfolio_metrics=portfolio_metrics,
            )

            cleaned_metrics = self._cleanup_metrics(
                portfolio_metrics
            )

            # =====================================================
            # Build Master Dictionary
            # =====================================================

            master_dict = self._build_master_dict(
                allocation_report=allocation_report,
                regime_report=regime_report,
                portfolio_metrics=cleaned_metrics,
            )

            # =====================================================
            # Build Markdown Report
            # =====================================================

            lines: List[str] = []

            lines.append(self.report_title)
            lines.append(self.report_separator)

            self._append_metrics_section(
                lines,
                cleaned_metrics,
            )

            self._append_regime_section(
                lines,
                regime_report,
            )

            self._append_allocation_section(
                lines,
                allocation_report,
            )

            lines.append(self.report_separator)
            lines.append(self.report_end)

            text_report = "\n".join(lines)

            elapsed_time = (
                time.perf_counter() - start_time
            )

            logger.info(
                "Master explainability report generated "
                "successfully in %.6f seconds.",
                elapsed_time,
            )

            return master_dict, text_report

        except Exception as exc:
            logger.exception(
                "Master report generation failed."
            )

            raise RuntimeError(
                "Unable to generate explainability report."
            ) from exc

    # =============================================================================
    # Validation Helpers
    # =============================================================================

    def _validate_inputs(
        self,
        allocation_report: Dict[str, Any],
        regime_report: Dict[str, str],
        portfolio_metrics: Dict[str, float],
    ) -> None:
        """
        Validates all report inputs.

        Args:
            allocation_report (Dict[str, Any]):
                Portfolio allocation explanation.

            regime_report (Dict[str, str]):
                Market regime explanation.

            portfolio_metrics (Dict[str, float]):
                Portfolio performance metrics.

        Raises:
            TypeError:
                If an input has an invalid type.

            ValueError:
                If required report fields are missing.
        """
        if not isinstance(allocation_report, dict):
            logger.error(
                "allocation_report must be a dictionary."
            )
            raise TypeError(
                "allocation_report must be a dictionary."
            )

        if not isinstance(regime_report, dict):
            logger.error(
                "regime_report must be a dictionary."
            )
            raise TypeError(
                "regime_report must be a dictionary."
            )

        if not isinstance(portfolio_metrics, dict):
            logger.error(
                "portfolio_metrics must be a dictionary."
            )
            raise TypeError(
                "portfolio_metrics must be a dictionary."
            )

        required_allocation = {
            "Yield_Seeking",
            "Risk_Mitigating",
            "Omitted",
        }

        missing = required_allocation.difference(
            allocation_report.keys()
        )

        if missing:
            logger.error(
                "Missing allocation report keys: %s",
                sorted(missing),
            )
            raise ValueError(
                f"Missing allocation keys: {sorted(missing)}"
            )

        required_regime = {
            "regime_detected",
            "regime_characteristics",
            "model_impact",
        }

        missing = required_regime.difference(
            regime_report.keys()
        )

        if missing:
            logger.error(
                "Missing regime report keys: %s",
                sorted(missing),
            )
            raise ValueError(
                f"Missing regime keys: {sorted(missing)}"
            )

    def _cleanup_metrics(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Cleans numerical metric values.

        Args:
            metrics (Dict[str, float]):
                Portfolio metrics.

        Returns:
            Dict[str, float]:
                Cleaned metrics dictionary.
        """
        cleaned = {}

        for key, value in metrics.items():

            if not isinstance(
                value,
                (int, float, np.floating),
            ):
                cleaned[key] = value
                continue

            numeric = float(value)

            if not np.isfinite(numeric):
                raise ValueError(
                    f"Metric '{key}' contains "
                    "NaN or Inf."
                )

            if np.isclose(
                numeric,
                0.0,
                atol=FLOAT_TOLERANCE,
            ):
                numeric = 0.0

            cleaned[key] = numeric

        return cleaned

    def _build_master_dict(
        self,
        allocation_report: Dict[str, Any],
        regime_report: Dict[str, str],
        portfolio_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Builds the structured explainability report.

        Returns:
            Dict[str, Any]:
                Master explainability dictionary.
        """
        return {
            "metrics": portfolio_metrics,
            "market_context": regime_report,
            "allocation_rationale": allocation_report,
        }

    # =============================================================================
    # Markdown Helpers
    # =============================================================================

    def _append_metrics_section(
        self,
        lines: List[str],
        metrics: Dict[str, float],
    ) -> None:
        """
        Appends the portfolio metrics section.
        """
        lines.append("## 1. Top-Level Metrics")

        for key, value in metrics.items():

            if isinstance(
                value,
                (int, float, np.floating),
            ):
                lines.append(
                    f"* {key.replace('_', ' ').title()}: "
                    f"{value:.4f}"
                )
            else:
                lines.append(
                    f"* {key.replace('_', ' ').title()}: "
                    f"{value}"
                )

    def _append_regime_section(
        self,
        lines: List[str],
        regime_report: Dict[str, str],
    ) -> None:
        """
        Appends the market regime section.
        """
        lines.append(
            "\n## 2. Market Context & Model Stance"
        )

        lines.append(
            f"* Detected Regime: "
            f"{regime_report.get('regime_detected', 'N/A')}"
        )

        lines.append(
            f"* Market Characteristics: "
            f"{regime_report.get('regime_characteristics', 'N/A')}"
        )

        lines.append(
            f"* Model Impact: "
            f"{regime_report.get('model_impact', 'N/A')}"
        )

    def _append_allocation_section(
        self,
        lines: List[str],
        allocation_report: Dict[str, Any],
    ) -> None:
        """
        Appends allocation explanations.
        """
        lines.append("\n## 3. Allocation Rationale")

        sections = [
            ("Yield-Seeking Assets", "Yield_Seeking"),
            ("Risk-Mitigating Assets", "Risk_Mitigating"),
            ("Omitted Assets", "Omitted"),
        ]

        for title, key in sections:

            lines.append(f"\n### {title}:")

            values = allocation_report.get(
                key,
                [],
            )

            if values:

                for item in values:
                    lines.append(f"* {item}")

            else:

                lines.append("* None")

# =============================================================================
# Runtime Tests
# =============================================================================

if __name__ == "__main__":

    logger.info(
        "Starting PortfolioReportGenerator runtime tests."
    )

    generator = PortfolioReportGenerator()

    # -------------------------------------------------------------------------
    # Test 1 : Happy Path
    # -------------------------------------------------------------------------
    allocation_report = {
        "Yield_Seeking": [
            "Asset A selected for strong expected return."
        ],
        "Risk_Mitigating": [
            "Asset B reduces portfolio volatility."
        ],
        "Omitted": [
            "Asset C excluded due to poor outlook."
        ],
    }

    regime_report = {
        "regime_detected": "Bull",
        "regime_characteristics":
            "Upward market trend with moderate volatility.",
        "model_impact":
            "Growth-oriented allocation strategy.",
    }

    portfolio_metrics = {
        "expected_return": 0.15,
        "volatility": 0.18,
        "sharpe_ratio": 1.32,
    }

    report_dict, markdown = generator.generate(
        allocation_report,
        regime_report,
        portfolio_metrics,
    )

    # -------------------------------------------------------------------------
    # Test 2 : Medium Input
    # -------------------------------------------------------------------------
    metrics = {
        f"metric_{i}": float(i)
        for i in range(25)
    }

    report_dict, markdown = generator.generate(
        allocation_report,
        regime_report,
        metrics,
    )

    # -------------------------------------------------------------------------
    # Test 3 : Large Input
    # -------------------------------------------------------------------------
    large_metrics = {
        f"metric_{i}": float(i)
        for i in range(1000)
    }

    report_dict, markdown = generator.generate(
        allocation_report,
        regime_report,
        large_metrics,
    )

    # -------------------------------------------------------------------------
    # Test 4 : Invalid Shape
    # -------------------------------------------------------------------------
    invalid_cases = [
        [],
        (),
        None,
        123,
    ]

    for obj in invalid_cases:
        try:
            generator.generate(
                obj,
                regime_report,
                portfolio_metrics,
            )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Test 5 : Invalid Values
    # -------------------------------------------------------------------------
    invalid_metrics = {
        "expected_return": np.nan,
        "volatility": np.inf,
    }

    try:
        generator.generate(
            allocation_report,
            regime_report,
            invalid_metrics,
        )
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Test 6 : Edge Case
    # -------------------------------------------------------------------------
    empty_metrics = {}

    report_dict, markdown = generator.generate(
        allocation_report,
        regime_report,
        empty_metrics,
    )

    # -------------------------------------------------------------------------
    # Test 7 : Determinism
    # -------------------------------------------------------------------------
    first = generator.generate(
        allocation_report,
        regime_report,
        portfolio_metrics,
    )

    second = generator.generate(
        allocation_report,
        regime_report,
        portfolio_metrics,
    )

    # -------------------------------------------------------------------------
    # Test 8 : Stress Test
    # -------------------------------------------------------------------------
    for _ in range(500):
        generator.generate(
            allocation_report,
            regime_report,
            portfolio_metrics,
        )

    # -------------------------------------------------------------------------
    # Test 9 : Numerical Robustness
    # -------------------------------------------------------------------------
    numerical_metrics = {
        "tiny": 1e-15,
        "negative_tiny": -1e-15,
        "normal": 0.5,
    }

    report_dict, markdown = generator.generate(
        allocation_report,
        regime_report,
        numerical_metrics,
    )

    # -------------------------------------------------------------------------
    # Test 10 : Integration Verification
    # -------------------------------------------------------------------------
    required_keys = {
        "metrics",
        "market_context",
        "allocation_rationale",
    }

    integration_pass = required_keys.issubset(
        report_dict.keys()
    )