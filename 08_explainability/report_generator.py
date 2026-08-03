"""
Module: 08_explainability.report_generator
Compiles individual explainability metrics into a master portfolio report.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class PortfolioReportGenerator:
    """
    Aggregates sub-reports into a final comprehensive explainability report.
    """

    def __init__(self):
        """
        Initializes the report generator.
        """
        pass

    def generate(
        self, 
        allocation_report: Dict[str, Any], 
        regime_report: Dict[str, str], 
        portfolio_metrics: Dict[str, float]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generates the combined JSON and formatted text reports.

        Args:
            allocation_report (Dict): Rationale for asset weights.
            regime_report (Dict): Market regime insights.
            portfolio_metrics (Dict): High-level performance numbers.

        Returns:
            Tuple[Dict, str]: The raw combined dictionary and the formatted Markdown string.
        """
        logger.info("Compiling master explainability report.")
        
        # 1. Compile the master dictionary
        master_dict = {
            "metrics": portfolio_metrics,
            "market_context": regime_report,
            "allocation_rationale": allocation_report
        }
        
        # 2. Build the formatted text report
        lines = []
        lines.append("# PORTFOLIO EXPLAINABILITY REPORT")
        lines.append("-" * 40)
        
        # Metrics Section
        lines.append("## 1. Top-Level Metrics")
        for key, value in portfolio_metrics.items():
            lines.append(f"* {key.replace('_', ' ').title()}: {value:.4f}")
            
        # Market Regime Section
        lines.append("\n## 2. Market Context & Model Stance")
        lines.append(f"* Detected Regime: {regime_report.get('regime_detected', 'N/A')}")
        lines.append(f"* Market Characteristics: {regime_report.get('regime_characteristics', 'N/A')}")
        lines.append(f"* Model Impact: {regime_report.get('model_impact', 'N/A')}")
        
        # Allocation Section
        lines.append("\n## 3. Allocation Rationale")
        
        lines.append("### Yield-Seeking Assets:")
        for reason in allocation_report.get("Yield_Seeking", []):
            lines.append(f"* {reason}")
        if not allocation_report.get("Yield_Seeking"):
            lines.append("* None")
            
        lines.append("\n### Risk-Mitigating Assets:")
        for reason in allocation_report.get("Risk_Mitigating", []):
            lines.append(f"* {reason}")
        if not allocation_report.get("Risk_Mitigating"):
            lines.append("* None")
            
        lines.append("\n### Omitted Assets:")
        for reason in allocation_report.get("Omitted", []):
            lines.append(f"* {reason}")
        if not allocation_report.get("Omitted"):
            lines.append("* None")
            
        lines.append("-" * 40)
        lines.append("END OF REPORT")
        
        text_report = "\n".join(lines)
        
        logger.info("Master report compilation successful.")
        return master_dict, text_report