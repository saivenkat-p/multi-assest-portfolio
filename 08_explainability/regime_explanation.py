"""
Module: 08_explainability.regime_explanation
Explains how the detected market regime influenced portfolio allocation.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RegimeExplainer:
    """
    Translates market regime states and risk parameters into human-readable insights.
    """

    def __init__(self):
        """
        Initializes the explainer with standard regime descriptions.
        """
        self.regime_descriptions = {
            "Bull": "Characterized by upward trends and lower volatility. The optimizer favors higher-yield, higher-risk assets.",
            "Bear": "Characterized by downward trends. The optimizer shifts focus to capital preservation, favoring defensive assets.",
            "High_Volatility": "Characterized by erratic market movements. The optimizer aggressively applies diversification penalties.",
            "Neutral": "Characterized by sideways movement. The optimizer balances yield and risk equally."
        }

    def explain(self, current_regime: str, risk_aversion: float) -> Dict[str, str]:
        """
        Generates a text-based explanation of the regime's impact on the model.

        Args:
            current_regime (str): The label of the current market regime.
            risk_aversion (float): The gamma parameter used in the objective function.

        Returns:
            Dict[str, str]: Structured explanations of the regime and model impact.
        """
        logger.info(f"Generating regime explanation for: {current_regime}")
        
        description = self.regime_descriptions.get(current_regime, "Unknown or custom market regime.")
        
        # Explain the impact of the risk aversion parameter
        impact = f"With a risk aversion parameter (gamma) set to {risk_aversion}, "
        if risk_aversion > 1.0:
            impact += "the model heavily penalized volatility, reflecting a cautious stance suitable for uncertain or bearish regimes."
        elif risk_aversion < 1.0:
            impact += "the model prioritized expected returns over variance, indicating an aggressive growth strategy typical of bull markets."
        else:
            impact += "the model maintained a neutral, baseline balance between expected return and variance risk."
            
        report = {
            "regime_detected": current_regime,
            "regime_characteristics": description,
            "model_impact": impact
        }
        
        logger.info("Regime explanation generated.")
        return report