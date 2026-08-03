"""
Module: 08_explainability.tradeoff_analysis
Analyzes and explains the risk-return tradeoff between two portfolios.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TradeoffAnalyzer:
    """
    Quantifies and explains the marginal risk/return differences between allocations.
    """

    def __init__(self):
        """
        Initializes the tradeoff analyzer.
        """
        pass

    def _calculate_metrics(self, weights: np.ndarray, returns: np.ndarray, cov: np.ndarray) -> tuple:
        """
        Calculates expected return and volatility for a given weight vector.
        """
        expected_return = float(np.dot(weights, returns))
        variance = float(weights.T @ cov @ weights)
        volatility = np.sqrt(max(variance, 0.0))  # Ensure no negative variance issues
        return expected_return, volatility

    def analyze(
        self, 
        weights_primary: np.ndarray, 
        weights_alternative: np.ndarray, 
        expected_returns: np.ndarray, 
        covariance_matrix: np.ndarray,
        primary_name: str = "Primary Portfolio",
        alternative_name: str = "Alternative Portfolio"
    ) -> Dict[str, Any]:
        """
        Generates the tradeoff explanation report.
        """
        logger.info(f"Analyzing tradeoff between {primary_name} and {alternative_name}.")
        
        # Calculate base metrics
        ret_pri, vol_pri = self._calculate_metrics(weights_primary, expected_returns, covariance_matrix)
        ret_alt, vol_alt = self._calculate_metrics(weights_alternative, expected_returns, covariance_matrix)
        
        # Calculate differences
        delta_return = ret_alt - ret_pri
        delta_volatility = vol_alt - vol_pri
        
        # Calculate ratio safely
        if abs(delta_volatility) > 1e-8:
            tradeoff_ratio = delta_return / delta_volatility
        else:
            tradeoff_ratio = 0.0
            
        # Build explanation string
        explanation = ""
        if abs(delta_return) < 1e-6 and abs(delta_volatility) < 1e-6:
            explanation = f"Both portfolios have identical risk and return profiles."
        elif delta_return > 0 and delta_volatility > 0:
            explanation = f"{alternative_name} is more aggressive. It yields {delta_return:.2%} more return but requires taking on {delta_volatility:.2%} more volatility. Tradeoff Ratio: {tradeoff_ratio:.4f} units of return per unit of added risk."
        elif delta_return < 0 and delta_volatility < 0:
            explanation = f"{alternative_name} is more conservative. It sacrifices {abs(delta_return):.2%} return to save {abs(delta_volatility):.2%} in volatility. Tradeoff Ratio: {tradeoff_ratio:.4f} units of return lost per unit of risk saved."
        elif delta_return > 0 and delta_volatility <= 0:
            explanation = f"{alternative_name} is strictly superior. It gains {delta_return:.2%} return while reducing or maintaining volatility by {abs(delta_volatility):.2%}."
        elif delta_return <= 0 and delta_volatility > 0:
            explanation = f"{alternative_name} is strictly inferior. It loses {abs(delta_return):.2%} return while increasing volatility by {delta_volatility:.2%}."

        report = {
            primary_name: {"return": ret_pri, "volatility": vol_pri},
            alternative_name: {"return": ret_alt, "volatility": vol_alt},
            "differences": {"delta_return": delta_return, "delta_volatility": delta_volatility},
            "tradeoff_ratio": tradeoff_ratio,
            "explanation": explanation
        }
        
        logger.info("Tradeoff analysis complete.")
        return report