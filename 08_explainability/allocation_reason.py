"""
Module: 08_explainability.allocation_reason
Generates plain-English rationales for portfolio weight distributions.
"""

import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AllocationExplainer:
    """
    Translates numerical portfolio weights into logical, human-readable explanations.
    """

    def __init__(self, weight_threshold: float = 0.01):
        """
        Initializes the explainer.
        
        Args:
            weight_threshold (float): Weights below this are considered omitted.
        """
        self.weight_threshold = weight_threshold

    def explain(
        self, 
        weights: np.ndarray, 
        asset_names: List[str], 
        expected_returns: np.ndarray, 
        covariance_matrix: np.ndarray
    ) -> Dict[str, List[str]]:
        """
        Generates rationales for the provided allocations.
        """
        if not (len(weights) == len(asset_names) == len(expected_returns) == len(covariance_matrix)):
            raise ValueError("All input arrays and lists must have the same length.")

        logger.info("Generating explainability report for asset allocations.")
        
        # Calculate individual asset volatilities (sqrt of diagonal)
        volatilities = np.sqrt(np.diag(covariance_matrix))
        
        # Determine market baselines
        median_return = np.median(expected_returns)
        median_volatility = np.median(volatilities)
        
        report = {
            "Yield_Seeking": [],
            "Risk_Mitigating": [],
            "Omitted": []
        }
        
        for i, asset in enumerate(asset_names):
            w = weights[i]
            ret = expected_returns[i]
            vol = volatilities[i]
            
            if w < self.weight_threshold:
                reason = f"{asset}: Omitted (Weight: {w:.1%}). "
                if ret < median_return and vol > median_volatility:
                    reason += "Sub-optimal risk/return profile compared to peers."
                elif ret < median_return:
                    reason += "Expected returns are too low to justify inclusion."
                else:
                    reason += "High volatility caused optimizer to reject it."
                report["Omitted"].append(reason)
                
            else:
                if ret >= median_return:
                    reason = f"{asset}: Allocated {w:.1%} primarily for yield generation (Return {ret:.2%} >= Median {median_return:.2%})."
                    report["Yield_Seeking"].append(reason)
                elif vol <= median_volatility:
                    reason = f"{asset}: Allocated {w:.1%} primarily for portfolio stability (Volatility {vol:.2%} <= Median {median_volatility:.2%})."
                    report["Risk_Mitigating"].append(reason)
                else:
                    # Edge case where an asset has low return and high volatility but still gets weight (often due to negative correlation/diversification)
                    reason = f"{asset}: Allocated {w:.1%} for diversification benefits and covariance offset."
                    report["Risk_Mitigating"].append(reason)
                    
        logger.info("Explainability report generated successfully.")
        return report
    