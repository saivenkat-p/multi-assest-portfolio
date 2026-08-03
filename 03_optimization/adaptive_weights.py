"""
Module: 03_optimization.adaptive_weights
Adjusts optimization parameters dynamically based on detected market regimes.
"""

import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RegimeAdaptiveParameters:
    """
    Modifies optimization constraints and objectives based on market state.
    """

    def __init__(self, base_risk_aversion: float = 1.0, base_penalty: float = 1000.0):
        """
        Initializes the adaptive parameter generator.

        Args:
            base_risk_aversion (float): Default gamma.
            base_penalty (float): Default QUBO penalty multiplier.
        """
        self.base_risk_aversion = base_risk_aversion
        self.base_penalty = base_penalty
        
        # Define multipliers for risk aversion based on regime
        self.risk_multipliers: Dict[str, float] = {
            "Bull": 0.5,             # Seek return
            "Stable": 1.0,           # Balanced
            "Bear": 2.5,             # Avoid risk
            "High Volatility": 4.0   # Extreme risk avoidance
        }

    def get_adjusted_risk_aversion(self, current_regime: str) -> float:
        """
        Calculates the adjusted risk aversion parameter.

        Args:
            current_regime (str): The active market regime label.

        Returns:
            float: Adjusted risk aversion.
        """
        multiplier = self.risk_multipliers.get(current_regime, 1.0)
        adjusted_gamma = self.base_risk_aversion * multiplier
        logger.info(f"Regime '{current_regime}' active. Adjusted Risk Aversion: {adjusted_gamma}")
        return float(adjusted_gamma)

    def get_adjusted_bounds(self, current_regime: str, n_assets: int) -> Tuple[Tuple[float, float], ...]:
        """
        Calculates asset weight bounds restricted by the current market regime.

        Args:
            current_regime (str): The active market regime label.
            n_assets (int): Number of assets in the portfolio.

        Returns:
            Tuple[Tuple[float, float], ...]: Adjusted (min, max) bounds for each asset.
        """
        # Default long-only bounds
        max_weight = 1.0 
        
        # Restrict maximum single-asset exposure in dangerous regimes
        if current_regime == "High Volatility":
            max_weight = min(1.0, 3.0 / n_assets) # Enforce strict diversification
            logger.info(f"High Volatility detected. Imposing strict maximum weight limit: {max_weight:.2f}")
        elif current_regime == "Bear":
            max_weight = min(1.0, 5.0 / n_assets)
            logger.info(f"Bear market detected. Imposing moderate maximum weight limit: {max_weight:.2f}")
            
        return tuple((0.0, max_weight) for _ in range(n_assets))
        
    def get_adjusted_penalty(self, current_regime: str) -> float:
        """
        Scales the penalty factor if constraints need to be stricter in certain regimes.
        
        Args:
            current_regime (str): The active market regime label.
            
        Returns:
            float: Adjusted penalty factor.
        """
        if current_regime in ["High Volatility", "Bear"]:
            return self.base_penalty * 2.0
        return self.base_penalty