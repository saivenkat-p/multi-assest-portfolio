"""
Module: 02_market_regime_detection.market_classifier
Wraps the HMM detector to map mathematical states to semantic financial regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Internal module imports (assuming relative execution context for the pipeline)
from .detector import RegimeDetector
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from base_utils import returns as ret_utils  # Conceptual import based on structure
from base_utils import normalization as norm_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MarketClassifierPipeline:
    """
    End-to-end pipeline to detect and semantically label market regimes.
    """

    def __init__(self, n_regimes: int = 4):
        """
        Initializes the pipeline.

        Args:
            n_regimes (int): Number of regimes to classify.
        """
        self.n_regimes = n_regimes
        self.detector = RegimeDetector(n_regimes=self.n_regimes)
        self.state_mapping: Dict[int, str] = {}
        self.regime_profiles: Dict[str, dict] = {}

    def _map_states_to_labels(self, df_analyzed: pd.DataFrame) -> None:
        """
        Analyzes the statistical properties of each hidden state to assign a semantic label.

        Args:
            df_analyzed (pd.DataFrame): DataFrame containing 'Market_Return', 'Market_Vol', and 'Hidden_State'.
        """
        logger.info("Mapping raw HMM states to semantic financial labels.")
        
        # Calculate median return and volatility for each hidden state
        stats = df_analyzed.groupby('Hidden_State').agg({
            'Market_Return': 'median',
            'Market_Vol': 'median'
        })
        
        # We need a robust heuristic to label 4 states.
        # Rank by return and volatility.
        stats['Vol_Rank'] = stats['Market_Vol'].rank()
        stats['Ret_Rank'] = stats['Market_Return'].rank()
        
        labels = {}
        for state in stats.index:
            ret = stats.loc[state, 'Market_Return']
            vol_rank = stats.loc[state, 'Vol_Rank']
            ret_rank = stats.loc[state, 'Ret_Rank']
            
            # Logic:
            # Highest vol rank is likely High Volatility or Bear
            # Highest ret rank is likely Bull
            if vol_rank == self.n_regimes and ret < 0:
                labels[state] = "Bear"
            elif vol_rank >= (self.n_regimes - 1) and ret >= 0:
                labels[state] = "High Volatility"
            elif ret_rank == self.n_regimes:
                labels[state] = "Bull"
            else:
                labels[state] = "Stable"
                
        # Fallback to ensure unique labels if the heuristic overlaps
        unique_labels = list(set(labels.values()))
        if len(unique_labels) < len(labels):
            logger.warning("Heuristic overlap detected. Reverting to strict rank-based assignment.")
            sorted_by_ret = stats.sort_values(by='Market_Return')
            sorted_states = sorted_by_ret.index.tolist()
            if self.n_regimes == 4:
                # Lowest ret -> Bear, Highest -> Bull. Middle two split by vol.
                labels[sorted_states[0]] = "Bear"
                labels[sorted_states[-1]] = "Bull"
                mid_state_1, mid_state_2 = sorted_states[1], sorted_states[2]
                
                if stats.loc[mid_state_1, 'Market_Vol'] > stats.loc[mid_state_2, 'Market_Vol']:
                    labels[mid_state_1] = "High Volatility"
                    labels[mid_state_2] = "Stable"
                else:
                    labels[mid_state_1] = "Stable"
                    labels[mid_state_2] = "High Volatility"

        self.state_mapping = labels
        
        # Store profiles for explainability module
        for state, label in self.state_mapping.items():
            self.regime_profiles[label] = {
                'median_return': stats.loc[state, 'Market_Return'],
                'median_volatility': stats.loc[state, 'Market_Vol']
            }
        
        logger.info(f"State mapping established: {self.state_mapping}")

    def run_pipeline(self, features_df: pd.DataFrame, market_returns: pd.Series, market_vol: pd.Series) -> Tuple[pd.DataFrame, dict]:
        """
        Executes the regime classification pipeline.

        Args:
            features_df (pd.DataFrame): Feature matrix for the HMM (normalized).
            market_returns (pd.Series): Time-series of aggregate market returns.
            market_vol (pd.Series): Time-series of aggregate market volatility.

        Returns:
            Tuple[pd.DataFrame, dict]: A DataFrame mapping Dates to Regime Labels, and the regime profiles.
        """
        logger.info("Starting Market Classifier Pipeline.")
        
        # 1. Fit HMM
        self.detector.fit(features_df)
        
        # 2. Predict states
        hidden_states = self.detector.predict(features_df)
        
        # 3. Consolidate data for analysis
        analysis_df = pd.DataFrame({
            'Market_Return': market_returns,
            'Market_Vol': market_vol,
            'Hidden_State': hidden_states
        }).dropna()
        
        # 4. Map states to labels
        self._map_states_to_labels(analysis_df)
        
        # 5. Apply mapping to the dataset
        analysis_df['Regime_Label'] = analysis_df['Hidden_State'].map(self.state_mapping)
        
        return analysis_df[['Hidden_State', 'Regime_Label']], self.regime_profiles