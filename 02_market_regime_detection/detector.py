"""
Module: 02_market_regime_detection.detector
Implements a Hidden Markov Model (HMM) to classify time-series data into discrete market regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("The 'hmmlearn' library is required. Install it via 'pip install hmmlearn'.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RegimeDetector:
    """
    Detects market regimes using a Gaussian Hidden Markov Model.
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        """
        Initializes the Regime Detector.

        Args:
            n_regimes (int): Number of distinct market environments to model.
                             (e.g., 4: Bull, Bear, Stable, High-Vol).
            random_state (int): Seed for reproducibility.
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=self.n_regimes, 
            covariance_type="full", 
            n_iter=1000, 
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit(self, feature_matrix: pd.DataFrame) -> None:
        """
        Fits the HMM to the historical feature matrix.

        Args:
            feature_matrix (pd.DataFrame): Time-series features (no NaNs).
        """
        logger.info(f"Fitting HMM with {self.n_regimes} regimes on {len(feature_matrix)} data points.")
        X = feature_matrix.values
        
        # Fit the model using Expectation-Maximization
        self.model.fit(X)
        self.is_fitted = True
        logger.info("HMM fitting complete.")

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """
        Predicts the market regime for each timestamp in the feature matrix.

        Args:
            feature_matrix (pd.DataFrame): Time-series features to evaluate.

        Returns:
            pd.Series: Integer sequence of predicted regimes, indexed by Date.
        """
        if not self.is_fitted:
            logger.error("Attempted to predict without fitting the model first.")
            raise RuntimeError("The model must be fitted before calling predict().")

        logger.info("Predicting hidden market states.")
        X = feature_matrix.values
        
        # Viterbi algorithm infers the most likely sequence of hidden states
        hidden_states = self.model.predict(X)
        
        return pd.Series(hidden_states, index=feature_matrix.index, name="Regime")

    def get_regime_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extracts the learned transition matrix, means, and covariances.

        Returns:
            Dict: Dictionary containing 'transition_matrix', 'means', and 'covariances'.
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before extracting parameters.")
            
        return {
            "transition_matrix": self.model.transmat_,
            "means": self.model.means_,
            "covariances": self.model.covars_
        }
        
    def get_current_regime(self, feature_matrix: pd.DataFrame) -> int:
        """
        Infers the regime of the most recent data point.
        
        Args:
            feature_matrix (pd.DataFrame): Latest time-series features.
            
        Returns:
            int: The current regime identifier.
        """
        regimes = self.predict(feature_matrix)
        return int(regimes.iloc[-1])