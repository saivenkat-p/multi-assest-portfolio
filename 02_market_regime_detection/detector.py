"""
Module: 02_market_regime_detection.detector

Implements a Gaussian Hidden Markov Model (HMM) for detecting
market regimes from engineered financial features.

This module integrates with:
    01_data_preprocessing.feature_engineering

and produces regime labels for downstream portfolio optimization.
"""

from typing import Dict
import logging
import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:
    raise ImportError(
        "The 'hmmlearn' package is required. "
        "Install it using: pip install hmmlearn"
    ) from exc


logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


class RegimeDetector:
    """
    Detect market regimes using a Gaussian Hidden Markov Model (HMM).

    The detector learns latent market states from engineered
    financial features and predicts the hidden regime associated
    with each observation.

    Attributes:
        n_regimes (int):
            Number of hidden market regimes.

        random_state (int):
            Random seed used for reproducibility.

        model (GaussianHMM):
            Underlying Hidden Markov Model.

        is_fitted (bool):
            Indicates whether the model has been trained.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the RegimeDetector.

        Args:
            n_regimes:
                Number of hidden market regimes.

            random_state:
                Random seed for reproducibility.

        Raises:
            TypeError:
                If n_regimes or random_state is not an integer.

            ValueError:
                If n_regimes is less than 2.
        """

        if not isinstance(n_regimes, int):
            logger.error("n_regimes must be an integer.")
            raise TypeError("n_regimes must be an integer.")

        if n_regimes < 2:
            logger.error("n_regimes must be at least 2.")
            raise ValueError("n_regimes must be at least 2.")

        if not isinstance(random_state, int):
            logger.error("random_state must be an integer.")
            raise TypeError("random_state must be an integer.")

        self.n_regimes = n_regimes
        self.random_state = random_state

        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=random_state,
        )

        self.is_fitted = False

        logger.info(
            "Initialized RegimeDetector with %d regimes.",
            self.n_regimes,
        )

    def _validate_feature_matrix(
        self,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """
        Validate the feature matrix before training or prediction.

        Args:
            feature_matrix:
                Engineered feature matrix.

        Raises:
            TypeError:
                If input is not a pandas DataFrame.

            ValueError:
                If the DataFrame is empty.

            ValueError:
                If the DataFrame contains fewer than two rows.

            ValueError:
                If NaN values are present.

            TypeError:
                If non-numeric columns are detected.
        """

        if not isinstance(feature_matrix, pd.DataFrame):
            logger.error(
                "feature_matrix must be a pandas DataFrame."
            )
            raise TypeError(
                "feature_matrix must be a pandas DataFrame."
            )

        if feature_matrix.empty:
            logger.error("feature_matrix cannot be empty.")
            raise ValueError(
                "feature_matrix cannot be empty."
            )

        if len(feature_matrix) < 2:
            logger.error(
                "feature_matrix must contain at least two observations."
            )
            raise ValueError(
                "feature_matrix must contain at least two observations."
            )

        if feature_matrix.isnull().values.any():
            logger.error(
                "feature_matrix contains missing values."
            )
            raise ValueError(
                "feature_matrix contains NaN values."
            )

        non_numeric = [
            column
            for column in feature_matrix.columns
            if not pd.api.types.is_numeric_dtype(
                feature_matrix[column]
            )
        ]

        if non_numeric:
            logger.error(
                "Non-numeric columns detected: %s",
                non_numeric,
            )
            raise TypeError(
                f"Non-numeric columns detected: {non_numeric}"
            )

    def fit(self, feature_matrix: pd.DataFrame) -> None:
        """
        Fit the Gaussian HMM to the provided feature matrix.
        """
        self._validate_feature_matrix(feature_matrix)

        logger.info(
            "Fitting Gaussian HMM with %d regimes on %d observations.",
            self.n_regimes,
            len(feature_matrix),
        )

        try:
            X = feature_matrix.to_numpy(dtype=np.float64)
            self.model.fit(X)
            self.is_fitted = True
            logger.info("HMM fitting completed successfully.")

        except Exception as exc:
            logger.exception("Failed to fit Hidden Markov Model.")
            raise RuntimeError(
                "Failed to fit Hidden Markov Model."
            ) from exc

    def predict(
        self,
        feature_matrix: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict the hidden market regime for each observation.

        Args:
            feature_matrix:
                DataFrame containing engineered financial features.

        Returns:
            pd.Series:
                Predicted regime labels indexed by the original
                DataFrame index.

        Raises:
            RuntimeError:
                If the model has not been fitted.

            TypeError:
                If the input is invalid.

            ValueError:
                If the feature matrix fails validation.
        """

        if not self.is_fitted:
            logger.error(
                "Prediction attempted before fitting the model."
            )
            raise RuntimeError(
                "The model must be fitted before calling predict()."
            )

        self._validate_feature_matrix(feature_matrix)

        logger.info(
            "Predicting market regimes for %d observations.",
            len(feature_matrix),
        )

        try:
            X = feature_matrix.to_numpy(dtype=np.float64)
            hidden_states = self.model.predict(X)
            logger.info("Prediction completed successfully.")

            return pd.Series(
                hidden_states,
                index=feature_matrix.index,
                name="Regime",
                dtype="int64",
            )

        except Exception as exc:
            logger.exception(
                "Failed to predict market regimes."
            )
            raise RuntimeError(
                "Failed to predict market regimes."
            ) from exc
            
    def predict_proba(
        self,
        feature_matrix:pd.DataFrame, 
        
    ) -> pd.DataFrame:    
       if not self.is_fitted:
            logger.error(
               "Prediction probabilities requested before fitting the model."     
            )
            raise RuntimeError(
                "The model must be fitted before calling predict_proba()."    
            )    
            
       self._validate_feature_matrix(feature_matrix)
        
       logger.info(
            "Predicting regime probabilities for %d observations.",
            len(feature_matrix),
        )
        
        
       try:
            X = feature_matrix.to_numpy(dtype=np.float64)

            probabilities = self.model.predict_proba(X)

            columns = [
                f"Regime_{i}_Probability"
                for i in range(self.n_regimes)
            ]

            logger.info(
                "Regime probability prediction completed successfully."
            )

            return pd.DataFrame(
                probabilities,
                index=feature_matrix.index,
                columns=columns,
            )
            
       except Exception as exc:
            logger.exception(
                "Failed to predict regime probabilities."
            )
            raise RuntimeError(
                "Failed to predict regime probabilities."
            ) from exc    
    

    def get_regime_parameters(self) -> Dict[str, np.ndarray]:
        """
        Retrieve the learned Hidden Markov Model parameters.

        Returns:
            Dict[str, np.ndarray]:
                Dictionary containing:

                - transition_matrix
                - means
                - covariances

        Raises:
            RuntimeError:
                If the model has not been fitted.
        """

        if not self.is_fitted:
            logger.error(
                "Attempted to retrieve HMM parameters before fitting."
            )
            raise RuntimeError(
                "The model must be fitted before extracting parameters."
            )

        logger.info("Extracting learned HMM parameters.")

        return {
            "transition_matrix": self.model.transmat_,
            "means": self.model.means_,
            "covariances": self.model.covars_,
        }

    def get_current_regime(
        self,
        feature_matrix: pd.DataFrame,
    ) -> int:
        """
        Predict the most recent market regime.

        This is typically used by the optimization module to
        determine the current market state before selecting
        adaptive optimization weights.

        Args:
            feature_matrix:
                Engineered financial feature matrix.

        Returns:
            int:
                Predicted regime label of the latest observation.

        Raises:
            RuntimeError:
                If prediction fails.
        """

        logger.info("Predicting current market regime.")

        regimes = self.predict(feature_matrix)
        current_regime = int(regimes.iloc[-1])

        logger.info(
            "Current detected regime: %d",
            current_regime,
        )

        return current_regime