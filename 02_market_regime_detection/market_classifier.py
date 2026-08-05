"""
Module: 02_market_regime_detection.market_classifier

Wraps the Hidden Markov Model (HMM) detector to map mathematical hidden
states into semantic financial market regimes (e.g., Bull, Bear,
Stable, High Volatility).

This module provides the final interpretation layer of the market regime
detection pipeline by converting unlabeled HMM states into meaningful
financial regimes based on market return and volatility statistics.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Internal module imports
from detector import RegimeDetector

# ---------------------------------------------------------------------
# Project Path Configuration
# ---------------------------------------------------------------------

# Append project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------

BULL = "Bull"
BEAR = "Bear"
STABLE = "Stable"
HIGH_VOLATILITY = "High Volatility"

REQUIRED_ANALYSIS_COLUMNS = (
    "Market_Return",
    "Market_Vol",
    "Hidden_State",
)

# ---------------------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MarketClassifierPipeline:
    """
    End-to-end pipeline for detecting and semantically labeling market
    regimes using a Hidden Markov Model (HMM).

    The pipeline performs the following operations:

    1. Fits the HMM regime detector.
    2. Predicts hidden market states.
    3. Computes statistical characteristics of each hidden state.
    4. Maps hidden states to semantic financial regimes.
    5. Returns regime labels and descriptive regime profiles.
    """

    def __init__(self, n_regimes: int = 4):
        """
        Initialize the market classification pipeline.

        Args:
            n_regimes (int):
                Number of hidden market regimes used by the HMM.

        Raises:
            TypeError:
                If ``n_regimes`` is not an integer.

            ValueError:
                If ``n_regimes`` is less than 2.
        """

        if not isinstance(n_regimes, int):
            raise TypeError("n_regimes must be an integer.")

        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2.")

        self.n_regimes = n_regimes
        self.detector = RegimeDetector(n_regimes=n_regimes)

        # Hidden State -> Semantic Label
        self.state_mapping: Dict[int, str] = {}

        # Semantic Label -> Statistical Profile
        self.regime_profiles: Dict[str, dict] = {}

        logger.info(
            "Initialized MarketClassifierPipeline with %d regimes.",
            self.n_regimes,
        )

    # -----------------------------------------------------------------
    # Validation Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        name: str,
    ) -> None:
        """
        Validate an input DataFrame.

        Args:
            df:
                Input DataFrame.

            name:
                Variable name for error messages.

        Raises:
            TypeError:
                If the input is not a pandas DataFrame.

            ValueError:
                If the DataFrame is empty, contains missing values,
                or contains non-numeric columns.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

        if df.empty:
            raise ValueError(f"{name} cannot be empty.")

        if df.isnull().values.any():
            raise ValueError(f"{name} contains missing values.")

        non_numeric = df.select_dtypes(exclude="number").columns.tolist()

        if non_numeric:
            raise ValueError(
                f"{name} contains non-numeric columns: {non_numeric}"
            )

    @staticmethod
    def _validate_series(
        series: pd.Series,
        name: str,
    ) -> None:
        """
        Validate an input Series.

        Args:
            series:
                Input Series.

            name:
                Variable name for error messages.

        Raises:
            TypeError:
                If the input is not a pandas Series.

            ValueError:
                If the Series is empty, contains missing values,
                or is not numeric.
        """

        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")

        if series.empty:
            raise ValueError(f"{name} cannot be empty.")

        if series.isnull().any():
            raise ValueError(f"{name} contains missing values.")

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"{name} must contain numeric values.")

    def _validate_inputs(
        self,
        features_df: pd.DataFrame,
        market_returns: pd.Series,
        market_vol: pd.Series,
    ) -> None:
        """
        Validate all pipeline inputs.

        Args:
            features_df:
                Normalized feature matrix.

            market_returns:
                Aggregate market returns.

            market_vol:
                Aggregate market volatility.

        Raises:
            ValueError:
                If input lengths or indices are inconsistent.
        """

        logger.info("Validating pipeline inputs.")

        self._validate_dataframe(features_df, "features_df")
        self._validate_series(market_returns, "market_returns")
        self._validate_series(market_vol, "market_vol")

        if len(features_df) != len(market_returns):
            raise ValueError(
                "features_df and market_returns must have the same length."
            )

        if len(features_df) != len(market_vol):
            raise ValueError(
                "features_df and market_vol must have the same length."
            )

        if not features_df.index.equals(market_returns.index):
            raise ValueError(
                "features_df and market_returns must share the same index."
            )

        if not features_df.index.equals(market_vol.index):
            raise ValueError(
                "features_df and market_vol must share the same index."
            )

        logger.info("Input validation completed successfully.")

    def _map_states_to_labels(self, df_analyzed: pd.DataFrame) -> None:
        """
        Map HMM hidden states to semantic financial market regimes.

        Each hidden state is summarized using the median market return and
        median market volatility. These statistics are then used to assign
        interpretable regime labels.

        Current labels:

        - Bull
        - Bear
        - Stable
        - High Volatility

        Args:
            df_analyzed:
                DataFrame containing the following columns:

                - Market_Return
                - Market_Vol
                - Hidden_State

        Raises:
            TypeError:
                If the input is not a pandas DataFrame.

            ValueError:
                If required columns are missing, the DataFrame is empty,
                or contains missing values.
        """

        logger.info("Mapping HMM hidden states to semantic market regimes.")

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        self._validate_dataframe(df_analyzed, "df_analyzed")

        missing_columns = [
            col
            for col in REQUIRED_ANALYSIS_COLUMNS
            if col not in df_analyzed.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        logger.info("Computing regime statistics.")

        # -------------------------------------------------------------
        # Compute state statistics
        # -------------------------------------------------------------

        stats = (
            df_analyzed
            .groupby("Hidden_State")
            .agg(
                {
                    "Market_Return": "median",
                    "Market_Vol": "median",
                }
            )
        )

        stats["Vol_Rank"] = stats["Market_Vol"].rank()
        stats["Ret_Rank"] = stats["Market_Return"].rank()

        logger.info(
            "Computed statistics for %d hidden states.",
            len(stats),
        )

        # -------------------------------------------------------------
        # Initial heuristic mapping
        # -------------------------------------------------------------

        labels: Dict[int, str] = {}

        for state in stats.index:

            median_return = stats.loc[state, "Market_Return"]
            vol_rank = stats.loc[state, "Vol_Rank"]
            ret_rank = stats.loc[state, "Ret_Rank"]

            if (
                vol_rank == self.n_regimes
                and median_return < 0
            ):
                labels[state] = BEAR

            elif (
                vol_rank >= (self.n_regimes - 1)
                and median_return >= 0
            ):
                labels[state] = HIGH_VOLATILITY

            elif ret_rank == self.n_regimes:
                labels[state] = BULL

            else:
                labels[state] = STABLE

        # -------------------------------------------------------------
        # Resolve duplicate labels
        # -------------------------------------------------------------

        if len(set(labels.values())) < len(labels):

            logger.warning(
                "Duplicate semantic labels detected. "
                "Applying deterministic fallback mapping."
            )

            sorted_states = (
                stats.sort_values("Market_Return")
                .index
                .tolist()
            )

            if self.n_regimes == 4:

                labels[sorted_states[0]] = BEAR
                labels[sorted_states[-1]] = BULL

                mid_state_1 = sorted_states[1]
                mid_state_2 = sorted_states[2]

                if (
                    stats.loc[mid_state_1, "Market_Vol"]
                    >
                    stats.loc[mid_state_2, "Market_Vol"]
                ):

                    labels[mid_state_1] = HIGH_VOLATILITY
                    labels[mid_state_2] = STABLE

                else:

                    labels[mid_state_1] = STABLE
                    labels[mid_state_2] = HIGH_VOLATILITY

        # -------------------------------------------------------------
        # Store mappings
        # -------------------------------------------------------------

        self.state_mapping = labels

        self.regime_profiles.clear()

        for state, label in self.state_mapping.items():

            self.regime_profiles[label] = {
                "median_return": stats.loc[state, "Market_Return"],
                "median_volatility": stats.loc[state, "Market_Vol"],
            }

        logger.info(
            "Successfully mapped %d hidden states.",
            len(self.state_mapping),
        )

        logger.info(
            "State mapping: %s",
            self.state_mapping,
        )

    def run_pipeline(
        self,
        features_df: pd.DataFrame,
        market_returns: pd.Series,
        market_vol: pd.Series,
    ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """
        Execute the complete market regime classification pipeline.

        The pipeline performs the following steps:

        1. Validate all inputs.
        2. Fit the Hidden Markov Model.
        3. Predict hidden market states.
        4. Combine market statistics.
        5. Assign semantic market regime labels.
        6. Return labeled regimes and regime profiles.

        Args:
            features_df:
                Normalized feature matrix used for HMM training.

            market_returns:
                Aggregate market return time series.

            market_vol:
                Aggregate market volatility time series.

        Returns:
            Tuple[pd.DataFrame, Dict[str, dict]]:

            First element:
                DataFrame containing

                - Hidden_State
                - Regime_Label

            Second element:
                Dictionary containing statistical profiles for each
                semantic market regime.

        Raises:
            TypeError:
                If any input has an invalid type.

            ValueError:
                If validation fails.

            RuntimeError:
                If the HMM fitting or prediction process fails.
        """

        logger.info("Starting Market Classifier Pipeline.")

        # -------------------------------------------------------------
        # Validate Inputs
        # -------------------------------------------------------------

        self._validate_inputs(
            features_df,
            market_returns,
            market_vol,
        )

        try:

            # ---------------------------------------------------------
            # Train HMM
            # ---------------------------------------------------------

            logger.info("Fitting Hidden Markov Model.")

            self.detector.fit(features_df)

            logger.info("Hidden Markov Model fitted successfully.")

            # ---------------------------------------------------------
            # Predict Hidden States
            # ---------------------------------------------------------

            logger.info("Predicting hidden market states.")

            hidden_states = self.detector.predict(features_df)

            logger.info(
                "Successfully predicted %d hidden states.",
                len(hidden_states),
            )

            # ---------------------------------------------------------
            # Prepare Analysis Data
            # ---------------------------------------------------------

            logger.info(
                "Constructing analysis DataFrame."
            )

            analysis_df = pd.DataFrame(
                {
                    "Market_Return": market_returns,
                    "Market_Vol": market_vol,
                    "Hidden_State": hidden_states,
                }
            ).dropna()

            if analysis_df.empty:
                raise ValueError(
                    "Analysis DataFrame is empty after removing "
                    "missing values."
                )

            logger.info(
                "Analysis DataFrame contains %d observations.",
                len(analysis_df),
            )

            # ---------------------------------------------------------
            # Map Hidden States
            # ---------------------------------------------------------

            self._map_states_to_labels(analysis_df)

            logger.info(
                "Applying semantic regime labels."
            )

            analysis_df["Regime_Label"] = (
                analysis_df["Hidden_State"]
                .map(self.state_mapping)
            )

            if analysis_df["Regime_Label"].isnull().any():
                raise RuntimeError(
                    "Failed to assign semantic labels "
                    "to one or more hidden states."
                )

            logger.info(
                "Market regime classification completed successfully."
            )

            return (
                analysis_df[
                    [
                        "Hidden_State",
                        "Regime_Label",
                    ]
                ],
                self.regime_profiles,
            )

        except Exception as exc:

            logger.exception(
                "Market classification pipeline failed."
            )

            raise RuntimeError(
                "Failed to execute MarketClassifierPipeline."
            ) from exc