"""
Module: 02_market_regime_detection.correlation
Calculates dynamic correlation metrics to detect systemic market movements.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_rolling_correlation(returns_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Calculates the rolling correlation matrices over time.

    Args:
        returns_df (pd.DataFrame): Time-series of asset returns.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: MultiIndex DataFrame (Date, Asset) containing correlation matrices.
    """
    logger.info(f"Calculating {window}-day rolling correlation matrices.")
    return returns_df.rolling(window=window, min_periods=window).corr()


def calculate_mean_cross_correlation(returns_df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Calculates the mean cross-correlation of all assets over time.
    A sudden rise in this metric often signals a transition to a high-stress (Bear) regime.

    Args:
        returns_df (pd.DataFrame): Time-series of asset returns.
        window (int): Rolling window size.

    Returns:
        pd.Series: Time-series of the average market correlation, indexed by Date.
    """
    logger.info(f"Calculating mean cross-correlation over a {window}-day window.")
    
    # 1. Compute rolling correlation matrices (returns a MultiIndex DataFrame)
    rolling_corr = calculate_rolling_correlation(returns_df, window)
    
    def _get_off_diagonal_mean(matrix_df: pd.DataFrame) -> float:
        """Helper to compute the mean of the upper triangle of a correlation matrix."""
        mat_values = matrix_df.values
        # Extract indices for the upper triangle, excluding the main diagonal (k=1)
        upper_triangle_indices = np.triu_indices_from(mat_values, k=1)
        
        if len(upper_triangle_indices[0]) == 0:
            return np.nan
            
        # Extract values and compute mean
        off_diag_values = mat_values[upper_triangle_indices]
        return np.nanmean(off_diag_values)
    
    # 2. Group by Date (level=0) and apply the helper function
    mean_corr_series = rolling_corr.groupby(level=0, group_keys=False).apply(_get_off_diagonal_mean)
    mean_corr_series.name = "Mean_Cross_Correlation"
    
    return mean_corr_series.dropna()