"""
Module: 01_data_preprocessing.covariance
Provides functions to calculate standard and exponentially weighted covariance matrices.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_sample_covariance(returns_df: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Calculates the standard annualized sample covariance matrix.

    Args:
        returns_df (pd.DataFrame): Time-series of asset returns.
        trading_days (int): Factor to annualize the covariance (default 252 for daily).

    Returns:
        pd.DataFrame: Annualized N x N covariance matrix.
    """
    logger.info("Calculating standard annualized sample covariance matrix.")
    
    # Calculate daily covariance
    daily_cov = returns_df.cov()
    
    # Annualize
    annualized_cov = daily_cov * trading_days
    return annualized_cov


def calculate_ewma_covariance(
    returns_df: pd.DataFrame, 
    span: int = 60, 
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculates the Exponentially Weighted Moving Average (EWMA) covariance matrix.
    Highly useful for adaptive, regime-aware portfolio optimization.

    Args:
        returns_df (pd.DataFrame): Time-series of asset returns.
        span (int): The span for the exponential weighting (decay factor).
        trading_days (int): Factor to annualize the covariance.

    Returns:
        pd.DataFrame: Annualized N x N EWMA covariance matrix for the latest timestamp.
    """
    logger.info(f"Calculating annualized EWMA covariance matrix with span={span}.")
    
    # Calculate exponentially weighted covariance over the entire series
    ewm_cov = returns_df.ewm(span=span).cov()
    
    # ewm().cov() returns a MultiIndex DataFrame (Date, Asset). 
    # We extract the covariance matrix of the most recent date for current optimization.
    latest_date = returns_df.index[-1]
    latest_ewma_cov = ewm_cov.loc[latest_date]
    
    # Annualize
    annualized_ewma_cov = latest_ewma_cov * trading_days
    return annualized_ewma_cov