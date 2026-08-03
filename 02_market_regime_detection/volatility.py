"""
Module: 02_market_regime_detection.volatility
Calculates various volatility metrics to aid in market regime classification.
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


def calculate_rolling_volatility(
    returns_df: pd.DataFrame, 
    window: int = 21, 
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculates the annualized rolling historical volatility.

    Args:
        returns_df (pd.DataFrame): Asset returns.
        window (int): The rolling window size in days (default 21).
        trading_days (int): Annualization factor.

    Returns:
        pd.DataFrame: Annualized rolling volatility, dropping initial NaNs.
    """
    logger.info(f"Calculating {window}-day annualized rolling volatility.")
    
    # Calculate rolling standard deviation
    rolling_std = returns_df.rolling(window=window, min_periods=window).std()
    
    # Annualize
    annualized_vol = rolling_std * np.sqrt(trading_days)
    
    return annualized_vol.dropna(how='all')


def calculate_ewma_volatility(
    returns_df: pd.DataFrame, 
    span: int = 21, 
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculates the annualized exponentially weighted moving average (EWMA) volatility.
    More responsive to recent market shocks than simple rolling volatility.

    Args:
        returns_df (pd.DataFrame): Asset returns.
        span (int): The span for exponential weighting.
        trading_days (int): Annualization factor.

    Returns:
        pd.DataFrame: Annualized EWMA volatility, dropping initial NaNs.
    """
    logger.info(f"Calculating EWMA volatility with span={span}.")
    
    # Calculate exponentially weighted standard deviation
    ewma_std = returns_df.ewm(span=span, min_periods=span).std()
    
    # Annualize
    annualized_ewma_vol = ewma_std * np.sqrt(trading_days)
    
    return annualized_ewma_vol.dropna(how='all')