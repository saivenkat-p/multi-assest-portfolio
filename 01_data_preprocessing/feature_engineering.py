"""
Module: 01_data_preprocessing.feature_engineering
Extracts technical indicators and features for market regime classification.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_sma(prices_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculates the Simple Moving Average (SMA).

    Args:
        prices_df (pd.DataFrame): Asset prices.
        window (int): Rolling window size.

    Returns:
        pd.DataFrame: SMA values.
    """
    return prices_df.rolling(window=window, min_periods=window).mean()


def calculate_momentum(prices_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculates price momentum over a specific window.

    Args:
        prices_df (pd.DataFrame): Asset prices.
        window (int): Lookback period for momentum.

    Returns:
        pd.DataFrame: Momentum values.
    """
    return prices_df.pct_change(periods=window)


def calculate_rsi(prices_df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        prices_df (pd.DataFrame): Asset prices.
        window (int): Lookback period (default is 14 days).

    Returns:
        pd.DataFrame: RSI values ranging from 0 to 100.
    """
    delta = prices_df.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def build_feature_matrix(
    prices_df: pd.DataFrame, 
    windows: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Constructs a comprehensive feature matrix including SMA, Momentum, and RSI for all assets.

    Args:
        prices_df (pd.DataFrame): Cleaned historical prices.
        windows (dict, optional): Custom windows for indicators.

    Returns:
        pd.DataFrame: Concatenated feature matrix with multi-level columns or flattened names.
    """
    logger.info("Building engineered feature matrix for market regime detection.")
    
    if windows is None:
        windows = {'sma_short': 20, 'sma_long': 50, 'momentum': 21, 'rsi': 14}
        
    features = []
    
    # 1. Simple Moving Averages
    sma_short = calculate_sma(prices_df, windows['sma_short']).add_suffix(f'_SMA_{windows["sma_short"]}')
    sma_long = calculate_sma(prices_df, windows['sma_long']).add_suffix(f'_SMA_{windows["sma_long"]}')
    features.extend([sma_short, sma_long])
    
    # 2. Momentum
    momentum = calculate_momentum(prices_df, windows['momentum']).add_suffix(f'_Mom_{windows["momentum"]}')
    features.append(momentum)
    
    # 3. RSI
    rsi = calculate_rsi(prices_df, windows['rsi']).add_suffix(f'_RSI_{windows["rsi"]}')
    features.append(rsi)
    
    # Combine all features along the columns
    feature_matrix = pd.concat(features, axis=1)
    
    # Drop rows with NaN values resulting from rolling windows
    initial_shape = feature_matrix.shape
    feature_matrix.dropna(inplace=True)
    logger.info(f"Dropped NaN rows. Feature matrix shape changed from {initial_shape} to {feature_matrix.shape}.")
    
    return feature_matrix