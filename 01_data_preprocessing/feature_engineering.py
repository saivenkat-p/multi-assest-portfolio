"""
Module: 01_data_preprocessing.feature_engineering
Extracts technical indicators and features for market regime classification.
"""

import logging
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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

    Raises:
        TypeError: If prices_df is not a pandas DataFrame or contains non-numeric columns.
        ValueError: If prices_df is empty or window is not positive.
    """
    logger.info("Calculating Simple Moving Average (SMA).")

    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame.")

    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices_df.dtypes):
        raise TypeError("All columns in prices_df must be numeric.")

    if window <= 0:
        raise ValueError("window must be a positive integer.")

    return prices_df.rolling(window=window, min_periods=window).mean()


def calculate_momentum(prices_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculates price momentum over a specific window.

    Args:
        prices_df (pd.DataFrame): Asset prices.
        window (int): Lookback period.

    Returns:
        pd.DataFrame: Momentum values.

    Raises:
        TypeError: If prices_df is not a pandas DataFrame or contains non-numeric columns.
        ValueError: If prices_df is empty or window is not positive.
    """
    logger.info("Calculating Momentum.")

    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame.")

    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices_df.dtypes):
        raise TypeError("All columns in prices_df must be numeric.")

    if window <= 0:
        raise ValueError("window must be a positive integer.")

    return prices_df.pct_change(periods=window)


def calculate_rsi(prices_df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        prices_df (pd.DataFrame): Asset prices.
        window (int): Lookback period.

    Returns:
        pd.DataFrame: RSI values ranging from 0 to 100.

    Raises:
        TypeError: If prices_df is not a pandas DataFrame or contains non-numeric columns.
        ValueError: If prices_df is empty or window is not positive.
    """
    logger.info("Calculating Relative Strength Index (RSI).")

    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame.")

    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices_df.dtypes):
        raise TypeError("All columns in prices_df must be numeric.")

    if window <= 0:
        raise ValueError("window must be a positive integer.")

    delta = prices_df.diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(window=window, min_periods=window)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(window=window, min_periods=window)
        .mean()
    )

    rs = gain / loss.replace(0, 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def build_feature_matrix(
    prices_df: pd.DataFrame,
    windows: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Constructs a comprehensive feature matrix including SMA, Momentum, and RSI.

    Args:
        prices_df (pd.DataFrame): Cleaned historical prices.
        windows (dict, optional): Custom indicator windows.

    Returns:
        pd.DataFrame: Engineered feature matrix.

    Raises:
        TypeError: If prices_df is not a pandas DataFrame or contains non-numeric columns.
        ValueError: If prices_df is empty or windows is missing required keys.
    """
    logger.info("Building engineered feature matrix for market regime detection.")

    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame.")

    if prices_df.empty:
        raise ValueError("prices_df cannot be empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices_df.dtypes):
        raise TypeError("All columns in prices_df must be numeric.")

    if windows is None:
        windows = {
            "sma_short": 20,
            "sma_long": 50,
            "momentum": 21,
            "rsi": 14,
        }

    required_keys = {"sma_short", "sma_long", "momentum", "rsi"}

    if not required_keys.issubset(windows):
        raise ValueError(
            f"windows must contain the keys: {required_keys}"
        )

    features = []

    sma_short = calculate_sma(
        prices_df,
        windows["sma_short"],
    ).add_suffix(f'_SMA_{windows["sma_short"]}')

    sma_long = calculate_sma(
        prices_df,
        windows["sma_long"],
    ).add_suffix(f'_SMA_{windows["sma_long"]}')

    features.extend([sma_short, sma_long])

    momentum = calculate_momentum(
        prices_df,
        windows["momentum"],
    ).add_suffix(f'_Mom_{windows["momentum"]}')

    features.append(momentum)

    rsi = calculate_rsi(
        prices_df,
        windows["rsi"],
    ).add_suffix(f'_RSI_{windows["rsi"]}')

    features.append(rsi)

    feature_matrix = pd.concat(features, axis=1)

    initial_shape = feature_matrix.shape

    feature_matrix = feature_matrix.dropna()

    logger.info(
        "Dropped NaN rows. Feature matrix shape changed from %s to %s.",
        initial_shape,
        feature_matrix.shape,
    )

    return feature_matrix