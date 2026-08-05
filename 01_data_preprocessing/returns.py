"""
Module: 01_data_preprocessing.returns

Provides functions to calculate various types of financial returns from price data.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_simple_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates period-to-period simple returns.

    Args:
        prices_df (pd.DataFrame): Time-series of asset prices.

    Returns:
        pd.DataFrame: Simple returns, with the first row (NaN) dropped.

    Raises:
        TypeError: If the input is not a pandas DataFrame or contains
            non-numeric columns.
        ValueError: If the input DataFrame is empty.
    """
    logger.info("Calculating simple returns.")

    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame.")

    if prices_df.empty:
        raise ValueError("Input price DataFrame is empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices_df.dtypes):
        raise TypeError("All columns in prices_df must be numeric.")

    # pct_change computes (price_t / price_{t-1}) - 1
    simple_returns = prices_df.pct_change().dropna(how="all")

    return simple_returns


def calculate_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates period-to-period logarithmic returns.

    Useful for time-additive properties in quantitative modeling.

    Args:
        prices_df (pd.DataFrame): Time-series of asset prices.

    Returns:
        pd.DataFrame: Log returns, with the first row (NaN) dropped.

    Raises:
        TypeError: If the input is not a pandas DataFrame or contains
            non-numeric columns.
        ValueError: If the input DataFrame is empty.
    """
    logger.info("Calculating logarithmic returns.")

    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("prices_df must be a pandas DataFrame.")

    if prices_df.empty:
        raise ValueError("Input price DataFrame is empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in prices_df.dtypes):
        raise TypeError("All columns in prices_df must be numeric.")

    # Log return is ln(price_t / price_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna(how="all")

    return log_returns


def calculate_cumulative_returns(
    returns_df: pd.DataFrame,
    is_log: bool = False
) -> pd.DataFrame:
    """
    Calculates cumulative returns over the time series.

    Args:
        returns_df (pd.DataFrame): Time-series of returns (simple or log).
        is_log (bool): Flag indicating if the input returns are logarithmic.

    Returns:
        pd.DataFrame: Cumulative returns.

    Raises:
        TypeError: If the input is not a pandas DataFrame or contains
            non-numeric columns.
        ValueError: If the input DataFrame is empty.
    """
    logger.info(f"Calculating cumulative returns (is_log={is_log}).")

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")

    if returns_df.empty:
        raise ValueError("Input returns DataFrame is empty.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns_df.dtypes):
        raise TypeError("All columns in returns_df must be numeric.")

    if is_log:
        # For log returns, sum them up and exponentiate
        cumulative_log = returns_df.cumsum()
        cumulative_returns = np.exp(cumulative_log) - 1
    else:
        # For simple returns, product of (1 + R)
        cumulative_returns = (1 + returns_df).cumprod() - 1

    return cumulative_returns