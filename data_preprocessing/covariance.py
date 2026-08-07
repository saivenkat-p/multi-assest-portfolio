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
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def calculate_sample_covariance(
    returns_df: pd.DataFrame,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculates the standard annualized sample covariance matrix.

    Args:
        returns_df (pd.DataFrame):
            Time-series of asset returns.

        trading_days (int):
            Number of trading days used for annualization
            (default: 252).

    Returns:
        pd.DataFrame:
            Annualized N × N covariance matrix.

    Raises:
        TypeError:
            If returns_df is not a pandas DataFrame.

        ValueError:
            If returns_df is empty.

        ValueError:
            If returns_df contains non-numeric columns.

        ValueError:
            If trading_days is not positive.
    """
    logger.info("Calculating standard annualized sample covariance matrix.")

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")

    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")

    if returns_df.select_dtypes(include="number").shape[1] != returns_df.shape[1]:
        raise ValueError("returns_df must contain only numeric columns.")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

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
    Calculates the Exponentially Weighted Moving Average (EWMA)
    covariance matrix.

    Highly useful for adaptive, regime-aware portfolio optimization.

    Args:
        returns_df (pd.DataFrame):
            Time-series of asset returns.

        span (int):
            Span used for exponential weighting.

        trading_days (int):
            Number of trading days used for annualization.

    Returns:
        pd.DataFrame:
            Annualized N × N EWMA covariance matrix for the latest
            timestamp.

    Raises:
        TypeError:
            If returns_df is not a pandas DataFrame.

        ValueError:
            If returns_df is empty.

        ValueError:
            If returns_df contains non-numeric columns.

        ValueError:
            If span is not positive.

        ValueError:
            If trading_days is not positive.
    """
    logger.info(
        f"Calculating annualized EWMA covariance matrix with span={span}."
    )

    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")

    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")

    if returns_df.select_dtypes(include="number").shape[1] != returns_df.shape[1]:
        raise ValueError("returns_df must contain only numeric columns.")

    if span <= 0:
        raise ValueError("span must be positive.")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    # Calculate exponentially weighted covariance over the entire series
    ewm_cov = returns_df.ewm(span=span).cov()

    # ewm().cov() returns a MultiIndex DataFrame (Date, Asset).
    # Extract the covariance matrix corresponding to the latest date.
    latest_date = returns_df.index[-1]
    latest_ewma_cov = ewm_cov.loc[latest_date]

    # Annualize
    annualized_ewma_cov = latest_ewma_cov * trading_days

    return annualized_ewma_cov