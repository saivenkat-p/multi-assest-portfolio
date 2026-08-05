"""
Module: 02_market_regime_detection.volatility

Calculates rolling and exponentially weighted volatility metrics for
financial asset returns. These volatility features are used as inputs
to the market regime detection pipeline.

Author: ARQPO Project
"""

import logging
from typing import Final

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logger Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

DEFAULT_TRADING_DAYS: Final[int] = 252


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _validate_returns_dataframe(returns_df: pd.DataFrame) -> None:
    """
    Validate the input returns DataFrame.

    Args:
        returns_df:
            DataFrame containing asset returns.

    Raises:
        TypeError:
            If the input is not a pandas DataFrame.

        ValueError:
            If the DataFrame is empty.

        ValueError:
            If the DataFrame contains missing values.

        TypeError:
            If one or more columns are non-numeric.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")

    if returns_df.empty:
        raise ValueError("returns_df cannot be empty.")

    if returns_df.isnull().values.any():
        raise ValueError("returns_df contains missing values.")

    non_numeric = returns_df.select_dtypes(exclude=[np.number]).columns

    if len(non_numeric) > 0:
        raise TypeError(
            f"Non-numeric columns detected: {list(non_numeric)}"
        )


def _validate_positive_integer(value: int, name: str) -> None:
    """
    Validate that a parameter is a positive integer.

    Args:
        value:
            Parameter value.

        name:
            Parameter name.

    Raises:
        TypeError:
            If value is not an integer.

        ValueError:
            If value is less than or equal to zero.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


# -----------------------------------------------------------------------------
# Rolling Volatility
# -----------------------------------------------------------------------------

def calculate_rolling_volatility(
    returns_df: pd.DataFrame,
    window: int = 21,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> pd.DataFrame:
    """
    Calculate annualized rolling historical volatility.

    Volatility is computed as the rolling standard deviation of returns
    and annualized using the square root of the number of trading days.

    Args:
        returns_df:
            DataFrame of asset returns.

        window:
            Rolling window size.

        trading_days:
            Number of trading days used for annualization.

    Returns:
        Annualized rolling volatility.

    Raises:
        TypeError:
            If inputs have invalid types.

        ValueError:
            If inputs contain invalid values.
    """
    logger.info(
        "Calculating %d-day rolling annualized volatility.",
        window,
    )

    try:
        _validate_returns_dataframe(returns_df)
        _validate_positive_integer(window, "window")
        _validate_positive_integer(trading_days, "trading_days")

        rolling_std = returns_df.rolling(
            window=window,
            min_periods=window,
        ).std()

        annualized_vol = rolling_std * np.sqrt(trading_days)

        logger.info("Rolling volatility calculation completed.")

        return annualized_vol.dropna(how="all")

    except Exception:
        logger.exception("Rolling volatility calculation failed.")
        raise


# -----------------------------------------------------------------------------
# EWMA Volatility
# -----------------------------------------------------------------------------

def calculate_ewma_volatility(
    returns_df: pd.DataFrame,
    span: int = 21,
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> pd.DataFrame:
    """
    Calculate annualized exponentially weighted moving volatility.

    EWMA volatility assigns larger weights to recent observations,
    making it more responsive to market changes than rolling volatility.

    Args:
        returns_df:
            DataFrame of asset returns.

        span:
            Exponential moving window span.

        trading_days:
            Number of trading days used for annualization.

    Returns:
        Annualized EWMA volatility.

    Raises:
        TypeError:
            If inputs have invalid types.

        ValueError:
            If inputs contain invalid values.
    """
    logger.info(
        "Calculating EWMA annualized volatility (span=%d).",
        span,
    )

    try:
        _validate_returns_dataframe(returns_df)
        _validate_positive_integer(span, "span")
        _validate_positive_integer(trading_days, "trading_days")

        ewma_std = returns_df.ewm(
            span=span,
            min_periods=span,
        ).std()

        annualized_vol = ewma_std * np.sqrt(trading_days)

        logger.info("EWMA volatility calculation completed.")

        return annualized_vol.dropna(how="all")

    except Exception:
        logger.exception("EWMA volatility calculation failed.")
        raise 