"""
Module: 02_market_regime_detection.correlation

Calculates rolling correlation metrics used for market regime detection.
The module provides rolling correlation matrices and mean cross-correlation
to identify periods of increased systemic market risk.
"""

import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------

DEFAULT_WINDOW = 60

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


# ---------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------

def _validate_returns_dataframe(returns_df: pd.DataFrame) -> None:
    """
    Validate the input returns DataFrame.

    Args:
        returns_df (pd.DataFrame):
            Asset returns DataFrame.

    Raises:
        TypeError:
            If input is not a pandas DataFrame.

        ValueError:
            If DataFrame is empty.

        ValueError:
            If DataFrame contains missing values.

        ValueError:
            If DataFrame contains non-numeric columns.

        ValueError:
            If index is not a DatetimeIndex.
    """
    if not isinstance(returns_df, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame.")
        raise TypeError("returns_df must be a pandas DataFrame.")

    if returns_df.empty:
        logger.error("Returns DataFrame is empty.")
        raise ValueError("returns_df cannot be empty.")

    if returns_df.isnull().values.any():
        logger.error("Returns DataFrame contains missing values.")
        raise ValueError("returns_df contains missing values.")

    if not all(
        np.issubdtype(dtype, np.number)
        for dtype in returns_df.dtypes
    ):
        logger.error("Returns DataFrame contains non-numeric columns.")
        raise ValueError("All columns must be numeric.")

    if not isinstance(returns_df.index, pd.DatetimeIndex):
        logger.error("Returns DataFrame index must be DatetimeIndex.")
        raise ValueError(
            "returns_df index must be a pandas DatetimeIndex."
        )

    logger.debug("Returns DataFrame validation successful.")


def _validate_window(window: int, data_length: int) -> None:
    """
    Validate rolling window parameter.

    Args:
        window (int):
            Rolling window size.

        data_length (int):
            Number of observations.

    Raises:
        TypeError:
            If window is not an integer.

        ValueError:
            If window is less than 2.

        ValueError:
            If window exceeds available observations.
    """
    if not isinstance(window, int):
        logger.error("Window must be an integer.")
        raise TypeError("window must be an integer.")

    if window < 2:
        logger.error("Window must be at least 2.")
        raise ValueError("window must be greater than or equal to 2.")

    if window > data_length:
        logger.error(
            "Window (%d) exceeds available observations (%d).",
            window,
            data_length,
        )
        raise ValueError(
            "window cannot be larger than the number of observations."
        )

    logger.debug("Rolling window validation successful.")


# ---------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------

def calculate_rolling_correlation(
    returns_df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Calculate rolling correlation matrices.

    Args:
        returns_df (pd.DataFrame):
            Time-series asset returns.

        window (int, optional):
            Rolling window size.
            Defaults to DEFAULT_WINDOW.

    Returns:
        pd.DataFrame:
            MultiIndex DataFrame containing rolling
            correlation matrices.

    Raises:
        TypeError:
            If inputs are invalid.

        ValueError:
            If validation fails.
    """
    logger.info(
        "Calculating rolling correlation matrices "
        "(window=%d).",
        window,
    )

    _validate_returns_dataframe(returns_df)
    _validate_window(window, len(returns_df))

    try:
        rolling_corr = returns_df.rolling(
            window=window,
            min_periods=window,
        ).corr()

        logger.info(
            "Rolling correlation calculation completed "
            "(shape=%s).",
            rolling_corr.shape,
        )

        return rolling_corr

    except Exception as exc:
        logger.exception("Rolling correlation calculation failed.")
        raise RuntimeError(
            "Failed to calculate rolling correlation."
        ) from exc


def calculate_mean_cross_correlation(
    returns_df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """
    Calculate mean cross-correlation across all assets.

    The average correlation is computed using only the
    upper triangle of each rolling correlation matrix,
    excluding the diagonal.

    Args:
        returns_df (pd.DataFrame):
            Asset returns DataFrame.

        window (int, optional):
            Rolling window size.

    Returns:
        pd.Series:
            Mean cross-correlation indexed by date.

    Raises:
        TypeError:
            If inputs are invalid.

        ValueError:
            If validation fails.
    """
    logger.info(
        "Calculating mean cross-correlation "
        "(window=%d).",
        window,
    )

    rolling_corr = calculate_rolling_correlation(
        returns_df,
        window,
    )

    def _get_off_diagonal_mean(
        matrix_df: pd.DataFrame,
    ) -> float:
        """
        Compute mean of upper triangular
        correlation values.
        """
        values = matrix_df.values

        upper = np.triu_indices_from(values, k=1)

        if len(upper[0]) == 0:
            return np.nan

        off_diag_values = values[upper]

        # Remove NaN values from incomplete rolling windows
        off_diag_values = off_diag_values[~np.isnan(off_diag_values)]

        if off_diag_values.size == 0:
            return np.nan

        return float(np.mean(off_diag_values))

    try:
        mean_corr = rolling_corr.groupby(
            level=0,
            group_keys=False,
        ).apply(_get_off_diagonal_mean)

        mean_corr.name = "Mean_Cross_Correlation"

        mean_corr = mean_corr.dropna()

        logger.info(
            "Mean cross-correlation calculated "
            "successfully (%d observations).",
            len(mean_corr),
        )

        return mean_corr

    except Exception as exc:
        logger.exception(
            "Mean cross-correlation calculation failed."
        )
        raise RuntimeError(
            "Failed to calculate mean cross-correlation."
        ) from exc
