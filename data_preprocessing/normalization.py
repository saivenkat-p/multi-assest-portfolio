"""
Module: 01_data_preprocessing.normalization
Provides functions to scale and normalize time-series financial data.
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


def z_score_normalize(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Z-score standardization across the columns of the DataFrame.
    Centers data to mean=0 and std=1.

    Args:
        data_df (pd.DataFrame): The raw financial data (e.g., returns, volume).

    Returns:
        pd.DataFrame: Z-score normalized DataFrame.

    Raises:
        TypeError:
            If data_df is not a pandas DataFrame.
        ValueError:
            If data_df is empty.
        TypeError:
            If any column is non-numeric.
    """
    logger.info("Applying Z-score normalization.")

    # Validate input type
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame.")

    # Validate non-empty DataFrame
    if data_df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Validate numeric columns
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data_df.dtypes):
        raise TypeError("All columns must be numeric.")

    # Calculate column-wise mean and standard deviation
    mean = data_df.mean()
    std = data_df.std()

    # Handle edge case where standard deviation is zero (constant values)
    std = std.replace(0, 1e-8)

    normalized_df = (data_df - mean) / std
    return normalized_df


def min_max_normalize(
    data_df: pd.DataFrame,
    feature_range: tuple = (0, 1)
) -> pd.DataFrame:
    """
    Applies Min-Max scaling to map data within a specific feature range.

    Args:
        data_df (pd.DataFrame): The raw financial data.
        feature_range (tuple): Desired (min, max) bounds for the scaled data.

    Returns:
        pd.DataFrame: Min-Max scaled DataFrame.

    Raises:
        TypeError:
            If data_df is not a pandas DataFrame.
        ValueError:
            If data_df is empty.
        TypeError:
            If any column is non-numeric.
        ValueError:
            If feature_range is invalid.
    """
    logger.info(f"Applying Min-Max normalization with range {feature_range}.")

    # Validate input type
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame.")

    # Validate non-empty DataFrame
    if data_df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Validate numeric columns
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data_df.dtypes):
        raise TypeError("All columns must be numeric.")

    # Validate feature_range
    if (
        not isinstance(feature_range, tuple)
        or len(feature_range) != 2
    ):
        raise ValueError(
            "feature_range must be a tuple of length 2."
        )

    lower_bound, upper_bound = feature_range

    if lower_bound >= upper_bound:
        raise ValueError(
            "feature_range minimum must be less than maximum."
        )

    # Calculate column-wise min and max
    col_min = data_df.min()
    col_max = data_df.max()

    # Prevent division by zero if min == max
    range_diff = (col_max - col_min).replace(0, 1e-8)

    # Scale to [0, 1]
    scaled_0_1 = (data_df - col_min) / range_diff

    # Scale to target range
    normalized_df = (
        scaled_0_1 * (upper_bound - lower_bound)
        + lower_bound
    )

    return normalized_df
