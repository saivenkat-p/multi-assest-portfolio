"""
Tests for market regime detection features (volatility and correlation).
"""

import numpy as np
import pandas as pd
import pytest

from market_regime_detection.volatility import (
    calculate_rolling_volatility,
    calculate_ewma_volatility,
)
from market_regime_detection.correlation import (
    calculate_rolling_correlation,
    calculate_mean_cross_correlation,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    data = {
        "Asset1": np.random.normal(0.001, 0.02, 100),
        "Asset2": np.random.normal(0.0005, 0.015, 100),
        "Asset3": np.random.normal(0.0015, 0.025, 100),
    }
    return pd.DataFrame(data, index=dates)


def test_rolling_volatility(sample_returns):
    vol = calculate_rolling_volatility(sample_returns, window=20)
    assert isinstance(vol, pd.DataFrame)
    assert vol.shape[1] == 3
    valid_vol = vol.dropna()
    assert (valid_vol > 0).all().all()


def test_ewma_volatility(sample_returns):
    ewma_vol = calculate_ewma_volatility(sample_returns, span=20)
    assert isinstance(ewma_vol, pd.DataFrame)
    assert ewma_vol.shape[1] == sample_returns.shape[1]
    assert (ewma_vol > 0).all().all()


def test_rolling_correlation(sample_returns):
    corr = calculate_rolling_correlation(sample_returns, window=30)
    assert isinstance(corr, pd.DataFrame)
    valid_corr = corr.dropna()
    assert (valid_corr >= -1.0 - 1e-6).all().all()
    assert (valid_corr <= 1.0 + 1e-6).all().all()


def test_mean_cross_correlation(sample_returns):
    mean_corr = calculate_mean_cross_correlation(sample_returns, window=30)
    assert isinstance(mean_corr, pd.Series)
    valid_mean = mean_corr.dropna()
    assert (valid_mean >= -1.0 - 1e-6).all()
    assert (valid_mean <= 1.0 + 1e-6).all()
