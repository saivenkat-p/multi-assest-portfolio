"""
Module: 01_data_preprocessing.data_loader
Provides the DataLoader class to ingest and clean financial datasets.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataLoader:
    """
    Ingests and validates CSV data files for portfolio optimization.
    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        """
        Initializes the DataLoader.

        Args:
            data_dir (Union[str, Path]): Path to the directory containing CSV files.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory '{self.data_dir}' does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_csv(self, filename: str, index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Internal method to load a CSV file safely.

        Args:
            filename (str): Name of the CSV file.
            index_col (Optional[str]): Column to use as the DataFrame index.

        Returns:
            pd.DataFrame: Loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or cannot be parsed.
        """
        file_path = self.data_dir / filename
        
        if not file_path.is_file():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Missing required data file: {file_path}")

        try:
            df = pd.read_csv(file_path, index_col=index_col)
            if df.empty:
                logger.warning(f"Loaded file '{filename}' is empty.")
            return df
        except Exception as e:
            logger.error(f"Error parsing {filename}: {e}")
            raise ValueError(f"Failed to read {filename}. Ensure it is a valid CSV.") from e

    def load_historical_prices(self, filename: str = "historical_prices.csv") -> pd.DataFrame:
        """
        Loads and cleans historical asset price data.

        Args:
        filename (str): Name of the historical prices CSV.

        Returns:
        pd.DataFrame: Cleaned price data with DatetimeIndex and no NaNs.
        """
        logger.info(f"Loading historical prices from {filename}")
        df = self._load_csv(filename, index_col=None)

        # Ensure a date column exists
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if not date_cols:
            raise ValueError("No recognizable date column found in price data.")

        date_col = date_cols[0]

    # Convert dates (invalid dates become NaT)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            logger.warning(
                "Removed %d rows with invalid dates.",
                invalid_dates,
            )
            df = df.dropna(subset=[date_col])

        # Set date as index
        df.set_index(date_col, inplace=True)

        # NEW: Remove duplicate timestamps
        duplicate_rows = df.index.duplicated().sum()
        if duplicate_rows > 0:
            logger.warning(
                "Removed %d duplicate timestamps.",
                duplicate_rows,
            )
            df = df[~df.index.duplicated(keep="first")]

        # NEW: Convert all price columns to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Sort chronologically
        df.sort_index(inplace=True)

        # NEW: Verify chronological order
        if not df.index.is_monotonic_increasing:
            raise ValueError("Price index is not sorted.")

        # Handle missing data
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing values in price data.")
            df = df.ffill().bfill()

        # NEW: Remove columns that are entirely NaN
        empty_columns = df.columns[df.isna().all()]
        if len(empty_columns) > 0:
            logger.warning(
                "Dropping empty asset columns: %s",
                list(empty_columns),
            )
            df = df.drop(columns=empty_columns)

        return df
        

    def load_asset_metadata(self, filename: str = "asset_metadata.csv") -> pd.DataFrame:
        """
        Loads asset metadata (tickers, sectors, constraints).

        Args:
            filename (str): Name of the metadata CSV.

        Returns:
            pd.DataFrame: Metadata DataFrame.
        """
        logger.info(f"Loading asset metadata from {filename}")
        return self._load_csv(filename)

    def load_synthetic_assets(self, filename: str = "synthetic_assets.csv") -> pd.DataFrame:
        """
        Loads synthetic asset data for quantum algorithm benchmarking.

        Args:
            filename (str): Name of the synthetic assets CSV.

        Returns:
            pd.DataFrame: Cleaned synthetic price data.
        """
        logger.info(f"Loading synthetic assets from {filename}")
        # Re-use historical price logic since synthetic data is structurally identical
        return self.load_historical_prices(filename=filename)
        
    def load_market_scenarios(self, filename: str = "market_scenarios.csv") -> pd.DataFrame:
        """
        Loads predefined market regime scenarios.

        Args:
            filename (str): Name of the scenarios CSV.

        Returns:
            pd.DataFrame: Market scenarios DataFrame.
        """
        logger.info(f"Loading market scenarios from {filename}")
        return self._load_csv(filename)