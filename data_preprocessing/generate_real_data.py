import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    # Target the root data directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    tickers = ["AAPL", "MSFT", "JPM", "GLD", "TLT"]
    
    print("Fetching historical prices...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Changed from 'Adj Close' to 'Close' to prevent KeyError in newer yfinance versions
    df_prices = yf.download(tickers, start=start_date, end=end_date, threads=False)['Close']
    df_prices.dropna(inplace=True)
    df_prices.index.name = "Date"
    df_prices.to_csv(os.path.join(data_dir, "historical_prices.csv"))
    print(f"Saved historical_prices.csv ({len(df_prices)} rows)")
    
    metadata = {
        "Ticker": ["AAPL", "MSFT", "JPM", "GLD", "TLT"],
        "Asset_Class": ["Equity", "Equity", "Equity", "Commodity", "Fixed Income"],
        "Sector": ["Technology", "Technology", "Financials", "Precious Metals", "Government Bonds"],
        "Risk_Profile": ["High", "High", "Medium", "Medium", "Low"]
    }
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(os.path.join(data_dir, "asset_metadata.csv"), index=False)
    print("Saved asset_metadata.csv")
    
    print("Generating synthetic assets...")
    np.random.seed(42)
    synthetic_returns = np.random.normal(0.0005, 0.02, size=(len(df_prices), 3))
    df_synthetic = pd.DataFrame(
        synthetic_returns, 
        index=df_prices.index, 
        columns=["SYN_HIGH_VOL", "SYN_STABLE", "SYN_BEAR"]
    )
    df_synthetic = 100 * (1 + df_synthetic).cumprod()
    df_synthetic.to_csv(os.path.join(data_dir, "synthetic_assets.csv"))
    print("Saved synthetic_assets.csv")
    
    scenarios = {
        "Regime": ["Bull", "Bear", "High Volatility", "Stable"],
        "Start_Date": ["2020-04-01", "2020-02-19", "2022-01-01", "2023-01-01"],
        "End_Date": ["2021-12-31", "2020-03-31", "2022-12-31", "2023-12-31"],
        "Description": [
            "Post-COVID recovery rally",
            "COVID-19 initial market crash",
            "Inflation and rate hike turbulence",
            "Normalized market recovery"
        ]
    }
    df_scenarios = pd.DataFrame(scenarios)
    df_scenarios.to_csv(os.path.join(data_dir, "market_scenarios.csv"), index=False)
    print("Saved market_scenarios.csv")
    
    print("\nData generation complete. All files populated in the root /data/ directory.")

if __name__ == "__main__":
    generate_data()  