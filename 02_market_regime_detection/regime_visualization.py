"""
Module: 02_market_regime_detection.regime_visualization
Generates plots overlaying detected market regimes on financial time series.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def plot_regimes(
    time_series: pd.Series, 
    regime_labels: pd.Series, 
    title: str = "Market Regimes Overlaid on Asset Performance",
    color_map: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Plots a time series with background shading corresponding to market regimes.

    Args:
        time_series (pd.Series): The financial time series to plot (e.g., price or cumulative return).
        regime_labels (pd.Series): The semantic labels of the market regimes.
        title (str): Title of the plot.
        color_map (Dict[str, str], optional): Custom mapping of regimes to colors.

    Returns:
        plt.Figure: The constructed matplotlib figure.
    """
    logger.info("Generating regime visualization plot.")
    
    # Ensure inputs are aligned
    df = pd.concat([time_series, regime_labels], axis=1).dropna()
    ts_col = df.columns[0]
    regime_col = df.columns[1]
    
    if color_map is None:
        color_map = {
            "Bull": "green",
            "Bear": "red",
            "Stable": "blue",
            "High Volatility": "orange"
        }

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Plot the main time series
    ax.plot(df.index, df[ts_col], color='black', linewidth=1.5, label='Market Proxy')
    
    # 2. Identify contiguous regime periods for shading
    # Shift regime labels to find points where the regime changes
    df['Regime_Shift'] = df[regime_col] != df[regime_col].shift(1)
    
    # Create block IDs to group contiguous periods
    df['Block_ID'] = df['Regime_Shift'].cumsum()
    
    # 3. Iterate through blocks and shade the background
    for _, group in df.groupby('Block_ID'):
        start_date = group.index[0]
        end_date = group.index[-1]
        regime = group[regime_col].iloc[0]
        
        color = color_map.get(regime, 'grey')
        
        ax.axvspan(start_date, end_date, color=color, alpha=0.2, lw=0)
        
    # 4. Create custom legend for the regime shadings
    legend_patches = [
        mpatches.Patch(color=color, alpha=0.3, label=regime)
        for regime, color in color_map.items() if regime in df[regime_col].unique()
    ]
    
    # Add the line plot to the legend
    line_patch = plt.Line2D([0], [0], color='black', lw=1.5, label=ts_col)
    legend_patches.insert(0, line_patch)
    
    ax.legend(handles=legend_patches, loc='upper left')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel("Value")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    logger.info("Visualization generated successfully.")
    
    return fig