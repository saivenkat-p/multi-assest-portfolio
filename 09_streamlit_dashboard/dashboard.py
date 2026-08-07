"""
Main Streamlit Dashboard application for ARQPO.

This module serves as the primary router and user interface for the 
Adaptive Regime-Aware Quantum Portfolio Optimizer. It handles sidebar 
navigation and dynamic rendering of sub-pages.
"""

import logging
import streamlit as st
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

# Ensure the project root is in the Python path for relative imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_page_config() -> None:
    """Configures the primary Streamlit page settings."""
    st.set_page_config(
        page_title="ARQPO | Quantum Portfolio Optimizer",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar() -> str:
    """
    Renders the sidebar navigation menu.
    
    Returns:
        str: The name of the selected page.
    """
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Qubit.png/240px-Qubit.png", width=100)
    st.sidebar.title("ARQPO Navigation")
    st.sidebar.markdown("---")
    
    pages = [
        "Market Regime View",
        "Quantum Portfolio View",
        "Classical Comparison",
        "Explainable AI Copilot"
    ]
    
    selection = st.sidebar.radio("Go to", pages)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**ARQPO**\n\n"
        "Adaptive Regime-Aware Quantum Portfolio Optimizer\n\n"
        "WISER × Vanguard Quantum Challenge 2026"
    )
    
    return selection

def load_page(page_name: str) -> None:
    """
    Dynamically loads and renders the selected page.
    Gracefully handles MissingModule errors during incremental development.
    
    Args:
        page_name (str): The name of the page selected from the sidebar.
    """
    try:
        if page_name == "Market Regime View":
            from market_view import render as render_market
            render_market()
            
        elif page_name == "Quantum Portfolio View":
            from portfolio_view import render as render_portfolio
            render_portfolio()
            
        elif page_name == "Classical Comparison":
            from comparison_page import render as render_comparison
            render_comparison()
            
        elif page_name == "Explainable AI Copilot":
            from copilot import render as render_copilot
            render_copilot()
            
    except ImportError as e:
        logger.warning(f"Module for '{page_name}' not yet implemented. Error: {e}")
        st.warning(f"🚧 **{page_name}** is currently under construction.")
        st.info("The required module has not been generated yet. Please proceed with the development order.")
    except Exception as e:
        logger.error(f"Error rendering {page_name}: {str(e)}")
        st.error(f"An error occurred while loading the page: {str(e)}")

def main() -> None:
    """
    Main execution function for the Streamlit dashboard.
    """
    setup_page_config()
    
    # Render Sidebar and get user selection
    selected_page = render_sidebar()
    
    # Log the navigation event
    logger.info(f"User navigated to {selected_page}")
    
    # Render the selected page body
    load_page(selected_page)

if __name__ == "__main__":
    main()