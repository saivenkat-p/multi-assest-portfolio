"""
ARQPO: Adaptive Regime-Aware Quantum Portfolio Optimizer
WISER × Vanguard Quantum Challenge 2026

Module: 09_streamlit_dashboard.dashboard
Main Streamlit application entry point and dynamic page router.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path for backend imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ARQPO.Dashboard")

# Import backend modules safely using importlib (handles folder names with numerical prefixes)
import importlib
try:
    DataLoader = importlib.import_module("01_data_preprocessing.data_loader").DataLoader
except Exception as e:
    logger.warning(f"Could not dynamically import DataLoader: {e}")
    DataLoader = None



def setup_page_config() -> None:
    """Configures the primary Streamlit page settings and layout."""
    st.set_page_config(
        page_title="ARQPO | Quantum Portfolio Optimizer",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    """Injects custom CSS for a research-grade institutional design system."""
    st.markdown(
        """
        <style>
        /* Modern Glassmorphism & Color Palette */
        :root {
            --bg-primary: #0A192F;
            --bg-card: #112240;
            --accent-cyan: #06B6D4;
            --accent-blue: #3B82F6;
            --text-bright: #F8FAFC;
            --text-muted: #94A3B8;
        }

        /* Top Banner Header */
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0284C7 100%);
            padding: 1.8rem 2rem;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .main-header h1 {
            color: #FFFFFF !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            letter-spacing: -0.5px;
        }
        .main-header p {
            color: #93C5FD !important;
            font-size: 1.05rem !important;
            margin-top: 0.3rem !important;
            margin-bottom: 0 !important;
        }

        /* Metric Cards Styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #0284C7 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 600 !important;
            color: #475569 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Sidebar Custom Styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
        }
        section[data-testid="stSidebar"] .stMarkdown h2, 
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #F8FAFC !important;
        }

        /* Regime Badges */
        .badge-bull {
            background-color: #10B981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-bear {
            background-color: #EF4444;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-volatility {
            background-color: #F59E0B;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .badge-stable {
            background-color: #3B82F6;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading historical financial datasets...")
def load_global_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and caches historical price data, asset metadata, and scenarios.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (prices, metadata, scenarios)
    """
    try:
        data_dir = PROJECT_ROOT / "data"
        loader = DataLoader(data_dir=data_dir)
        prices = loader.load_historical_prices()
        metadata = loader.load_asset_metadata()
        scenarios = loader.load_market_scenarios()
        logger.info("Successfully loaded global datasets.")
        return prices, metadata, scenarios
    except Exception as e:
        logger.error(f"Error loading global datasets: {e}")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df


def initialize_session_state(prices: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Initializes global Streamlit session state variables."""
    if "prices" not in st.session_state:
        st.session_state["prices"] = prices
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = metadata
    if "selected_assets" not in st.session_state and not prices.empty:
        st.session_state["selected_assets"] = list(prices.columns[:5])
    if "gamma" not in st.session_state:
        st.session_state["gamma"] = 1.0
    if "quantum_backend" not in st.session_state:
        st.session_state["quantum_backend"] = "aer_simulator"
    if "n_qubits" not in st.session_state:
        st.session_state["n_qubits"] = 5
    if "current_regime" not in st.session_state:
        st.session_state["current_regime"] = "Bull"


def render_header() -> None:
    """Renders the top institutional application header banner."""
    st.markdown(
        """
        <div class="main-header">
            <h1>Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)</h1>
            <p>WISER × Vanguard Quantum Challenge 2026 | Hybrid Quantum-Classical Portfolio Framework</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """
    Renders the persistent navigation menu and global configuration parameters.

    Returns:
        str: Selected page name.
    """
    st.sidebar.title("🌌 ARQPO Navigation")
    st.sidebar.markdown("---")

    pages = [
        "Market Regime View",
        "Quantum Portfolio View",
        "Classical Comparison",
        "Explainable AI Copilot",
    ]

    selected_page = st.sidebar.radio(
        "Select Dashboard Module",
        pages,
        index=0,
        help="Switch between regime analysis, quantum allocation, baseline benchmark, and XAI copilot.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Global Parameters")

    # Risk Aversion Parameter Slider
    gamma = st.sidebar.slider(
        "Risk Aversion (γ)",
        min_value=0.1,
        max_value=5.0,
        value=st.session_state.get("gamma", 1.0),
        step=0.1,
        help="Higher values prioritize risk minimization over return maximization.",
    )
    st.session_state["gamma"] = gamma

    # Quantum Backend Selector
    backend = st.sidebar.selectbox(
        "Quantum Execution Engine",
        options=["aer_simulator", "fake_santiago", "fake_jakarta", "statevector_simulator"],
        index=0,
        help="Select Qiskit Aer simulator or noisy quantum hardware emulator.",
    )
    st.session_state["quantum_backend"] = backend

    # Number of Qubits / Assets
    prices = st.session_state.get("prices", pd.DataFrame())
    max_assets = len(prices.columns) if not prices.empty else 5
    n_qubits = st.sidebar.slider(
        "Qubit Asset Scope (N)",
        min_value=2,
        max_value=min(max_assets, 10),
        value=min(st.session_state.get("n_qubits", 5), max_assets),
        step=1,
        help="Number of portfolio assets mapped to quantum qubits in QUBO Hamiltonian.",
    )
    st.session_state["n_qubits"] = n_qubits

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**WISER × Vanguard Challenge 2026**\n\n"
        "Status: 🟢 Backend Modules 01–08 Active\n\n"
        "Architecture: QAOA & VQE Hybrid Solver"
    )

    return selected_page


def route_page(page_name: str) -> None:
    """
    Dynamically imports and renders the selected sub-page module.

    Args:
        page_name (str): Name of the active sidebar selection.
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

    except ModuleNotFoundError as err:
        logger.warning(f"Sub-module loading note for {page_name}: {err}")
        st.warning(f"🚧 **{page_name}** module interface is initializing.")
        st.info("Generating next component according to project structure...")
    except Exception as e:
        logger.error(f"Unexpected error rendering {page_name}: {e}", exc_info=True)
        st.error(f"An unexpected UI error occurred while rendering {page_name}: {str(e)}")


def main() -> None:
    """Main application execution pipeline."""
    setup_page_config()
    inject_custom_css()

    # Load global datasets
    prices, metadata, scenarios = load_global_data()
    initialize_session_state(prices, metadata)

    # Render persistent layout components
    render_header()
    selected_page = render_sidebar()

    # Route to selected page module
    route_page(selected_page)


if __name__ == "__main__":
    main()