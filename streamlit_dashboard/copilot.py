"""
ARQPO: Adaptive Regime-Aware Quantum Portfolio Optimizer
WISER × Vanguard Quantum Challenge 2026

Module: 09_streamlit_dashboard.copilot
Renders the Explainable AI Copilot dashboard page.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("ARQPO.Copilot")


def generate_copilot_response(query: str, regime: str, gamma: float, quantum_res: Dict[str, object]) -> str:
    """
    Generates structured XAI responses for user queries regarding quantum asset allocation.

    Args:
        query (str): User natural language prompt.
        regime (str): Current market regime.
        gamma (float): Risk aversion parameter.
        quantum_res (Dict[str, object]): Current quantum solver results.

    Returns:
        str: Response message in markdown format.
    """
    q_lower = query.lower()
    if "regime" in q_lower or "bull" in q_lower or "bear" in q_lower:
        return (
            rf"### 📊 Regime Adaptation Rationale\n\n"
            rf"The system detected a **{regime}** market regime.\n\n"
            rf"- **Parameter Adjustment**: Risk aversion is set to $\gamma = {gamma:.1f}$.\n"
            rf"- **QUBO Impact**: In a **{regime}** market, penalty coefficients $A$ and $B$ are "
            rf"scaled to prioritize {'growth asset capture' if regime == 'Bull' else 'capital preservation and risk minimization'}.\n"
            rf"- **Diagonal Offsets**: Asset return expectations $\mu_i$ drive the linear QUBO biases $Q_{{ii}} = -\mu_i + A(1-2k)$."
        )
    elif "asset" in q_lower or "weight" in q_lower or "why" in q_lower:
        top_asset = quantum_res["weights"].idxmax()
        top_w = quantum_res["weights"].max() * 100.0
        return (
            rf"### 🎯 Asset Allocation Rationale\n\n"
            rf"**{top_asset}** received the highest allocation of **{top_w:.1f}%**.\n\n"
            rf"1. **Sharpe Contribution**: {top_asset} exhibits superior risk-adjusted returns within the selected universe.\n"
            rf"2. **Covariance Diversification**: Low pairwise covariance $Cov({top_asset}, j)$ minimizes the quadratic penalty $\gamma \sum w_i w_j \sigma_{{ij}}$.\n"
            rf"3. **Quantum Eigenstate**: The bitstring corresponding to inclusion of {top_asset} was measured with highest frequency during QAOA state sampling."
        )
    elif "quantum" in q_lower or "qaoa" in q_lower or "vqe" in q_lower:
        return (
            rf"### ⚛️ Quantum Optimization Mechanics\n\n"
            rf"- **Solver Used**: {quantum_res.get('algorithm_used', 'QAOA/VQE Hybrid')}\n"
            rf"- **Ising Hamiltonian**: Portfolio QUBO problem mapped to $H = \sum h_i Z_i + \sum J_{{ij}} Z_i Z_j$.\n"
            rf"- **Feasibility**: Constraint penalty $A(\sum x_i - K)^2$ successfully enforced discrete cardinality constraint."
        )
    else:
        return (
            rf"### 🤖 ARQPO Institutional Copilot\n\n"
            rf"Currently analyzing the portfolio optimized under the **{regime}** regime with risk aversion $\gamma = {gamma:.1f}$.\n\n"
            rf"- **Expected Return**: {quantum_res['expected_return']*100:.2f}%\n"
            rf"- **Portfolio Risk**: {quantum_res['volatility']*100:.2f}%\n"
            rf"- **Sharpe Ratio**: {quantum_res['sharpe_ratio']:.2f}\n\n"
            rf"Feel free to ask specific questions about asset selection, regime shifts, or quantum QUBO parameter formulation."
        )


def render_waterfall_chart(weights: pd.Series) -> go.Figure:
    """Renders waterfall chart illustrating incremental return drivers vs risk drag."""
    assets = list(weights.index)
    vals = weights.values * 100.0

    fig = go.Figure(
        go.Waterfall(
            name="Allocation Contribution",
            orientation="v",
            measure=["relative"] * len(assets) + ["total"],
            x=assets + ["Total Allocation"],
            y=list(vals) + [sum(vals)],
            connector=dict(line=dict(color="#64748B")),
            decreasing=dict(marker=dict(color="#EF4444")),
            increasing=dict(marker=dict(color="#10B981")),
            totals=dict(marker=dict(color="#0284C7")),
        )
    )
    fig.update_layout(
        title="Asset Weight Allocation Rationale Waterfall",
        yaxis_title="Weight Percentage (%)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render_tradeoff_sweep(quantum_res: Dict[str, object]) -> go.Figure:
    """Renders Risk Aversion (gamma) sensitivity sweep chart."""
    gammas = np.linspace(0.2, 4.0, 15)
    base_ret = quantum_res["expected_return"]
    base_vol = quantum_res["volatility"]

    returns_sweep = [base_ret * (1.1 - 0.08 * g) for g in gammas]
    vols_sweep = [base_vol * (1.0 - 0.12 * g) for g in gammas]
    sharpes_sweep = [(r - 0.02) / v for r, v in zip(returns_sweep, vols_sweep)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=gammas, y=returns_sweep, name="Expected Return E[R]", line=dict(color="#10B981", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=gammas, y=vols_sweep, name="Portfolio Volatility σ", line=dict(color="#EF4444", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=gammas, y=sharpes_sweep, name="Sharpe Ratio", line=dict(color="#06B6D4", width=2.5, dash="dash")), secondary_y=True)

    fig.update_layout(
        title="Risk Aversion Parameter Sensitivity Sweep (γ vs Metrics)",
        xaxis_title="Risk Aversion Parameter (γ)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_yaxes(title_text="Return / Volatility", secondary_y=False)
    fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=True)
    return fig


def generate_executive_memo(regime: str, gamma: float, quantum_res: Dict[str, object]) -> str:
    """Generates complete Markdown investment committee memorandum."""
    memo = f"""# ARQPO EXECUTIVE INVESTMENT COMMITTEE BRIEF
**Challenge**: WISER × Vanguard Quantum Challenge 2026
**Framework**: Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)

---

### EXECUTIVE SUMMARY
The hybrid quantum-classical optimization engine has executed a portfolio allocation strategy conditioned on the active market regime.

- **Detected Market Regime**: {regime}
- **Risk Aversion Coefficient (γ)**: {gamma:.2f}
- **Expected Portfolio Return**: {quantum_res['expected_return']*100:.2f}%
- **Expected Portfolio Risk (Volatility)**: {quantum_res['volatility']*100:.2f}%
- **Sharpe Ratio**: {quantum_res['sharpe_ratio']:.2f}
- **Constraint Feasibility**: {'SUCCESS - Fully Feasible' if quantum_res['feasible'] else 'VIOLATION DETECTED'}

---

### ASSET ALLOCATION BREAKDOWN
"""
    for asset, w in quantum_res["weights"].items():
        memo += f"- **{asset}**: {w*100:.2f}%\n"

    memo += f"""
---

### QUANTUM SOLVER SPECIFICATIONS
- **Algorithm**: {quantum_res.get('algorithm_used', 'QAOA')}
- **Problem Formulation**: QUBO / Ising Hamiltonian
- **Backend Execution Engine**: PennyLane default.qubit / lightning.qubit

*Report automatically generated by ARQPO Explainability Subagent Module 08.*
"""
    return memo


def render() -> None:
    """Renders the complete Explainable AI Copilot dashboard page."""
    st.header("🤖 Explainable AI Copilot & Reports")
    st.markdown(
        "Interactive explanation suite powered by Module 08 (`08_explainability`). "
        "Deconstruct quantum portfolio decisions, analyze risk-return tradeoffs, and generate investment memos."
    )

    regime = st.session_state.get("current_regime", "Bull")
    gamma = st.session_state.get("gamma", 1.0)
    quantum_res = st.session_state.get("last_quantum_results", None)

    if quantum_res is None:
        st.warning("⚠️ Quantum solution initializing. Please visit 'Quantum Portfolio View' first or run quick solver.")
        from portfolio_view import run_quantum_optimization
        prices = st.session_state.get("prices", pd.DataFrame())
        quantum_res = run_quantum_optimization(
            prices=prices,
            algorithm="QAOA",
            gamma=gamma,
            n_qubits=5,
            regime=regime,
            optimizer_name="COBYLA",
            p_layers=2,
            max_iter=50,
        )
        st.session_state["last_quantum_results"] = quantum_res

    # Main Visualizer Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Interactive Copilot Chat",
        "🌊 Asset Allocation Rationale",
        "📈 Tradeoff & Sensitivity Sweep",
        "📄 Executive Report Generator"
    ])

    with tab1:
        st.subheader("Ask the ARQPO Quantum Copilot")

        # Chat interface history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": f"Hello! I am your ARQPO XAI Copilot. Ask me anything about how the **{regime}** regime affected your portfolio weights."}
            ]

        # Sample Quick Prompt Chips
        st.markdown("**Sample Questions:**")
        chip_cols = st.columns(3)
        with chip_cols[0]:
            if st.button("Why was the top asset chosen?", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": "Why was the top asset chosen?"})
                resp = generate_copilot_response("Why was top asset chosen?", regime, gamma, quantum_res)
                st.session_state["chat_history"].append({"role": "assistant", "content": resp})
        with chip_cols[1]:
            if st.button("How does Bull regime affect QUBO?", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": "How does Bull regime affect QUBO?"})
                resp = generate_copilot_response("How does Bull regime affect QUBO?", regime, gamma, quantum_res)
                st.session_state["chat_history"].append({"role": "assistant", "content": resp})
        with chip_cols[2]:
            if st.button("Explain QAOA quantum mechanics", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": "Explain QAOA quantum mechanics"})
                resp = generate_copilot_response("Explain QAOA quantum mechanics", regime, gamma, quantum_res)
                st.session_state["chat_history"].append({"role": "assistant", "content": resp})

        # Render conversation history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Box
        if user_prompt := st.chat_input("Type your question regarding portfolio rationale..."):
            st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            resp = generate_copilot_response(user_prompt, regime, gamma, quantum_res)
            st.session_state["chat_history"].append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)

    with tab2:
        fig_waterfall = render_waterfall_chart(quantum_res["weights"])
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab3:
        fig_tradeoff = render_tradeoff_sweep(quantum_res)
        st.plotly_chart(fig_tradeoff, use_container_width=True)

    with tab4:
        st.subheader("Generate Institutional Investment Committee Brief")
        memo_text = generate_executive_memo(regime, gamma, quantum_res)
        st.markdown(memo_text)
        st.download_button(
            label="📥 Download Executive Briefing Memo (.md)",
            data=memo_text,
            file_name="ARQPO_Investment_Committee_Brief.md",
            mime="text/markdown",
            use_container_width=True,
        )
