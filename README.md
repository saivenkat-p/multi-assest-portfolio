# ARQPO: Adaptive Regime-Aware Quantum Portfolio Optimizer
### Enterprise Quantum-Classical Portfolio Optimization via Xanadu PennyLane

[![PennyLane](https://img.shields.io/badge/PennyLane-v0.45+-blue.svg)](https://pennylane.ai/)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(12%20Suites)-success.svg)](tests/)

---

## 1. Overview

The **Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)** is a production-grade hybrid quantum-classical financial optimization platform. ARQPO bridges macroeconomic statistical regime classification with combinatorial portfolio selection using variational quantum computing.

Financial portfolio selection is formulated as a continuous Markowitz mean-variance optimization problem, discretized using binary encoding into a **Quadratic Unconstrained Binary Optimization (QUBO)** model, algebraically converted to an exact **Ising spin Hamiltonian**, and solved variationally using **Xanadu PennyLane** implementations of:
1. **QAOA** (Quantum Approximate Optimization Algorithm)
2. **VQE** (Variational Quantum Eigensolver with Hardware-Efficient Ansätze)

---

## 2. System Architecture

```
Market Data (yfinance / historical prices)
        │
        ▼
Data Preprocessing (Returns, Shrinkage Covariance, EWMA Volatility)
        │
        ▼
Market Regime Detection (Hidden Markov / Statistical State Classifier)
        │
        ▼
Adaptive Parameter Tuning (Regime-aware risk-aversion γ and budget penalty λ)
        │
        ▼
QUBO Formulation (Discrete weight expansion w_i = scale * Σ 2^k x_{i,k})
        │
        ▼
PennyLane Ising Hamiltonian Translation (x_i = (I - Z_i)/2, Exact Energy Offset)
        ├──► PennyLane QAOA Solver (Parameterized Cost & Mixer Layers)
        └──► PennyLane VQE Solver (Hardware-Efficient Ry/Rz Ansatz + CNOT Entanglement)
        │
        ▼
PennyLane Execution Devices (default.qubit statevector / lightning.qubit)
        │
        ▼
Measurement Probability Spectrum & Bitstring Decoding -> Simplex Normalized Weights
        │
        ▼
Performance Evaluation (Sharpe, Drawdown, Turnover) & Interactive Streamlit Dashboard
```

---

## 3. Mathematical Foundations

### 3.1 Mean-Variance Markowitz Formulation
$$\min_{w \in \Delta^N} \frac{1}{2} w^T \Sigma w - \gamma \mu^T w + \lambda \left(\sum_{i=0}^{N-1} w_i - 1\right)^2$$
where $\mu$ is the annualized expected return vector, $\Sigma$ is the asset covariance matrix, $\gamma$ is risk aversion, and $\lambda$ is the investment budget penalty coefficient.

### 3.2 Discrete Binary Encoding
Each asset weight $w_i$ is discretized using $B$ binary variables $x_{i,k} \in \{0, 1\}$:
$$w_i = \frac{1}{2^B - 1} \sum_{k=0}^{B-1} 2^k x_{i,k}$$
Total qubits: $M = N \times B$.

The resulting objective is:
$$E(x) = x^T Q x + \lambda$$

### 3.3 Exact QUBO-to-Ising Spin Transformation
Binary decision variables $x_i \in \{0, 1\}$ are mapped to Pauli-Z spin operators $Z_i \in \{+1, -1\}$ via:
$$x_i = \frac{I - Z_i}{2}$$

Substituting into $x^T Q x + \lambda$ yields the exact Ising spin Hamiltonian:
$$\hat{H}_C = c_0 I + \sum_{i=0}^{M-1} h_i Z_i + \sum_{0 \le i < j < M} J_{ij} Z_i Z_j$$
where:
- $J_{ij} = \frac{1}{2} Q_{ij}$ (quadratic couplings)
- $h_i = -\frac{1}{2} \sum_{j=0}^{M-1} Q_{ij}$ (linear fields)
- $c_0 = \lambda + \frac{1}{4} \sum_{i,j} Q_{ij} + \frac{1}{4} \sum_i Q_{ii}$ (exact constant energy offset)

**Energy Equivalence Guarantee:** For every computational basis state $|b\rangle$, $\langle b | \hat{H}_C | b \rangle \equiv b^T Q b + \lambda$ within $10^{-10}$ numerical precision.

---

## 4. PennyLane Implementation Details

### 4.1 QAOA Architecture (`quantum_algorithms/qaoa_solver.py`)
- **Initial State:** $|+\rangle^{\otimes M} = \bigotimes_{i=0}^{M-1} H |0\rangle$
- **Cost Layer Unitary:** $U_C(\gamma) = \exp(-i \gamma \hat{H}_C)$ implemented via single-qubit $R_Z(2 \gamma h_i)$ rotations and two-qubit $\text{IsingZZ}(2 \gamma J_{ij})$ interaction gates.
- **Mixer Layer Unitary:** $U_M(\beta) = \exp(-i \beta \sum X_i)$ implemented via single-qubit $R_X(2 \beta)$ gates.
- **QNodes:** Differentiable expectation value evaluation and computational basis state probability measurement.

### 4.2 VQE Architecture (`quantum_algorithms/vqe_solver.py` & `ansatz.py`)
- **Ansatz:** Parameterized Hardware-Efficient Ansatz (HEA) with $(p+1)$ rotation layers ($R_Y, R_Z$) and $p$ entanglement layers (linear or circular CNOT topologies).
- **Variational Parameters:** $K = 2M(p+1)$ angles optimized using classical (COBYLA, SLSQP) or gradient-based (Adam) optimizers.
- **Solution Decoding:** Computational basis probabilities $P(b) = |\langle b | \psi(\theta^*) \rangle|^2$ are extracted, identifying the ground state bitstring $b^*$ and projecting it to non-negative portfolio weights on the simplex.

---

## 5. Installation and Setup

### 5.1 Prerequisites
- Python 3.10 - 3.12
- Git

### 5.2 Environment Setup
```bash
# Clone the repository
git clone https://github.com/saivenkat-p/multi-assest-portfolio.git
cd multi-assest-portfolio

# Switch to the PennyLane migration branch
git checkout feature/pennylane-migration

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
```

---

## 6. Running the Test Suite

ARQPO includes comprehensive test suites covering mathematical equivalence, quantum solvers, and classical regression:

```bash
# Run the complete test suite
pytest

# Run mathematical QUBO-to-Ising energy equivalence tests
pytest tests/test_energy_equivalence.py -v

# Run PennyLane QAOA and VQE solver tests
pytest tests/test_qaoa.py tests/test_vqe.py -v

# Run backend and optimizer tests
pytest tests/test_backend.py tests/test_optimizer.py -v

# Run end-to-end integration tests
pytest tests/test_integration.py -v
```

---

## 7. Running the Streamlit Dashboard

Launch the interactive multi-page dashboard:
```bash
streamlit run streamlit_dashboard/dashboard.py
```

### Dashboard Modules:
1. **Market Regime View:** Historical asset prices, rolling volatility, correlation heatmaps, and statistical regime detection (Bull, Bear, Stable, High Volatility).
2. **Quantum Portfolio View:** Interactive QAOA/VQE solver console with live variational convergence traces, optimal asset weight donut charts, and measurement probability spectrum.
3. **Classical Comparison:** Side-by-side benchmark against Markowitz Mean-Variance, SciPy SLSQP, and Equal Weight baselines.
4. **Explainable AI Copilot:** Natural language portfolio rationales, risk-return trade-off analysis, and automated investment memo generation.

---

## 8. Authors & Acknowledgments

- **Pendyala Sai Venkat** ([@saivenkat-p](https://github.com/saivenkat-p))
- **Saladi Ganesh**

Developed as a flagship engineering project for the **Xanadu Quantum Code Residency Portfolio**, transitioning quantum financial optimization to the native PennyLane software stack.
