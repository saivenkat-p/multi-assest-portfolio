# ARQPO Architecture Specification
## Adaptive Regime-Aware Quantum Portfolio Optimizer

**Target Framework:** Xanadu PennyLane  
**Architecture Version:** 2.0.0 (PennyLane-Native)  

---

## 1. System Overview

ARQPO integrates macroeconomic and statistical regime detection with combinatorial portfolio optimization. The continuous portfolio selection problem is discretized, formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem, converted to an Ising spin Hamiltonian, and solved using variational quantum algorithms (QAOA and VQE) in PennyLane.

```
+--------------------------------------------------------------------------+
|                            Financial Market Data                         |
|                 (Yahoo Finance / Pre-processed Price Histories)          |
+------------------------------------+-------------------------------------+
                                     |
+------------------------------------v-------------------------------------+
|                           Data Preprocessing                             |
|       - Returns calculation (log & arithmetic)                           |
|       - Covariance matrix estimation & shrinkage                         |
|       - Rolling and EWMA volatility computation                          |
+------------------------------------+-------------------------------------+
                                     |
+------------------------------------v-------------------------------------+
|                        Market Regime Detection                           |
|       - Statistical hidden state classification                          |
|       - Regime mapping: Bull, Bear, Stable, High-Volatility             |
|       - Adaptive risk-aversion (gamma) tuning                            |
+------------------------------------+-------------------------------------+
                                     |
+------------------------------------v-------------------------------------+
|                     QUBO & Hamiltonian Construction                      |
|       - Continuous weight discretization (B bits per asset)              |
|       - Markowitz objective: 0.5 * w^T Sigma w - gamma * mu^T w          |
|       - Budget constraint: lambda * (sum(w_i) - 1)^2                    |
|       - QUBO matrix Q_ij and constant scalar offset lambda              |
+------------------------------------+-------------------------------------+
                                     |
+------------------------------------v-------------------------------------+
|                  PennyLane Hamiltonian Mapping Layer                     |
|       - Mapping: x_i = (I - Z_i) / 2                                     |
|       - Cost Hamiltonian H_C = c0*I + sum(h_i*Z_i) + sum(J_ij*Z_i*Z_j)   |
|       - Mixer Hamiltonian H_M = sum(X_i)                                 |
+------------------+------------------------------------+------------------+
                   |                                    |
+------------------v---------------+    +---------------v------------------+
|          PennyLane QAOA          |    |          PennyLane VQE           |
|  - Parameterized cost/mixer      |    |  - Hardware-Efficient Ansatz     |
|    alternating layers (p)        |    |    (Ry, Rz, CNOT entanglement)   |
|  - QNode expectation evaluation  |    |  - Variational parameter vector  |
|  - Computational basis sampling  |    |  - Statevector probability norm  |
+------------------+---------------+    +---------------+------------------+
                   |                                    |
                   +-----------------+------------------+
                                     |
+------------------------------------v-------------------------------------+
|                     PennyLane Execution & Optimization                   |
|       - Backend device provisioning (default.qubit, lightning.qubit)     |
|       - Unified optimizer factory (COBYLA, SLSQP, Adam, SPSA)            |
|       - Exact statevector evaluation & shot-based sampling               |
|       - Solution bitstring decoding to continuous weights                |
+------------------------------------+-------------------------------------+
                                     |
+------------------------------------v-------------------------------------+
|                  Validation, Evaluation & Explainability                 |
|       - Full investment budget feasibility checking                      |
|       - Classical baseline comparison (Markowitz SLSQP, Greedy)          |
|       - Sharpe ratio, drawdown, and turnover analysis                    |
|       - Investment memo generation and XAI copilot                       |
+------------------------------------+-------------------------------------+
                                     |
+------------------------------------v-------------------------------------+
|                        Streamlit Dashboard UI                            |
|       - Market Regime View, Portfolio View, Comparison, Copilot          |
+--------------------------------------------------------------------------+
```

---

## 2. Module Responsibilities

### 2.1 `optimization/`
- **`hamiltonian_builder.py`**: Discretizes asset weights into binary variables ($w_i = \text{scale} \sum 2^k x_{i,k}$) and builds the symmetric QUBO matrix $Q$ with offset $\lambda$.
- **`constraints.py`**: Formulates budget and weight bounds and evaluates constraint penalties.
- **`adaptive_weights.py`**: Dynamically adjusts risk-aversion parameters conditioned on market regimes.
- **`portfolio_metrics.py`**: Evaluates expected return, portfolio variance, volatility, and Sharpe ratios.

### 2.2 `quantum_algorithms/`
- **`backend.py`**: Provisions PennyLane simulation devices (`default.qubit`, `lightning.qubit`) with validated wire counts, shots, and seeds.
- **`optimizer.py`**: Unified optimization layer supporting classical SciPy minimizers (COBYLA, SLSQP) and PennyLane differentiable optimizers (Adam, GradientDescent, SPSA).
- **`ansatz.py`**: Constructs parameterized quantum circuits using PennyLane operations (Ry, Rz, and CNOT entanglement).
- **`qaoa_solver.py`**: Executes the Quantum Approximate Optimization Algorithm using native PennyLane QNodes.
- **`vqe_solver.py`**: Executes the Variational Quantum Eigensolver with hardware-efficient and strongly-entangling ansätze.
- **`circuit_visualizer.py`**: Visualizes and inspects PennyLane quantum circuits using text and Matplotlib drawing tools.

### 2.3 `classical_baseline/`
- **`mean_variance.py`**: Standard quadratic programming baseline for continuous Markowitz portfolio selection.
- **`scipy_optimizer.py`**: Numerical constrained optimization via SLSQP.
- **`greedy_baseline.py`**: Greedy asset selection baseline.

### 2.4 `streamlit_dashboard/`
- **`dashboard.py`**: Main application container, session state manager, and navigation coordinator.
- **`portfolio_view.py`**: Interactive quantum allocation console with real-time variational convergence, asset weights donut, and probability spectrum.
- **`market_view.py`**: Market regime classification, historical volatility, and correlation matrices.
- **`comparison_page.py`**: Side-by-side performance benchmarking against classical algorithms.
- **`copilot.py`**: Natural language quantum portfolio explanation and investment memo synthesis.
