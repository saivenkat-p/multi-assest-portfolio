# ARQPO: PennyLane Migration Plan
## Architecture Transition from Qiskit to Xanadu PennyLane

**Document Version:** 1.0.0  
**Target Branch:** `feature/pennylane-migration`  
**Repository:** [saivenkat-p/multi-assest-portfolio](https://github.com/saivenkat-p/multi-assest-portfolio)  
**Author:** Senior Quantum Software Engineering Team  

---

## 1. Executive Summary

The **Adaptive Regime-Aware Quantum Portfolio Optimizer (ARQPO)** framework integrates macroeconomic and statistical regime detection with combinatorial portfolio optimization. The optimization problem is cast as a Quadratic Unconstrained Binary Optimization (QUBO) problem, converted into an Ising spin Hamiltonian, and solved using variational quantum algorithms (QAOA and VQE).

This migration replaces the legacy Qiskit implementation (`qiskit==2.5.0`, `qiskit-algorithms==0.4.0`) with a native **PennyLane** architecture. This transition is not a cosmetic API wrapper; it redesigns the quantum compilation, execution, optimization, and measurement pipelines to harness PennyLane's differentiable quantum programming paradigms, native QNodes, flexible device ecosystems (`default.qubit`, `lightning.qubit`), and clean mathematical abstractions.

---

## 2. Current Architecture & Qiskit Dependency Inventory

### 2.1 File-by-File Dependency Audit

| File | Current Qiskit Components | Role / Limitations |
| :--- | :--- | :--- |
| `quantum_algorithms/qaoa_solver.py` | `qiskit.quantum_info.SparsePauliOp`<br>`qiskit.primitives.StatevectorSampler`<br>`qiskit.primitives.StatevectorEstimator`<br>`qiskit_algorithms.QAOA`<br>`qiskit_algorithms.optimizers.COBYLA` | High-level Qiskit algorithm wrapper. Rigid parameter bindings, little-endian bitstring convention inversion, tight coupling to deprecated Qiskit Algorithms primitives. |
| `quantum_algorithms/vqe_solver.py` | `qiskit.circuit.library.real_amplitudes`<br>`qiskit.quantum_info.SparsePauliOp`<br>`qiskit.quantum_info.Statevector`<br>`qiskit_algorithms.VQE`<br>`qiskit_algorithms.optimizers.COBYLA` | Relies on Qiskit circuit templates and Statevector extraction. High overhead, deprecated module interfaces. |
| `quantum_algorithms/ansatz.py` | `qiskit.QuantumCircuit`<br>`qiskit.circuit.ParameterVector` | Constructs Hardware-Efficient Ansatz (HEA) with `Ry` and `Rz` rotations and CX entanglement. Directly tied to Qiskit AST. |
| `quantum_algorithms/backend.py` | `qiskit.primitives.StatevectorSampler`<br>`qiskit.primitives.StatevectorEstimator` | Factory returning Qiskit primitive instances. Mocked/stubbed IBM runtime support. |
| `quantum_algorithms/optimizer.py` | `qiskit_algorithms.optimizers.COBYLA`<br>`qiskit_algorithms.optimizers.SPSA`<br>`qiskit_algorithms.optimizers.SLSQP` | Wraps Qiskit-specific optimizer classes. |
| `quantum_algorithms/circuit_visualizer.py` | `qiskit.QuantumCircuit.draw`<br>`qiskit.visualization` | Qiskit circuit drawer (`mpl` and `text`). |
| `optimization/hamiltonian_builder.py` | *None (Pure NumPy/SciPy)* | Formulates QUBO matrix $Q$ and constant scalar offset $\lambda$. Clean decoupling from quantum backends. |
| `streamlit_dashboard/portfolio_view.py` | References legacy import path `04_quantum_algorithms` | Had a broken import path fallback to random mock values; needs direct PennyLane solver wiring. |
| `streamlit_dashboard/dashboard.py` | Lists `aer_simulator`, `fake_santiago`, etc. | Dashboard UI backend selector exposes Qiskit Aer artifacts. |
| `streamlit_dashboard/copilot.py` | Text strings citing Qiskit Aer and QAOA | Hardcoded Qiskit references in AI explanations and memos. |
| `requirements.txt` | `qiskit==2.5.0`, `qiskit-algorithms==0.4.0` | Encoded as UTF-16, contains obsolete quantum dependencies. |
| `tests/` (`test_*.py`) | 0-byte stub files | No existing automated test coverage for quantum solvers. |

---

## 3. Target PennyLane Architecture

### 3.1 Architectural Decomposition

The target architecture maintains strict separation of concerns across 6 modular tiers:

```
+-------------------------------------------------------------------+
|                     Streamlit Dashboard / UI                      |
|       (Market View, Portfolio View, Comparison, Copilot)          |
+---------------------------------+---------------------------------+
                                  |
+---------------------------------v---------------------------------+
|                       Evaluation & Baseline                       |
|       (BenchmarkEvaluator, MeanVariance, ScipyOptimizer)          |
+---------------------------------+---------------------------------+
                                  |
+---------------------------------v---------------------------------+
|               Financial Optimization & QUBO Layer                 |
|       (HamiltonianBuilder, ObjectiveFunction, Constraints)        |
+---------------------------------+---------------------------------+
                                  | QUBO Matrix Q & Offset
+---------------------------------v---------------------------------+
|               PennyLane Hamiltonian Translation                   |
|       (qubo_to_pennylane_hamiltonian: qml.Hamiltonian / LinearOps)|
+---------------------------------+---------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
+--------v-----------------------+     +-------------------v--------+
|       PennyLane QAOA           |     |       PennyLane VQE        |
|  - qml.qaoa.cost_layer         |     |  - Custom HEA / Strongly   |
|  - qml.qaoa.mixer_layer        |     |    Entangling Ansatz       |
|  - Cost & Mixer Hamiltonians   |     |  - Differentiable QNode    |
|  - Multi-layer repetition (p)  |     |  - Parameter Optimization  |
|  - Analytical / Sample States  |     |  - State Probabilities     |
+--------+-----------------------+     +-------------------+--------+
         |                                                 |
         +------------------------+------------------------+
                                  |
+---------------------------------v---------------------------------+
|                   PennyLane Backend & Execution                   |
|  - Device Management (default.qubit, lightning.qubit)             |
|  - Optimizers (COBYLA, Adam, GradientDescent, SLSQP via SciPy)    |
|  - Circuit Visualization (qml.draw, qml.draw_mpl)                |
|  - Solution Decoding & Bitstring Mapping                          |
+-------------------------------------------------------------------+
```

---

## 4. Mathematical Migration Requirements

### 4.1 Financial Portfolio Formulation
The continuous portfolio optimization problem minimizes risk subject to target return and full investment budget:
$$\min_{w \in [0, 1]^N} \frac{1}{2} w^T \Sigma w - \gamma \mu^T w + \lambda \left(\sum_{i=0}^{N-1} w_i - 1\right)^2$$

### 4.2 Discretization & Binary Encoding
Each asset weight $w_i$ is discretized using $B$ binary variables $x_{i,k} \in \{0, 1\}$:
$$w_i = \text{scale} \sum_{k=0}^{B-1} 2^k x_{i,k}, \quad \text{scale} = \frac{1}{2^B - 1}$$
Total number of binary decision variables (qubits): $M = N \times B$.

The resulting QUBO problem minimizes:
$$E(x) = x^T Q x + \text{offset}$$
where $Q$ is an $M \times M$ symmetric matrix and $\text{offset} = \lambda$.

### 4.3 QUBO-to-Ising Transformation
To map binary variables $x_i \in \{0, 1\}$ to Pauli spin operators $Z_i \in \{+1, -1\}$ (acting on $|0\rangle \to +1$, $|1\rangle \to -1$):
$$x_i = \frac{I - Z_i}{2}$$

Substituting into $x^T Q x = \sum_i Q_{ii} x_i + \sum_{i < j} 2 Q_{ij} x_i x_j$:
1. **Diagonal terms ($x_i^2 = x_i$):**
   $$Q_{ii} x_i = Q_{ii} \frac{I - Z_i}{2} = \frac{Q_{ii}}{2} I - \frac{Q_{ii}}{2} Z_i$$
2. **Off-diagonal terms ($i < j$):**
   $$2 Q_{ij} x_i x_j = 2 Q_{ij} \left(\frac{I - Z_i}{2}\right)\left(\frac{I - Z_j}{2}\right) = \frac{Q_{ij}}{2} I - \frac{Q_{ij}}{2} Z_i - \frac{Q_{ij}}{2} Z_j + \frac{Q_{ij}}{2} Z_i Z_j$$

Summing all contributions yields the Ising Hamiltonian:
$$\hat{H}_C = c_0 I + \sum_{i=0}^{M-1} h_i Z_i + \sum_{0 \le i < j < M} J_{ij} Z_i Z_j$$
where:
$$J_{ij} = \frac{Q_{ij}}{2}$$
$$h_i = -\frac{1}{2} Q_{ii} - \sum_{j \neq i} \frac{Q_{ij}}{2} = -\frac{1}{2} \sum_{j=0}^{M-1} Q_{ij}$$
$$c_0 = \text{offset} + \frac{1}{2} \sum_{i=0}^{M-1} Q_{ii} + \sum_{0 \le i < j < M} \frac{Q_{ij}}{2} = \text{offset} + \frac{1}{4} \sum_{i=0}^{M-1} \sum_{j=0}^{M-1} Q_{ij} + \frac{1}{4} \sum_{i=0}^{M-1} Q_{ii}$$

### 4.4 Verification Requirement: Basis State Energy Equivalence
For any computational basis bitstring $b \in \{0, 1\}^M$, where eigenvalue of $Z_i$ is $s_i = (-1)^{b_i} \in \{+1, -1\}$:
$$E_{\text{QUBO}}(b) = b^T Q b + \text{offset} \equiv \langle b | \hat{H}_C | b \rangle$$
This must hold identically to within floating-point tolerance ($\le 10^{-10}$) across all $2^M$ basis states for test systems.

---

## 5. API Compatibility Requirements

The migration must preserve the public interfaces of solver classes and methods:

### 5.1 `QAOAPortfolioSolver`
```python
class QAOAPortfolioSolver:
    def __init__(
        self,
        device_name: str = "default.qubit",
        reps: int = 2,
        optimizer_name: str = "COBYLA",
        optimizer_iter: int = 100,
        shots: Optional[int] = None,
        seed: Optional[int] = 42,
    ): ...

    def solve(
        self,
        qubo: np.ndarray,
        n_assets: int,
        binary_bits: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]: ...
```

### 5.2 `VQEPortfolioSolver`
```python
class VQEPortfolioSolver:
    def __init__(
        self,
        device_name: str = "default.qubit",
        reps: int = 3,
        entanglement: str = "linear",
        optimizer_name: str = "COBYLA",
        optimizer_iter: int = 150,
        shots: Optional[int] = None,
        seed: Optional[int] = 42,
    ): ...

    def solve(
        self,
        qubo: np.ndarray,
        n_assets: int,
        binary_bits: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]: ...
```

### 5.3 Return Tuple & Metadata Contract
The return value remains:
1. `weights: np.ndarray` (shape `(n_assets,)`, sum $\approx 1$, non-negative)
2. `metadata: Dict[str, Any]` containing:
   - `optimal_parameters`: optimized variational parameter array
   - `final_energy`: minimum expectation value found $\langle \hat{H} \rangle$
   - `best_bitstring`: most probable bitstring
   - `optimizer_evals`: number of cost evaluations
   - `reps`: circuit layers
   - `num_qubits`: total qubits ($N \times B$)
   - `num_assets`: asset count ($N$)
   - `execution_time_sec`: duration of run
   - `solver`: `"QAOA"` or `"VQE"`
   - `backend`: e.g. `"PennyLane:default.qubit"`

---

## 6. Test Strategy

1. **Unit Mathematical Tests (`tests/test_hamiltonian.py`)**:
   - Direct verification that $b^T Q b + \text{offset} == \langle b | \hat{H}_C | b \rangle$ for all $2^M$ states ($M \in [2, 6]$).
   - Validation that energy ground state bitstring decodes to optimal discrete weight portfolio.
2. **QAOA Solver Tests (`tests/test_qaoa.py`)**:
   - Parameter optimization convergence on deterministic benchmark portfolios.
   - Exact bitstring decoding, little-endian vs big-endian ordering checks.
   - Seed reproducibility on `default.qubit`.
3. **VQE Solver Tests (`tests/test_vqe.py`)**:
   - Hardware-efficient ansatz evaluation.
   - Eigenvalue matching against exact matrix diagonalization (NumPy/SciPy `eigvalsh`).
4. **Backend & Optimizer Tests (`tests/test_backend.py`, `tests/test_optimizer.py`)**:
   - Instantiation of PennyLane devices (`default.qubit`, `lightning.qubit` fallback).
   - Optimizer convergence with COBYLA, SLSQP, SPSA, and Adam.
5. **Dashboard & Classical Regression Tests (`tests/test_integration.py`, `tests/test_classical_regression.py`)**:
   - Verification that `run_quantum_optimization` in `portfolio_view.py` executes without error.
   - Verification of `mean_variance.py`, `scipy_optimizer.py`, `data_loader.py`, and `market_classifier.py`.

---

## 7. Migration Sequence & Execution Phases

- **Phase 1: Audit and Documentation Baseline**
  - Establish `docs/PENNYLANE_MIGRATION_PLAN.md` and `docs/PENNYLANE_MIGRATION_LOG.md`.
  - Initial git commits for audit milestone.
- **Phase 2: Mathematical Foundation & Hamiltonian Construction**
  - Implement native PennyLane QUBO-to-Ising mapper with exact constant energy preservation.
  - Implement full exhaustive validation tests.
- **Phase 3: Quantum Circuit & Ansatz Layer**
  - Implement PennyLane native parameterized ansatz (`Ry`, `Rz`, and CNOT entanglement).
  - Port circuit visualizer using `qml.draw` and `qml.draw_mpl`.
- **Phase 4: Backend & Optimizer Abstractions**
  - Modernize `backend.py` for PennyLane devices.
  - Modernize `optimizer.py` to route classical and gradient-based optimizers cleanly.
- **Phase 5: PennyLane QAOA Solver**
  - Implement full QAOA pipeline with QNodes, cost/mixer layers, probability sampling, and decoding.
- **Phase 6: PennyLane VQE Solver**
  - Implement VQE pipeline with parameter optimization, statevector probabilities, and decoding.
- **Phase 7: Dashboard & Visualization Integration**
  - Fix import path in `portfolio_view.py` to call real solvers directly.
  - Update UI labels, backend selectors, and Copilot explanations.
- **Phase 8: Comprehensive Testing, Regression & Dependency Cleanup**
  - Execute full test suite.
  - Clean `requirements.txt` to UTF-8 and replace Qiskit with PennyLane.
  - Final technical audit and push to `feature/pennylane-migration`.

---

## 8. Known Risks & Mitigations

1. **Bitstring Indexing Mismatch**: Qiskit historically orders qubits little-endian (`q_0` rightmost in string representation), whereas PennyLane QNodes index wires from left to right (`wire 0` leftmost).
   - *Mitigation:* Explicitly standardize on canonical wire indexing and test with asymmetric QUBO matrices where bit order inversion would produce incorrect energies.
2. **Continuous vs Discrete Weights**: Quantum QUBO solvers solve discretized approximations of the continuous simplex.
   - *Mitigation:* Maintain post-processing normalization and feasibility checks so portfolios always satisfy budget $\sum w_i = 1$.
3. **Execution Time in Browser / Streamlit**: Simulated quantum optimization can cause latency if iterations or qubit counts are large.
   - *Mitigation:* Provide sensible default iterations (30-80) and caching in Streamlit session state.
