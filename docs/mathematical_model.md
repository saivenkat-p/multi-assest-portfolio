# ARQPO: Mathematical Model & Formulation
## Adaptive Regime-Aware Quantum Portfolio Optimization

**Mathematical Framework:** Mean-Variance Portfolio Theory, Quadratic Unconstrained Binary Optimization (QUBO), and PennyLane Variational Quantum Algorithms.

---

## 1. Classical Portfolio Optimization

The continuous Markowitz mean-variance portfolio selection problem determines asset allocation weights $w = [w_0, w_1, \dots, w_{N-1}]^T \in \mathbb{R}^N$ to maximize expected return while penalizing portfolio variance:

$$\min_{w \in \Delta^N} \mathcal{L}(w) = \frac{1}{2} w^T \Sigma w - \gamma \mu^T w$$

subject to the investment simplex constraint:
$$\sum_{i=0}^{N-1} w_i = 1, \quad w_i \ge 0 \quad \forall i \in \{0, \dots, N-1\}$$

where:
- $\mu \in \mathbb{R}^N$: Annualized expected asset returns.
- $\Sigma \in \mathbb{R}^{N \times N}$: Annualized covariance matrix of asset returns.
- $\gamma > 0$: Regime-dependent risk-aversion coefficient.
- $\Delta^N$: Standard probability simplex in $\mathbb{R}^N$.

To cast the constrained problem into an unconstrained objective suitable for quantum optimization, the budget equality constraint is incorporated as a quadratic penalty:

$$\mathcal{P}_{\text{budget}}(w) = \lambda \left(\sum_{i=0}^{N-1} w_i - 1\right)^2 = \lambda \left( \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} w_i w_j - 2 \sum_{i=0}^{N-1} w_i + 1 \right)$$

where $\lambda > 0$ is a large penalty factor guaranteeing constraint compliance.

---

## 2. Discretization and Binary Encoding

Quantum algorithms such as QAOA and VQE operate on discrete computational basis states $|x\rangle$ where $x \in \{0, 1\}^M$. Continuous weights are discretized using a base-2 fixed-point binary representation with $B$ bits per asset:

$$w_i = \text{scale} \sum_{k=0}^{B-1} 2^k x_{i, k}, \quad \text{where } \text{scale} = \frac{1}{2^B - 1}$$

The total number of required qubits is:
$$M = N \times B$$

Substituting the discrete weight expansion into the Markowitz objective and budget penalty yields the Quadratic Unconstrained Binary Optimization (QUBO) problem:

$$\min_{x \in \{0, 1\}^M} E(x) = x^T Q x + \text{offset}$$

where $Q \in \mathbb{R}^{M \times M}$ is a symmetric interaction matrix and $\text{offset} = \lambda$.

### 2.1 QUBO Matrix Construction

1. **Portfolio Variance Term:**
   $$Q_{\text{var}}(i \cdot B + k_1, j \cdot B + k_2) = \frac{1}{2} \Sigma_{ij} \cdot \text{scale}^2 \cdot 2^{k_1 + k_2}$$

2. **Expected Return Term (Diagonal):**
   $$Q_{\text{ret}}(i \cdot B + k, i \cdot B + k) = -\gamma \mu_i \cdot \text{scale} \cdot 2^k$$

3. **Budget Constraint Term:**
   $$Q_{\text{penalty}}(i \cdot B + k_1, j \cdot B + k_2) = \lambda \cdot \text{scale}^2 \cdot 2^{k_1 + k_2} - 2\lambda \cdot \text{scale} \cdot 2^{k_1} \delta_{ij}\delta_{k_1 k_2}$$

The overall QUBO matrix is symmetrized: $Q = \frac{1}{2}(Q + Q^T)$.

---

## 3. QUBO-to-Ising Spin Transformation

To evaluate the objective on quantum hardware using PennyLane, binary variables $x_i \in \{0, 1\}$ are mapped to spin observables $Z_i \in \{+1, -1\}$:

$$x_i = \frac{I - Z_i}{2}$$

where $Z_i$ is the Pauli-Z operator acting on qubit $i$ with eigenbasis $Z|0\rangle = +|0\rangle$ and $Z|1\rangle = -|1\rangle$.

### 3.1 Term-by-Term Algebraic Expansion

For any quadratic binary form $x^T Q x = \sum_i Q_{ii} x_i + \sum_{i < j} 2 Q_{ij} x_i x_j$:

1. **Diagonal (Linear) Terms ($x_i^2 = x_i$):**
   $$Q_{ii} x_i = Q_{ii} \left(\frac{I - Z_i}{2}\right) = \frac{Q_{ii}}{2} I - \frac{Q_{ii}}{2} Z_i$$

2. **Off-Diagonal (Quadratic Interaction) Terms ($i < j$):**
   $$2 Q_{ij} x_i x_j = 2 Q_{ij} \left(\frac{I - Z_i}{2}\right)\left(\frac{I - Z_j}{2}\right) = \frac{Q_{ij}}{2} I - \frac{Q_{ij}}{2} Z_i - \frac{Q_{ij}}{2} Z_j + \frac{Q_{ij}}{2} Z_i Z_j$$

### 3.2 Resulting Ising Spin Hamiltonian

Summing all terms yields the PennyLane Ising cost Hamiltonian:

$$\hat{H}_C = c_0 I + \sum_{i=0}^{M-1} h_i Z_i + \sum_{0 \le i < j < M} J_{ij} Z_i Z_j$$

where the exact parameters are:
$$J_{ij} = \frac{Q_{ij}}{2}$$
$$h_i = -\frac{1}{2} \sum_{j=0}^{M-1} Q_{ij}$$
$$c_0 = \text{offset} + \frac{1}{4} \sum_{i=0}^{M-1} \sum_{j=0}^{M-1} Q_{ij} + \frac{1}{4} \sum_{i=0}^{M-1} Q_{ii}$$

### 3.3 Exact Energy Equivalence Theorem
For any computational basis bitstring $b = (b_0, \dots, b_{M-1}) \in \{0, 1\}^M$ and corresponding state $|b\rangle$:
$$\langle b | \hat{H}_C | b \rangle \equiv b^T Q b + \text{offset}$$
This equality holds identically for all $2^M$ basis states.

---

## 4. Quantum Approximate Optimization Algorithm (QAOA)

QAOA prepares a parameterized quantum state $|\psi(\gamma, \beta)\rangle$ via alternating applications of cost and mixer unitaries applied to the uniform superposition $|+\rangle^{\otimes M}$:

$$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^p \left( e^{-i \beta_l \hat{H}_M} e^{-i \gamma_l \hat{H}_C} \right) |+\rangle^{\otimes M}$$

1. **Cost Unitary $U_C(\gamma_l) = e^{-i \gamma_l \hat{H}_C}$:**
   - Single-qubit rotations: $R_Z(2 \gamma_l h_i)$ on wire $i$.
   - Two-qubit interactions: $\text{IsingZZ}(2 \gamma_l J_{ij})$ on wires $[i, j]$.

2. **Mixer Unitary $U_M(\beta_l) = e^{-i \beta_l \sum_i X_i}$:**
   - Single-qubit rotations: $R_X(2 \beta_l)$ on wire $i$.

3. **Variational Expectation Objective:**
   $$E(\gamma, \beta) = \langle \psi(\gamma, \beta) | \hat{H}_C | \psi(\gamma, \beta) \rangle$$
   Optimized variationally over $(\gamma, \beta) \in \mathbb{R}^{2p}$ using PennyLane QNodes and classical optimizers (COBYLA, Adam).

---

## 5. Variational Quantum Eigensolver (VQE)

VQE utilizes a hardware-efficient ansatz (HEA) with alternating layers of parameterized $R_Y(\theta)$ and $R_Z(\theta)$ single-qubit rotations followed by entangling CNOT gates in linear or circular topology:

$$|\psi(\theta)\rangle = U_{\text{ansatz}}(\theta) |0\rangle^{\otimes M}$$

1. **Parameter Count:** For $M$ qubits and $p$ repetition layers:
   $$K = 2 M (p + 1)$$
2. **Ground State Energy:**
   $$\min_\theta \langle \psi(\theta) | \hat{H}_C | \psi(\theta) \rangle \ge E_{\text{ground}}$$
3. **Measurement & Decoding:**
   Computational basis probabilities $P(b) = |\langle b | \psi(\theta^*) \rangle|^2$ are evaluated. The state $b^* = \arg\max_b P(b)$ with highest measurement probability is decoded into portfolio weights and projected onto the simplex $\sum w_i = 1$.
