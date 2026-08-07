 # Architecture
## System Overview

Market Data
    ↓
Data Preprocessing
    ↓
Regime Detection
    ↓
Optimization
    ↓
Quantum Algorithms
    ↓
Validation
    ↓
Evaluation
    ↓
Explainability
    ↓
Streamlit Dashboard

## Module Responsibilities

### data_preprocessing
- Clean data
- Calculate returns
- Prepare covariance matrix

### market_regime_detection
- Identify bull/bear/sideways markets

### optimization
- Objective function
- Constraints
- Hamiltonian builder

### quantum_algorithms
- QAOA
- VQE
- Backend interface

### classical_baseline
- Mean-Variance Optimization

### validation
- Constraint checking
- Exposure limits

### evaluation
- Performance metrics
- Runtime comparison

### explainability
- Human-readable explanations

### streamlit_dashboard
- Interactive visualization