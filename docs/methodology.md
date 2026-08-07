You are a Senior Quantitative Finance Researcher, Quantum Computing Researcher, and Technical Documentation Expert.

Your task is to write a professional `docs/methodology.md` document for my completed project.

The document must be based ONLY on the repository implementation. Do not invent algorithms or features that are not present in the code.

The document should be written in Markdown and suitable for an international competition submission such as the WISER × Vanguard Quantum Challenge.

Include the following sections:

# Methodology

## 1. Problem Statement
- Explain the portfolio optimization problem.
- Explain why classical portfolio optimization is challenging.
- Explain the motivation for quantum optimization.

## 2. Overall Pipeline
Describe the complete workflow from market data input to final portfolio recommendation.

Include:
Market Data
→ Data Preprocessing
→ Market Regime Detection
→ Optimization
→ Classical Baseline
→ Quantum Optimization
→ Validation
→ Evaluation
→ Explainability
→ Streamlit Dashboard

## 3. Data Preprocessing
Explain:
- Missing value handling
- Return calculation
- Covariance matrix generation
- Data normalization (if implemented)

## 4. Market Regime Detection
Explain:
- How market regimes are identified
- Features used
- Output of the module
- Why regime awareness improves portfolio optimization

## 5. Portfolio Optimization
Explain:
- Objective function
- Mathematical formulation
- Constraints
- Risk-return trade-off
- Diversification
- Transaction costs
- Portfolio metrics

Include equations where appropriate.

## 6. Classical Baseline
Explain:
- Mean-Variance Optimization
- Optimization solver
- Advantages
- Limitations

## 7. Quantum Optimization
Explain:
- QUBO formulation
- Hamiltonian construction
- QAOA
- VQE
- Ansatz selection
- Classical optimizer
- Backend selection
- Measurement process

Include the mathematical intuition behind each step.

## 8. Validation
Explain:
- Constraint checking
- Weight normalization
- Exposure limits
- Feasibility verification

## 9. Evaluation
Explain every evaluation metric used:
- Expected Return
- Volatility
- Sharpe Ratio
- Maximum Drawdown
- Portfolio Turnover
- Runtime
- Benchmark comparison

Explain why each metric is important.

## 10. Explainability
Explain:
- Portfolio allocation explanation
- Risk explanation
- Regime explanation
- Trade-off explanation

## 11. Dashboard
Explain:
- Interactive workflow
- User inputs
- Visualizations
- Comparison views

## 12. Design Decisions
Explain why the project was designed using modular architecture.

## 13. Assumptions
List all important assumptions made by the implementation.

## 14. Limitations
Only mention limitations that actually exist in the repository.

## 15. Future Scope
Suggest realistic future improvements without contradicting the existing implementation.

Requirements:

- Write professionally as if preparing documentation for an IEEE research prototype.
- Use Markdown.
- Use mathematical notation where appropriate.
- Do not fabricate features.
- Keep explanations aligned with the repository implementation.
- Produce a document of approximately 5–8 pages when rendered in Markdown.