## Architecture
* **High-Level Pipeline Diagram:**
* Data Ingestion -> Regime Detection -> QUBO Formulation -> Quantum Optimization (QAOA/VQE) -> Dashboard Visualization.
* **Module Descriptions:**
* **Data Loader:** Handles financial data fetching and preprocessing.
* **Regime Classifier:** Models market states to adjust risk tolerance.
* **Quantum Optimizer:** Maps financial models to quantum circuits.
* **UI/Dashboard:** Streamlit interface for user interaction and metric visualization.

## Repository Structure
* `data_preprocessing/`
* `generate_real_data.py` - Fetches historical data via `yfinance`.
* `data_loader.py` - Cleans and validates datasets.
* `src/` - Core financial and quantum logic.
* `tests/` - Unit and integration tests.
* `app.py` - Streamlit application entry point.
* `requirements.txt` - Project dependencies.

## Dataset Generation
* Uses `yfinance` to automatically pull ~5 years of historical closing prices for a multi-asset portfolio (e.g., AAPL, MSFT, JPM, GLD, TLT).
* Executed via `data_preprocessing/generate_real_data.py`.
* Outputs generated:
  * `historical_prices.csv` - Historical closing prices.
  * `asset_metadata.csv` - Asset information.
  * `synthetic_assets.csv` - Benchmark assets for experimentation.
  * `market_scenarios.csv` - Predefined regime scenarios.
* `data_loader.py` handles parsing, DatetimeIndex conversion, missing value imputation, and validation.

## Installation
* Clone the repository: `git clone <repo_url>`
* Create a virtual environment: `python -m venv venv`
* Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
* Install requirements: `pip install -r requirements.txt`

## Running the Project
* **Dashboard:** Execute `streamlit run app.py`
* **Individual Modules:** Run specific scripts from the `src/` directory.
* **Tests:** Execute `pytest` to run the test suite.

## Technologies Used
* Python
* Qiskit
* Streamlit
* NumPy
* Pandas
* SciPy
* Matplotlib

## Financial Methodology
* **Mean-Variance Optimization:** Core basis for finding optimal risk-return trade-offs.
* **Transaction Costs:** Penalties applied for portfolio rebalancing.
* **Portfolio Constraints:** Adherence to realistic trading rules and bounds.
* **Risk Metrics:** Calculation of expected volatility and covariance matrices.

## Quantum Methodology
* **QUBO Formulation:** Translating the financial constraints into a Quadratic Unconstrained Binary Optimization problem.
* **Hamiltonian Construction:** Mapping the QUBO to an Ising model suitable for quantum hardware.
* **QAOA:** Quantum Approximate Optimization Algorithm used for finding approximate solutions.
* **VQE:** Variational Quantum Eigensolver employed as an alternative hybrid optimization method.

## Evaluation Metrics
* **Sharpe Ratio:** Risk-adjusted return measure.
* **Maximum Drawdown:** Peak-to-trough decline over the period.
* **Turnover:** Rate of portfolio rebalancing and associated costs.
* **Runtime:** Computational efficiency comparison.
* **Diversification:** Spread of asset allocation to minimize specific risks.

## Validation
* **Weight Constraints:** Ensures total asset weights sum correctly (e.g., 100%).
* **Exposure Limits:** Checks that no single asset exceeds predefined maximum bounds.
* **Feasibility Checks:** Verifies the portfolio is mathematically and practically sound before execution.

## Explainability
* **Portfolio Rationale:** Justification for the selected asset weights based on the algorithm's output.
* **Regime Explanation:** Transparency on why a specific market regime (bull, bear, volatile) was identified.
* **Risk-Return Trade-Offs:** Visual breakdown of how expected returns justify the assumed risks.

## Future Work
* **Multi-Period Optimization:** Extending the model to handle continuous time-step rebalancing.
* **Real Hardware Execution:** Testing and error-mitigating the circuits on physical IBM Quantum processors.
* **Additional Quantum Algorithms:** Exploring alternatives like Grover's Adaptive Search.

## Authors
* Pendyala Sai Venkat
* Saladi Ganesh





