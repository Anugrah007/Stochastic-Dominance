# Stochastic Order Dominance Portfolio Construction

This project implements **Second-Order Stochastic Dominance (SSD) portfolio optimization** in Python, ensuring that the constructed portfolio stochastically dominates a benchmark (e.g., equally weighted portfolio) for all risk-averse investors.  
The framework includes **data preprocessing, portfolio optimization under SSD constraints, and backtesting** against benchmarks.

---

## 📌 Highlights

- **Theory-backed optimization:** Implements **first-order** and **second-order stochastic dominance (SSD)** constraints for portfolio selection.  
- **Optimization approach:** Uses **convex optimization (CVXPY)** to solve SSD-constrained allocation problems.  
- **Backtesting engine:** Compares SSD portfolio vs. benchmark portfolio performance over time.  
- **Clean modular design:**  
  - `data_import.py` & `data_clean.py` – load and preprocess raw financial data.  
  - `strategy.py` – SSD-based portfolio construction logic.  
  - `backtest.py` – evaluate portfolio performance historically.  
- **Applications:** Portfolio optimization, risk-sensitive decision-making, and benchmark comparisons under risk aversion.  

---

## 📂 Repository Structure

```text
├── ssd_method.ipynb             # Notebook: SSD optimization demonstration
├── backtesting_ssd.ipynb        # Notebook: backtesting analysis
├── data_import.py               # Import market data
├── data_clean.py                # Data cleaning and preparation
├── strategy.py                  # SSD portfolio optimization logic
├── backtest.py                  # Backtesting framework
├── theory.pdf                   # Theoretical background on stochastic dominance
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Methodology

1. **Stochastic Dominance Framework**  
   - *First-Order Stochastic Dominance (FSD):* Ensures one distribution yields higher utility for all non-decreasing utility functions.  
   - *Second-Order Stochastic Dominance (SSD):* Captures risk aversion; guarantees superiority for all concave utility functions:contentReference[oaicite:0]{index=0}.  


### Optimization Problem:


$\max_{x \in X} E[R_p]$


Subject to:


Here considered long only portflio
$\[x_i \geq 0, \quad \forall i\]$

Allocation contraints
$\[ \sum x_i = 1 \]$

### Second-Order Stochastic Dominance (SSD)
Accounts for risk aversion and ensures that one distribution is preferable for all risk-averse investors. Mathematically, SSD holds if:

$\int_{-\infty}^{t} F_X(s) ds \leq \int_{-\infty}^{t} F_Y(s) ds, \quad \forall t$

This ensures that \( X \) provides greater expected utility than \( Y \) for concave utility functions.


## Stochastic Dominance Constraints in Optimization

SDCs are implemented in optimization problems to ensure that the chosen decision variables yield outcomes that are stochastically superior. These constraints typically appear in:

- **Portfolio Optimization**: Enforcing SD constraints ensures that an optimal portfolio dominates a benchmark portfolio in terms of risk-return trade-offs.
- **Risk-Aware Decision Making**: Ensures that selected strategies minimize downside risk and align with investor preferences.
- **Insurance and Risk Management**: Helps in selecting policies that offer better probabilistic guarantees against losses.

## Formulating Stochastic Dominance Constraints

In practical optimization models, SD constraints are often formulated using:

- **Linear Programming (LP)** for SSD by integrating over empirical CDFs.
- **Convex Optimization Methods**: Formulating SDCs as convex constraints where possible.


---

## 📊 Results

- **SSD Portfolio consistently dominates the benchmark** (equal-weighted).  
- **Risk-averse preference:** SSD portfolios selected under this method are preferred by all risk-averse investors over the benchmark.  
- **Backtest findings:**  
  - Outperforms equally weighted benchmark in cumulative returns.  
  - Provides smoother downside protection due to SSD constraints.  
  - Demonstrates stronger Sharpe ratios and lower drawdowns in test periods.  

*(See `backtesting_ssd.ipynb` for detailed plots and results.)*

---

## Conclusion

- **Risk Averse Preference**: By employing Second order Stochastic Dominace constraints all Risk Averese people will select the obtained portfolio compared to benchmark portfolio
3. **Implementation**  
   - Returns extracted from historical data (via `yfinance` or CSVs).  
   - SSD constraints expressed as convex conditions in **CVXPY**.  
   - Portfolio weights optimized at each rebalance period.  
   - Backtesting engine simulates allocation over historical returns.

 --- 

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run optimization notebook**
   ```bash
   jupyter notebook ssd_method.ipynb
   ```

3. **Run backtest**
   ```bash
   jupyter notebook backtesting_ssd.ipynb
   ```

4. **Workflow**
   - Import & clean financial data (`data_import.py`, `data_clean.py`).  
   - Construct portfolio using **SSD constraints** (`strategy.py`).  
   - Backtest portfolio vs. benchmark (`backtest.py`).  
   - Visualize and compare results.  

---
