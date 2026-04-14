# AI-Driven Portfolio Optimisation

This project demonstrates different portfolio construction techniques using 
classic finance theory and AI-based tools.

## Features
- Mean-Variance Optimisation (Markowitz)
- Black–Litterman Model
- Risk Parity Allocation
- Robust Optimisation (handles estimation errors)
- Stress Testing & Scenario Analysis

## Tech Stack
- Python
- Jupyter Notebooks
- pandas, numpy, matplotlib
- PyPortfolioOpt, scikit-learn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/alzum73/portfolio-optimisation.git

## Option Strategy Decision Framework

A reusable options decision framework is available in `skew/decision_framework.py` and integrated via `skew/utils.py`.

It provides:
- Probability of finishing in-the-money (`probability_itm`)
- Probability-weighted expected payoff (`probability_weighted_payoff`)
- Risk/reward asymmetry diagnostics (`evaluate_risk_reward_asymmetry`)
- Direct strategy construction from fetched chains (`build_legs_from_chain`)
- Convenience integration hook (`evaluate_strategy_from_chain` in `skew/utils.py`)

Example strategy spec:

```python
leg_specs = [
    ("C", 105.0, +1),
    ("C", 110.0, -1),
]
```
