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
- Strategy constructors for straddle/strangle (`build_straddle`, `build_strangle`)
- Terminal density + payoff visualization (`plot_strategy_distribution`)
- Direct strategy construction from fetched chains (`build_legs_from_chain`)
- Convenience integration hook (`evaluate_strategy_from_chain` in `skew/utils.py`)

Example strategy spec:

```python
leg_specs = [
    ("C", 105.0, +1),
    ("C", 110.0, -1),
]
```

Visualization example (includes net risk premium and probability of profit):

```python
from skew.decision_framework import (
    build_straddle,
    plot_strategy_distribution,
)

legs = build_straddle(
    strike=100,
    call_premium=3.2,
    put_premium=2.8,
    quantity=1,
)

fig, axes, diagnostics = plot_strategy_distribution(
    spot=100,
    time_to_expiry=30 / 365,
    risk_free_rate=0.04,
    dividend_yield=0.0,
    volatility=0.25,
    legs=legs,
    title="30D ATM Long Straddle",
)

print("Probability of profit:", diagnostics.probability_of_profit)
print("Net premium:", diagnostics.net_premium)
print("Breakevens:", diagnostics.breakeven_points)
```
