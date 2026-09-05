# Option Risk — Vol Surface & Skew Analysis

Toolkit for fetching equity option chains, building implied-volatility
surfaces (SVI), extracting risk-neutral densities (Breeden–Litzenberger),
tracking skew metrics over time, and evaluating option strategies.

## Data architecture

All fetched data lands in **one SQLite database**:

```
data/options.db
├── option_snapshots   ← raw chains (one row per ticker/date/expiry/strike/type)
└── skew_metrics       ← per-expiry σATM / RR25 / BF25 computed by the analysis
```

Every row in `option_snapshots` is tagged with a `source` column so the two
fetchers coexist without clashing:

| `source` | Written by | Notes |
|---|---|---|
| `ibkr` | `notebooks/fetch_ibkr_data.ipynb` | Live chain via IB Gateway (`ib_insync`), model greeks + IV |
| `ibkr_hist` | `notebooks/fetch_ibkr_data.ipynb` § 4 | Historical IV backfill |
| `yfinance` | `notebooks/fetch_opt_data.ipynb` | Yahoo Finance chain, enriched with SOFR discount factors and forwards |

## Notebooks — what reads and writes what

| Notebook | Role | Reads | Writes |
|---|---|---|---|
| `fetch_ibkr_data.ipynb` | **Fetch** (IBKR) | IB Gateway (port 4002) | `option_snapshots` (`ibkr`, `ibkr_hist`) |
| `fetch_opt_data.ipynb` | **Fetch** (Yahoo) | Yahoo Finance API | `option_snapshots` (`yfinance`) |
| `ibkr_skew_analysis.ipynb` | Analysis | `option_snapshots` — **IBKR sources only** | `skew_metrics` |
| `pdf_calc.ipynb` | Analysis | `option_snapshots` — needs yfinance-schema columns (`isCall`, `disc_factor`, `div_yield`) | — |
| `option_metrics_calculator.ipynb` | Analysis | `option_snapshots` (latest snapshot, any source) + `data/option_data.csv` (portfolio) | — |

Typical daily workflow:

1. Run one (or both) fetch notebooks → today's chains appended to the DB
2. Run `ibkr_skew_analysis.ipynb` → skew metrics saved per snapshot date, surface + PDF plots
3. Run `option_metrics_calculator.ipynb` → greeks and diagnostics for your CSV portfolio

## Package layout

```
skew/
├── data_store.py          # SQLite persistence: save/load snapshots & skew metrics
├── utils.py               # BS pricing, deltas, SVI fit/eval, IV interpolation
├── zero_curve.py          # USD zero curve (SOFR) → discount factors, forwards
└── decision_framework.py  # strategy evaluation (see below)
```

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

## Tech stack

Python · Jupyter · pandas · numpy · scipy · matplotlib · ib_insync · yfinance
