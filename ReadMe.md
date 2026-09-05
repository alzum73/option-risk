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

**IBKR is the single option-chain source.** The fetch notebook enriches every
snapshot before saving, so the DB holds everything the analyses need:

| Field | Where it comes from |
|---|---|
| bid / ask / mid, IV, greeks, open interest, spot | IBKR (IB Gateway via `ib_insync`) |
| `disc_factor` | USD SOFR zero curve (`skew/zero_curve.py`, fetched from the Fed) |
| `div_yield` | Yahoo Finance — the only chain field still looked up there (one number per ticker) |
| `forward`, `T` | computed: F = S·e^(−q·T)/D(T) |

Rows are tagged with a `source` column: `ibkr` (live snapshot) or `ibkr_hist`
(historical backfill). Legacy `yfinance` rows may still exist in old databases —
all notebooks now ignore them.

## Notebooks — what reads and writes what

| Notebook | Role | Reads | Writes |
|---|---|---|---|
| `fetch_ibkr_data.ipynb` | **Fetch** | IB Gateway (port 4002) + SOFR curve + Yahoo div yield | `option_snapshots` (`ibkr`, `ibkr_hist`) |
| `ibkr_skew_analysis.ipynb` | Analysis | `option_snapshots` (IBKR sources) | `skew_metrics` |
| `option_metrics_calculator.ipynb` | Analysis | `option_snapshots` (IBKR sources, latest snapshot) + `data/option_data.csv` (portfolio) + Yahoo price history (HV vs IV table) | — |

Typical daily workflow:

1. Run `fetch_ibkr_data.ipynb` during market hours → today's enriched chain
   appended to the DB (IBKR returns no option data outside trading hours)
2. Run `ibkr_skew_analysis.ipynb` → skew metrics saved per snapshot date, surface + PDF plots
3. Run `option_metrics_calculator.ipynb` → greeks and diagnostics for your CSV portfolio

## Project structure

```
option-risk/
├── ReadMe.md
├── data/
│   ├── options.db                       # shared SQLite DB — created by the fetch notebook
│   └── option_data.csv                  # your portfolio positions (symbol, option_type, strike, expiry[, contracts])
├── notebooks/
│   ├── fetch_ibkr_data.ipynb            # FETCH    — IBKR chain + SOFR/div-yield enrichment + IV backfill
│   ├── ibkr_skew_analysis.ipynb         # ANALYSIS — skew metrics, SVI surface, BL PDF, skew history
│   └── option_metrics_calculator.ipynb  # ANALYSIS — portfolio greeks & diagnostics
└── skew/
    ├── data_store.py                    # SQLite persistence: save/load snapshots & skew metrics
    ├── utils.py                         # BS pricing, deltas, SVI fit/eval, IV interpolation
    ├── zero_curve.py                    # USD zero curve (SOFR) → discount factors, forwards
    └── decision_framework.py            # option strategy evaluation (see below)
```

> Removed notebooks: `pdf_calc.ipynb` (surface/PDF/skew fully covered by
> `ibkr_skew_analysis.ipynb`) and `fetch_opt_data.ipynb` (Yahoo chain fetcher —
> IBKR is the single chain source; Yahoo is used only for the dividend yield
> during the IBKR fetch and for underlying price history in the HV table).

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
