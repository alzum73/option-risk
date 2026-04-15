"""Decision framework for options strategy selection.

This module adds three core pieces of analysis:

1) Probability of finishing in-the-money (ITM)
2) Probability-weighted payoff (expected payoff)
3) Risk/reward asymmetry diagnostics

Assumptions:
- Black-Scholes / lognormal terminal distribution
- Risk-neutral drift for terminal spot distribution
- Premiums are paid/received upfront and included in P&L at expiry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm


@dataclass(frozen=True)
class OptionLeg:
    """Single option leg in a strategy.

    quantity > 0 means long, quantity < 0 means short.
    premium is per-share premium paid (long) or received (short) at inception.
    """

    option_type: str  # "C" or "P"
    strike: float
    premium: float
    quantity: float = 1.0
    contract_size: int = 100

    def __post_init__(self):
        t = self.option_type.upper()
        if t not in {"C", "P"}:
            raise ValueError("option_type must be 'C' (call) or 'P' (put)")


@dataclass(frozen=True)
class StrategyEvaluation:
    expected_payoff: float
    probability_of_profit: float
    expected_upside: float
    expected_downside: float
    upside_downside_ratio: float
    worst_case_payoff: float
    best_case_payoff: float
    net_premium: float
    breakeven_points: tuple[float, ...]
    leg_itm_probabilities: tuple[float, ...]  # P(ITM) per leg, same order as input legs


def probability_itm(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> float:
    """Risk-neutral probability that an option expires ITM.

    For Black-Scholes under risk-neutral measure:
    - P(S_T > K) = N(d2)
    - P(S_T < K) = N(-d2)
    """

    if any(x <= 0 for x in [spot, strike, time_to_expiry, volatility]):
        return np.nan

    t = option_type.upper()
    d2 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))

    if t == "C":
        return float(norm.cdf(d2))
    if t == "P":
        return float(norm.cdf(-d2))
    raise ValueError("option_type must be 'C' or 'P'")


def _terminal_spot_grid(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    grid_size: int,
    tail_probability: float,
) -> np.ndarray:
    """Construct a terminal spot grid using lognormal quantiles."""

    sigma_t = volatility * np.sqrt(time_to_expiry)
    mu = np.log(spot) + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry

    # avoid extreme infinities in quantile inversion
    low_q = max(1e-6, tail_probability)
    high_q = 1.0 - low_q
    quantiles = np.linspace(low_q, high_q, grid_size)
    return lognorm.ppf(quantiles, s=sigma_t, scale=np.exp(mu))


def strategy_payoff_at_expiry(terminal_spot: np.ndarray, legs: Sequence[OptionLeg]) -> np.ndarray:
    """Return total strategy payoff at expiry over terminal spot values."""

    st = np.asarray(terminal_spot, dtype=float)
    payoff = np.zeros_like(st)

    for leg in legs:
        t = leg.option_type.upper()
        if t == "C":
            intrinsic = np.maximum(st - leg.strike, 0.0)
        else:
            intrinsic = np.maximum(leg.strike - st, 0.0)

        leg_payoff = (intrinsic - leg.premium) * leg.quantity * leg.contract_size
        payoff += leg_payoff

    return payoff


def strategy_net_premium(legs: Sequence[OptionLeg]) -> float:
    """Net premium at inception (negative = debit paid, positive = credit received)."""

    return float(-sum(leg.premium * leg.quantity * leg.contract_size for leg in legs))


def estimate_breakevens(
    terminal_spot: np.ndarray,
    payoff: np.ndarray,
) -> tuple[float, ...]:
    """Estimate breakeven terminal spots where strategy payoff crosses zero."""

    st = np.asarray(terminal_spot, dtype=float)
    pnl = np.asarray(payoff, dtype=float)
    sign = np.sign(pnl)
    changes = np.where(np.diff(sign) != 0)[0]

    points: list[float] = []
    for idx in changes:
        x0, x1 = st[idx], st[idx + 1]
        y0, y1 = pnl[idx], pnl[idx + 1]
        if np.isclose(y1, y0):
            points.append(float(x0))
            continue
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        points.append(float(root))
    return tuple(points)


def probability_weighted_payoff(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    legs: Sequence[OptionLeg],
    grid_size: int = 2000,
    tail_probability: float = 0.001,
) -> float:
    """Compute expected strategy payoff at expiry via numerical integration."""

    st = _terminal_spot_grid(
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        grid_size=grid_size,
        tail_probability=tail_probability,
    )
    payoff = strategy_payoff_at_expiry(st, legs)

    sigma_t = volatility * np.sqrt(time_to_expiry)
    mu = np.log(spot) + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    pdf = lognorm.pdf(st, s=sigma_t, scale=np.exp(mu))

    return float(np.trapz(payoff * pdf, st))


def evaluate_risk_reward_asymmetry(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    legs: Sequence[OptionLeg],
    grid_size: int = 2000,
    tail_probability: float = 0.001,
) -> StrategyEvaluation:
    """Evaluate asymmetry and probability-weighted diagnostics for a strategy."""

    st = _terminal_spot_grid(
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        grid_size=grid_size,
        tail_probability=tail_probability,
    )
    payoff = strategy_payoff_at_expiry(st, legs)

    sigma_t = volatility * np.sqrt(time_to_expiry)
    mu = np.log(spot) + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    pdf = lognorm.pdf(st, s=sigma_t, scale=np.exp(mu))

    expected = float(np.trapz(payoff * pdf, st))

    pos = payoff > 0
    neg = payoff < 0

    probability_profit = float(np.trapz(pdf[pos], st[pos])) if np.any(pos) else 0.0
    expected_upside = float(np.trapz(payoff[pos] * pdf[pos], st[pos])) if np.any(pos) else 0.0
    expected_downside = float(-np.trapz(payoff[neg] * pdf[neg], st[neg])) if np.any(neg) else 0.0

    ratio = np.inf if expected_downside == 0 else expected_upside / expected_downside
    net_premium = strategy_net_premium(legs)
    breakevens = estimate_breakevens(st, payoff)

    leg_itm_probs = tuple(
        probability_itm(
            spot=spot,
            strike=leg.strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            option_type=leg.option_type,
        )
        for leg in legs
    )

    return StrategyEvaluation(
        expected_payoff=expected,
        probability_of_profit=probability_profit,
        expected_upside=expected_upside,
        expected_downside=expected_downside,
        upside_downside_ratio=ratio,
        worst_case_payoff=float(np.min(payoff)),
        best_case_payoff=float(np.max(payoff)),
        net_premium=net_premium,
        breakeven_points=breakevens,
        leg_itm_probabilities=leg_itm_probs,
    )


def build_straddle(
    strike: float,
    call_premium: float,
    put_premium: float,
    quantity: float = 1.0,
    contract_size: int = 100,
) -> list[OptionLeg]:
    """Build a long/short straddle from explicit premiums."""

    q = float(quantity)
    return [
        OptionLeg("C", strike=float(strike), premium=float(call_premium), quantity=q, contract_size=contract_size),
        OptionLeg("P", strike=float(strike), premium=float(put_premium), quantity=q, contract_size=contract_size),
    ]


def build_strangle(
    put_strike: float,
    call_strike: float,
    call_premium: float,
    put_premium: float,
    quantity: float = 1.0,
    contract_size: int = 100,
) -> list[OptionLeg]:
    """Build a long/short strangle from explicit premiums."""

    q = float(quantity)
    return [
        OptionLeg("P", strike=float(put_strike), premium=float(put_premium), quantity=q, contract_size=contract_size),
        OptionLeg("C", strike=float(call_strike), premium=float(call_premium), quantity=q, contract_size=contract_size),
    ]


def plot_strategy_distribution(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    legs: Sequence[OptionLeg],
    grid_size: int = 3000,
    tail_probability: float = 0.001,
    title: str | None = None,
):
    """Visualize terminal spot distribution + strategy expiry P&L.

    Returns
    -------
    fig, axes, diagnostics
        diagnostics is a StrategyEvaluation object including probability of profit
        and risk premium (net premium debit/credit).
    """

    st = _terminal_spot_grid(
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        grid_size=grid_size,
        tail_probability=tail_probability,
    )
    payoff = strategy_payoff_at_expiry(st, legs)

    sigma_t = volatility * np.sqrt(time_to_expiry)
    mu = np.log(spot) + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    pdf = lognorm.pdf(st, s=sigma_t, scale=np.exp(mu))
    diagnostics = evaluate_risk_reward_asymmetry(
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        legs=legs,
        grid_size=grid_size,
        tail_probability=tail_probability,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].plot(st, pdf, color="tab:blue", lw=2, label="Terminal density")
    axes[0].axvline(spot, color="tab:gray", ls="--", lw=1.5, label="Spot today")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right")

    axes[1].plot(st, payoff, color="tab:orange", lw=2, label="Expiry P&L")
    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].axvline(spot, color="tab:gray", ls="--", lw=1.5)
    for breakeven in diagnostics.breakeven_points:
        axes[1].axvline(breakeven, color="tab:green", ls=":", lw=1.2)
    axes[1].set_xlabel("Terminal spot at expiry")
    axes[1].set_ylabel("Payoff ($)")
    axes[1].legend(loc="upper right")

    plot_title = title or "Strategy terminal distribution and payoff"
    subtitle = (
        f"POP={diagnostics.probability_of_profit:.1%} | "
        f"E[payoff]={diagnostics.expected_payoff:,.2f} | "
        f"Risk premium (net)={diagnostics.net_premium:,.2f}"
    )
    fig.suptitle(f"{plot_title}\n{subtitle}", fontsize=12)
    return fig, axes, diagnostics


def build_legs_from_chain(
    chain_df,
    leg_specs: Iterable[tuple[str, float, float]],
    premium_col: str = "mid",
) -> list[OptionLeg]:
    """Build strategy legs from an option chain dataframe.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Must contain columns: opt_type, K, and premium_col.
    leg_specs : iterable of tuples
        (option_type, strike, quantity)
    premium_col : str
        Which premium column to read from chain (default: mid).
    """

    required = {"opt_type", "K", premium_col}
    missing = required - set(chain_df.columns)
    if missing:
        raise ValueError(f"chain_df missing required columns: {sorted(missing)}")

    legs: list[OptionLeg] = []

    for option_type, strike, quantity in leg_specs:
        row = chain_df[
            (chain_df["opt_type"].str.upper() == option_type.upper())
            & (np.isclose(chain_df["K"].astype(float), float(strike)))
        ]
        if row.empty:
            raise ValueError(f"No chain row found for {option_type}@{strike}")

        premium = float(row.iloc[0][premium_col])
        legs.append(
            OptionLeg(
                option_type=option_type.upper(),
                strike=float(strike),
                premium=premium,
                quantity=float(quantity),
            )
        )

    return legs


# ---------------------------------------------------------------------------
# Additional strategy builders
# ---------------------------------------------------------------------------

def build_vertical_spread(
    option_type: str,
    long_strike: float,
    short_strike: float,
    long_premium: float,
    short_premium: float,
    quantity: float = 1.0,
    contract_size: int = 100,
) -> list[OptionLeg]:
    """Build a vertical debit spread.

    For calls (option_type='C') this is a bull call spread: long lower-strike
    call, short higher-strike call.  For puts (option_type='P') this is a bear
    put spread: long higher-strike put, short lower-strike put.

    Parameters
    ----------
    option_type : 'C' or 'P'
    long_strike, short_strike : strikes for the long and short legs.
    long_premium, short_premium : per-share premiums for each leg.
    quantity : number of spreads (positive = long spread).
    """
    q = float(quantity)
    return [
        OptionLeg(option_type.upper(), strike=float(long_strike), premium=float(long_premium), quantity=q, contract_size=contract_size),
        OptionLeg(option_type.upper(), strike=float(short_strike), premium=float(short_premium), quantity=-q, contract_size=contract_size),
    ]


def build_iron_condor(
    put_buy_strike: float,
    put_sell_strike: float,
    call_sell_strike: float,
    call_buy_strike: float,
    put_buy_premium: float,
    put_sell_premium: float,
    call_sell_premium: float,
    call_buy_premium: float,
    quantity: float = 1.0,
    contract_size: int = 100,
) -> list[OptionLeg]:
    """Build a short iron condor (sell put spread + sell call spread).

    Strike ordering expected: put_buy < put_sell < call_sell < call_buy.
    The net credit is (put_sell_premium - put_buy_premium) +
                      (call_sell_premium - call_buy_premium).
    """
    q = float(quantity)
    return [
        OptionLeg("P", strike=float(put_buy_strike),  premium=float(put_buy_premium),  quantity=-q, contract_size=contract_size),
        OptionLeg("P", strike=float(put_sell_strike), premium=float(put_sell_premium), quantity=q,  contract_size=contract_size),
        OptionLeg("C", strike=float(call_sell_strike),premium=float(call_sell_premium),quantity=q,  contract_size=contract_size),
        OptionLeg("C", strike=float(call_buy_strike), premium=float(call_buy_premium), quantity=-q, contract_size=contract_size),
    ]


def build_butterfly(
    option_type: str,
    low_strike: float,
    mid_strike: float,
    high_strike: float,
    low_premium: float,
    mid_premium: float,
    high_premium: float,
    quantity: float = 1.0,
    contract_size: int = 100,
) -> list[OptionLeg]:
    """Build a long butterfly spread.

    Buys one low-strike and one high-strike option, sells two mid-strike
    options.  Maximum profit is achieved when the underlying expires at
    mid_strike.  Wings should be equidistant: high - mid == mid - low.

    Parameters
    ----------
    option_type : 'C' or 'P' — both wings use the same option type.
    low_strike, mid_strike, high_strike : strikes for the three legs.
    low_premium, mid_premium, high_premium : per-share premiums.
    quantity : number of butterflies (positive = long butterfly).
    """
    q = float(quantity)
    return [
        OptionLeg(option_type.upper(), strike=float(low_strike),  premium=float(low_premium),  quantity=q,    contract_size=contract_size),
        OptionLeg(option_type.upper(), strike=float(mid_strike),  premium=float(mid_premium),  quantity=-2*q, contract_size=contract_size),
        OptionLeg(option_type.upper(), strike=float(high_strike), premium=float(high_premium), quantity=q,    contract_size=contract_size),
    ]


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    strategies: dict[str, list[OptionLeg]],
    grid_size: int = 2000,
    tail_probability: float = 0.001,
):
    """Evaluate and rank multiple option strategies side by side.

    Parameters
    ----------
    spot, time_to_expiry, risk_free_rate, dividend_yield, volatility : float
        Shared market inputs used for the terminal distribution.
    strategies : dict[str, list[OptionLeg]]
        Mapping of strategy name → list of OptionLeg objects.
    grid_size : int
        Number of integration points for numerical integration.
    tail_probability : float
        Quantile cut-off for the terminal spot grid at each tail.

    Returns
    -------
    pd.DataFrame
        One row per strategy sorted by expected_payoff (descending).
        Columns: expected_payoff, probability_of_profit, expected_upside,
        expected_downside, upside_downside_ratio, worst_case_payoff,
        best_case_payoff, net_premium.
    """
    import pandas as pd

    records = []
    for name, legs in strategies.items():
        ev = evaluate_risk_reward_asymmetry(
            spot=spot,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            legs=legs,
            grid_size=grid_size,
            tail_probability=tail_probability,
        )
        records.append({
            "strategy": name,
            "net_premium": ev.net_premium,
            "expected_payoff": ev.expected_payoff,
            "probability_of_profit": ev.probability_of_profit,
            "expected_upside": ev.expected_upside,
            "expected_downside": ev.expected_downside,
            "upside_downside_ratio": ev.upside_downside_ratio,
            "worst_case_payoff": ev.worst_case_payoff,
            "best_case_payoff": ev.best_case_payoff,
            "breakeven_points": ev.breakeven_points,
        })

    df = pd.DataFrame(records).set_index("strategy")
    return df.sort_values("expected_payoff", ascending=False)


# ---------------------------------------------------------------------------
# Multi-strategy visualisation
# ---------------------------------------------------------------------------

def plot_strategy_comparison(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    strategies: dict[str, list[OptionLeg]],
    grid_size: int = 3000,
    tail_probability: float = 0.001,
    title: str | None = None,
):
    """Plot expiry P&L curves for multiple strategies on shared axes.

    The top panel shows the risk-neutral terminal spot density; the bottom
    panel overlays each strategy's payoff at expiry.

    Returns
    -------
    fig, axes
    """
    st = _terminal_spot_grid(
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        grid_size=grid_size,
        tail_probability=tail_probability,
    )

    sigma_t = volatility * np.sqrt(time_to_expiry)
    mu = np.log(spot) + (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    pdf = lognorm.pdf(st, s=sigma_t, scale=np.exp(mu))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    # --- terminal density ---
    axes[0].fill_between(st, pdf, alpha=0.25, color="tab:blue")
    axes[0].plot(st, pdf, color="tab:blue", lw=1.5)
    axes[0].axvline(spot, color="tab:gray", ls="--", lw=1.5, label=f"Spot {spot:,.0f}")
    axes[0].set_ylabel("Risk-neutral density")
    axes[0].legend(loc="upper right", fontsize=9)

    # --- payoff curves ---
    colors = plt.cm.tab10.colors
    for i, (name, legs) in enumerate(strategies.items()):
        payoff = strategy_payoff_at_expiry(st, legs)
        axes[1].plot(st, payoff, lw=2, color=colors[i % len(colors)], label=name)

    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].axvline(spot, color="tab:gray", ls="--", lw=1.5)
    axes[1].set_xlabel("Terminal spot at expiry")
    axes[1].set_ylabel("Payoff per contract ($)")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(title or "Strategy comparison — expiry payoffs", fontsize=13)
    return fig, axes
