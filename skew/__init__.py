"""skew – option strategy decision framework.

Public API
----------
Core dataclasses
    OptionLeg, StrategyEvaluation

Probability & payoff
    probability_itm
    probability_weighted_payoff

Risk/reward diagnostics
    evaluate_risk_reward_asymmetry
    compare_strategies

Strategy builders
    build_straddle
    build_strangle
    build_vertical_spread
    build_iron_condor
    build_butterfly
    build_legs_from_chain

Visualisation
    plot_strategy_distribution
    plot_strategy_comparison

Greeks / smile utilities
    iv_from_price, rr_bf_from_chain, evaluate_strategy_from_chain
    bs_greeks_full, compute_option_metrics, portfolio_greeks
    market_implied_pdf, prob_profit_from_pdf

Data store
    save_option_snapshot, load_option_snapshots, list_snapshots

Yield curves
    FlatYieldCurve, PiecewiseLinearZeroCurve, RiskFreeCurveFactory
"""

from skew.decision_framework import (
    OptionLeg,
    StrategyEvaluation,
    probability_itm,
    probability_weighted_payoff,
    evaluate_risk_reward_asymmetry,
    compare_strategies,
    build_straddle,
    build_strangle,
    build_vertical_spread,
    build_iron_condor,
    build_butterfly,
    build_legs_from_chain,
    plot_strategy_distribution,
    plot_strategy_comparison,
)

from skew.utils import (
    iv_from_price,
    rr_bf_from_chain,
    evaluate_strategy_from_chain,
    bs_greeks_full,
    compute_option_metrics,
    portfolio_greeks,
    market_implied_pdf,
    prob_profit_from_pdf,
)

from skew.data_store import (
    save_option_snapshot,
    load_option_snapshots,
    list_snapshots,
)

from skew.zero_curve import (
    FlatYieldCurve,
    PiecewiseLinearZeroCurve,
    RiskFreeCurveFactory,
)

__all__ = [
    # dataclasses
    "OptionLeg",
    "StrategyEvaluation",
    # probability & payoff
    "probability_itm",
    "probability_weighted_payoff",
    # risk/reward
    "evaluate_risk_reward_asymmetry",
    "compare_strategies",
    # builders
    "build_straddle",
    "build_strangle",
    "build_vertical_spread",
    "build_iron_condor",
    "build_butterfly",
    "build_legs_from_chain",
    # visualisation
    "plot_strategy_distribution",
    "plot_strategy_comparison",
    # utilities
    "iv_from_price",
    "rr_bf_from_chain",
    "evaluate_strategy_from_chain",
    "bs_greeks_full",
    "compute_option_metrics",
    "portfolio_greeks",
    "market_implied_pdf",
    "prob_profit_from_pdf",
    # data store
    "save_option_snapshot",
    "load_option_snapshots",
    "list_snapshots",
    # yield curves
    "FlatYieldCurve",
    "PiecewiseLinearZeroCurve",
    "RiskFreeCurveFactory",
]
