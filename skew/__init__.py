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
    bs_greeks_full,
    compute_option_metrics,
    portfolio_greeks,
    iv_from_price,
    rr_bf_from_chain,
    evaluate_strategy_from_chain,
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
    "bs_greeks_full",
    "compute_option_metrics",
    "portfolio_greeks",
    "iv_from_price",
    "rr_bf_from_chain",
    "evaluate_strategy_from_chain",
    # yield curves
    "FlatYieldCurve",
    "PiecewiseLinearZeroCurve",
    "RiskFreeCurveFactory",
]
