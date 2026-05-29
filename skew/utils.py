import numpy as np
from scipy.stats import norm
from dataclasses import dataclass



# --- Black–Scholes (equity) d1, deltas ---

def d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, q, sigma):
    return d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

def forward_price(S, T, r, q):
    return S * np.exp((r - q) * T)

def bs_delta(spot, K, r, q, vol, T, is_call: bool):
    if spot <= 0 or K <= 0 or vol <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(spot / K) + (r - q + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    if is_call:
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return -np.exp(-q * T) * norm.cdf(-d1)

# --------- Black-Scholes utilities (forward measure) ----------
def bs_price_forward(is_call, F, K, sigma, T, D):
    if sigma <= 0 or T <= 0:
        return D * max((F-K) if is_call else (K-F), 0.0)
    vol = sigma * np.sqrt(T)
    d1 = np.log(F/K)/vol + 0.5*vol
    d2 = d1 - vol
    if is_call:
        return D * (F*norm.cdf(d1) - K*norm.cdf(d2))
    else:
        return D * (K*norm.cdf(-d2) - F*norm.cdf(-d1))

def bs_vega_forward(F, K, sigma, T, D):
    if sigma <= 0 or T <= 0: return 0.0
    vol = sigma * np.sqrt(T)
    d1 = np.log(F/K)/vol + 0.5*vol
    return D * F * np.sqrt(T) * norm.pdf(d1)

def bs_delta_forward(is_call, F, K, sigma, T):
    # forward premium-unadjusted deltas: Call Δ = N(d1), Put Δ = N(d1)-1
    if sigma <= 0 or T <= 0:
        return 1.0 if (is_call and F > K) else (0.0 if is_call else -1.0 if K>F else 0.0)
    vol = sigma * np.sqrt(T)
    d1 = np.log(F/K)/vol + 0.5*vol
    return norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)

def iv_from_price(is_call, price, F, K, T, D, guess=0.2, tol=1e-7, maxit=100):
    # Newton-Raphson on price
    sigma = max(1e-6, guess)
    for _ in range(maxit):
        f = bs_price_forward(is_call, F, K, sigma, T, D) - price
        v = bs_vega_forward(F, K, sigma, T, D)
        if abs(f) < tol: break
        if v < 1e-12: 
            # fallback bisection corridor
            lo, hi = 1e-6, 5.0
            for _ in range(60):
                mid = 0.5*(lo+hi)
                pmid = bs_price_forward(is_call, F, K, mid, T, D)
                (lo,hi) = (mid,hi) if pmid < price else (lo,mid)
            return 0.5*(lo+hi)
        sigma = max(1e-8, sigma - f / v)
    return float(sigma)

def iv_at_target_delta(df_with_delta, target_delta):
    """
    df_with_delta: must have columns ['delta','iv'] and be sorted by delta.
    We'll interpolate iv in delta-space.
    """
    tmp = df_with_delta.dropna(subset=["delta", "iv"]).sort_values("delta")
    if tmp.empty:
        return np.nan
    # if target is outside range, just take closest
    dmin, dmax = tmp["delta"].min(), tmp["delta"].max()
    if target_delta <= dmin:
        return tmp.iloc[0]["iv"]
    if target_delta >= dmax:
        return tmp.iloc[-1]["iv"]
    # interpolate
    return np.interp(target_delta, tmp["delta"].values, tmp["iv"].values)

# --------- 25Δ RR and BF per snapshot ----------
@dataclass
class SmilePoints:
    sigma_25C: float
    sigma_25P: float
    sigma_atm: float
    rr25: float
    bf25: float

def interpolate_sigma_at_strike(df, K_query):
    # linear in K on IVs (simple & robust)
    tmp = df[['K','iv']].dropna().sort_values('K')
    return np.interp(K_query, tmp['K'].values, tmp['iv'].values)

def strike_for_delta(df_calls_or_puts, target_delta, F, T):
    # Find strike whose *observed* delta is closest to target_delta, with local linear interpolation in delta.
    d = df_calls_or_puts.copy()
    d = d[['K','iv']].dropna().sort_values('K')
    if d.empty: return None
    # compute deltas using each row's own IV
    d['delta'] = d.apply(lambda r: bs_delta_forward(df_calls_or_puts.name=='C', F, r.K, r.iv, T), axis=1)
    xs, ys = d['delta'].values, d['K'].values  # invert delta->K via local interpolation
    # enforce monotone xs for interp
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # clamp
    target = np.clip(target_delta, xs.min(), xs.max())
    return float(np.interp(target, xs, ys))

def rr_bf_from_chain(chain_df):
    """
    chain_df: single [date, expiry] slice with columns:
      K, opt_type in {'C','P'}, iv, T, r, q, F  (consistent within slice)
    """
    df = chain_df.copy()
    assert df['T'].nunique()==1 and df['F'].nunique()==1 and df['r'].nunique()==1 and df['q'].nunique()==1
    T = df['T'].iloc[0]; F = df['F'].iloc[0]
    # ATM-forward vol at K=F
    sigma_atm = interpolate_sigma_at_strike(df, F)

    calls = df[df['opt_type']=='C'].copy(); calls.name = 'C'
    puts  = df[df['opt_type']=='P'].copy(); puts.name  = 'P'

    K_25C = strike_for_delta(calls, 0.25, F, T)
    K_25P = strike_for_delta(puts, -0.25, F, T)

    if K_25C is None or K_25P is None:
        raise ValueError("Insufficient strikes to locate 25Δ points.")

    sigma_25C = interpolate_sigma_at_strike(df, K_25C)
    sigma_25P = interpolate_sigma_at_strike(df, K_25P)

    rr25 = sigma_25C - sigma_25P
    bf25 = 0.5*(sigma_25C + sigma_25P) - sigma_atm
    return SmilePoints(sigma_25C, sigma_25P, sigma_atm, rr25, bf25)

# --------- Strategy decision framework integration ----------
def evaluate_strategy_from_chain(
    chain_df,
    leg_specs,
    spot,
    T,
    r,
    q,
    vol,
    premium_col="mid",
):
    """Convenience wrapper to evaluate a strategy from fetched chain data.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Option chain slice with at least: opt_type, K, and a premium column.
    leg_specs : list[tuple[str, float, float]]
        Sequence of (option_type, strike, quantity), e.g.
        [('C', 105, +1), ('C', 110, -1)] for a call spread.
    spot, T, r, q, vol : float
        Market inputs for terminal distribution assumptions.
    premium_col : str
        Premium column in `chain_df` (default: "mid").

    Returns
    -------
    StrategyEvaluation
        Probability-weighted payoff and risk/reward asymmetry diagnostics.
    """
    from skew.decision_framework import build_legs_from_chain, evaluate_risk_reward_asymmetry

    legs = build_legs_from_chain(chain_df=chain_df, leg_specs=leg_specs, premium_col=premium_col)
    return evaluate_risk_reward_asymmetry(
        spot=spot,
        time_to_expiry=T,
        risk_free_rate=r,
        dividend_yield=q,
        volatility=vol,
        legs=legs,
    )
