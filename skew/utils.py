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

# --------- Full BS Greeks (spot measure, with dividend yield) ----------

def bs_greeks_full(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> dict:
    """Black-Scholes Greeks including continuous dividend yield.

    Parameters
    ----------
    option_type : 'call' or 'put' (case-insensitive, only first letter checked)
    S, K        : spot and strike
    T           : time to expiry in years (clamped to 1e-8)
    r           : continuously compounded risk-free rate
    q           : continuous dividend yield
    sigma       : implied volatility (clamped to 1e-8)

    Returns
    -------
    dict
        d1, d2, delta, gamma,
        vega      (per +1 vol-point, i.e. ÷100),
        theta_day (per calendar day, i.e. ÷365),
        rho       (per +1 percentage-point rate move, i.e. ÷100),
        prob_itm  (risk-neutral N(d2) probability of expiring ITM)
    """
    T     = max(float(T), 1e-8)
    sigma = max(float(sigma), 1e-8)

    _d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    _d2 = _d1 - sigma * np.sqrt(T)

    eq  = np.exp(-q * T)
    er  = np.exp(-r * T)
    nd1 = norm.pdf(_d1)

    gamma = eq * nd1 / (S * sigma * np.sqrt(T))
    vega  = S * eq * nd1 * np.sqrt(T) / 100.0   # per vol point

    is_call = option_type.lower().startswith('c')
    if is_call:
        delta    = eq * norm.cdf(_d1)
        theta    = (
            -S * eq * nd1 * sigma / (2.0 * np.sqrt(T))
            - r * K * er * norm.cdf(_d2)
            + q * S * eq * norm.cdf(_d1)
        ) / 365.0
        rho      = K * T * er * norm.cdf(_d2)  / 100.0
        prob_itm = float(norm.cdf(_d2))
    else:
        delta    = -eq * norm.cdf(-_d1)
        theta    = (
            -S * eq * nd1 * sigma / (2.0 * np.sqrt(T))
            + r * K * er * norm.cdf(-_d2)
            - q * S * eq * norm.cdf(-_d1)
        ) / 365.0
        rho      = -K * T * er * norm.cdf(-_d2) / 100.0
        prob_itm = float(norm.cdf(-_d2))

    return {
        'd1': float(_d1), 'd2': float(_d2),
        'delta': float(delta), 'gamma': float(gamma),
        'vega': float(vega), 'theta_day': float(theta),
        'rho': float(rho), 'prob_itm': prob_itm,
    }


def compute_option_metrics(df, default_rf: float = 0.04, contract_size: int = 100):
    """Compute trading metrics and Greeks for a DataFrame of option quotes.

    Expected input columns
    ----------------------
    Required : option_type ('call'/'put'), strike, expiry, underlying_price,
               option_price, implied_vol
    Optional : risk_free_rate, div_yield, contracts

    Returns
    -------
    pd.DataFrame
        Original columns plus: days_to_expiry, t_years, intrinsic, extrinsic,
        moneyness, break_even, max_loss, premium_pct,
        prob_itm, prob_profit (net of premium paid),
        d1, d2, delta, gamma, vega, theta_day, rho,
        position_delta, position_gamma, position_vega, position_theta_day, position_rho
    """
    import pandas as pd

    out = df.copy()
    out['option_type'] = out['option_type'].str.lower().str.strip()
    out['expiry'] = pd.to_datetime(out['expiry'], utc=True, errors='coerce')

    for col in ['strike', 'underlying_price', 'option_price', 'implied_vol']:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    if 'risk_free_rate' not in out.columns:
        out['risk_free_rate'] = default_rf
    else:
        out['risk_free_rate'] = pd.to_numeric(out['risk_free_rate'], errors='coerce').fillna(default_rf)

    if 'div_yield' not in out.columns:
        out['div_yield'] = 0.0
    else:
        out['div_yield'] = pd.to_numeric(out['div_yield'], errors='coerce').fillna(0.0)

    if 'contracts' not in out.columns:
        out['contracts'] = 1
    else:
        out['contracts'] = pd.to_numeric(out['contracts'], errors='coerce').fillna(1).clip(lower=0)

    now = pd.Timestamp.now(tz='UTC')
    out['days_to_expiry'] = (out['expiry'] - now).dt.total_seconds() / 86400.0
    out['days_to_expiry'] = out['days_to_expiry'].clip(lower=0)
    out['t_years'] = (out['days_to_expiry'] / 365.0).clip(lower=1e-8)

    is_call = out['option_type'] == 'call'
    out['intrinsic'] = np.where(
        is_call,
        np.maximum(out['underlying_price'] - out['strike'], 0.0),
        np.maximum(out['strike'] - out['underlying_price'], 0.0),
    )
    out['extrinsic']  = out['option_price'] - out['intrinsic']
    out['moneyness']  = out['underlying_price'] / out['strike']
    out['break_even'] = np.where(
        is_call,
        out['strike'] + out['option_price'],
        out['strike'] - out['option_price'],
    )
    out['max_loss']    = out['option_price'] * contract_size * out['contracts']
    out['premium_pct'] = (out['option_price'] / out['underlying_price']) * 100.0

    # Greeks row by row (handles edge cases internally)
    greek_rows = [
        bs_greeks_full(
            row.option_type,
            float(row.underlying_price),
            float(row.strike),
            float(row.t_years),
            float(row.risk_free_rate),
            float(row.div_yield),
            float(row.implied_vol),
        )
        for row in out.itertuples(index=False)
    ]
    greeks_df = pd.DataFrame(greek_rows)
    out = pd.concat([out.reset_index(drop=True), greeks_df.reset_index(drop=True)], axis=1)

    # Probability of profit net of premium:
    #   Call  →  P(S_T > K + premium)  =  N(d2 at breakeven strike)
    #   Put   →  P(S_T < K − premium)  =  N(−d2 at breakeven strike)
    S_arr   = out['underlying_price'].values.astype(float)
    be_arr  = out['break_even'].values.astype(float)
    T_arr   = out['t_years'].values.astype(float)
    r_arr   = out['risk_free_rate'].values.astype(float)
    q_arr   = out['div_yield'].values.astype(float)
    sig_arr = out['implied_vol'].values.astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        be_d2 = (np.log(S_arr / be_arr) + (r_arr - q_arr - 0.5 * sig_arr**2) * T_arr) / (
            sig_arr * np.sqrt(T_arr)
        )
    raw_pp = np.where(is_call.values, norm.cdf(be_d2), norm.cdf(-be_d2))
    out['prob_profit'] = np.where(np.isfinite(raw_pp), raw_pp, np.nan)

    # Position Greeks (per contract multiplier)
    for g in ['delta', 'gamma', 'vega', 'theta_day', 'rho']:
        out[f'position_{g}'] = out[g] * contract_size * out['contracts']

    return out


def portfolio_greeks(metrics_df) -> dict:
    """Aggregate portfolio-level Greeks from a compute_option_metrics result.

    Returns
    -------
    dict with keys: total_contracts, total_premium_at_risk, net_delta,
    net_gamma, net_vega_per_vol_pt, net_theta_per_day, net_rho_per_1pct_rate
    """
    return {
        'total_contracts':       float(metrics_df['contracts'].sum()),
        'total_premium_at_risk': float(metrics_df['max_loss'].sum()),
        'net_delta':             float(metrics_df['position_delta'].sum()),
        'net_gamma':             float(metrics_df['position_gamma'].sum()),
        'net_vega_per_vol_pt':   float(metrics_df['position_vega'].sum()),
        'net_theta_per_day':     float(metrics_df['position_theta_day'].sum()),
        'net_rho_per_1pct_rate': float(metrics_df['position_rho'].sum()),
    }


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
