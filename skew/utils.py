import numpy as np
import pandas as pd
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

    rr25 = sigma_25P - sigma_25C   # equity convention: positive = put skew (fear)
    bf25 = 0.5*(sigma_25C + sigma_25P) - sigma_atm
    return SmilePoints(sigma_25C, sigma_25P, sigma_atm, rr25, bf25)

# --------- Strategy decision framework integration ----------
# --------- Full Black-Scholes Greeks ---------
def bs_greeks_full(option_type: str, S: float, K: float, T: float, r: float, q: float, sigma: float) -> dict:
    """Return a dict of greeks and diagnostics for one option.

    option_type : 'C' or 'P'
    Returns keys: d1, d2, delta, gamma, vega, theta_day, rho, prob_itm
    """
    is_call = option_type.upper() == "C"
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return dict(d1=np.nan, d2=np.nan, delta=np.nan, gamma=np.nan,
                    vega=np.nan, theta_day=np.nan, rho=np.nan, prob_itm=np.nan)
    sqrt_T = np.sqrt(T)
    _d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    _d2 = _d1 - sigma * sqrt_T
    Nd1  = norm.cdf(_d1 if is_call else -_d1)
    Nd2  = norm.cdf(_d2 if is_call else -_d2)
    nd1  = norm.pdf(_d1)
    sign = 1.0 if is_call else -1.0
    delta = sign * np.exp(-q * T) * Nd1
    gamma = np.exp(-q * T) * nd1 / (S * sigma * sqrt_T)
    vega  = S * np.exp(-q * T) * nd1 * sqrt_T / 100.0   # per 1 vol-pt
    theta = (
        -(S * np.exp(-q * T) * nd1 * sigma) / (2 * sqrt_T)
        - sign * r * K * np.exp(-r * T) * Nd2
        + sign * q * S * np.exp(-q * T) * Nd1
    ) / 365.0
    rho = sign * K * T * np.exp(-r * T) * Nd2 / 100.0   # per 1 rate pct-pt
    return dict(d1=_d1, d2=_d2, delta=delta, gamma=gamma,
                vega=vega, theta_day=theta, rho=rho, prob_itm=Nd2)


def compute_option_metrics(df, default_rf: float = 0.04, contract_size: int = 100):
    """Compute full set of option metrics from a chain DataFrame.

    Expected columns (from yfinance fetch):
        isCall / option_type, strike, T, lastPrice / option_price,
        bid, ask, forward, div_yield (optional), disc_factor (optional)

    Returns the same DataFrame with extra computed columns.
    """
    out = df.copy()

    # Normalise column names
    if "isCall" in out.columns and "option_type" not in out.columns:
        out["option_type"] = out["isCall"].map({True: "C", False: "P"})
    else:
        out["option_type"] = out["option_type"].str.upper().str.strip().replace(
            {"CALL": "C", "PUT": "P"}
        )

    if "strike" not in out.columns and "K" in out.columns:
        out = out.rename(columns={"K": "strike"})
    if "option_price" in out.columns and "lastPrice" not in out.columns:
        out = out.rename(columns={"option_price": "lastPrice"})

    # Derived market inputs
    S   = out["spot"].values if "spot" in out.columns else out["underlying_price"].values
    out["spot"] = S   # always present regardless of source column name
    K   = out["strike"].values
    T   = out["T"].values
    r   = out.get("r", pd.Series([default_rf] * len(out))).values if "r" in out.columns else np.full(len(out), default_rf)
    q   = out["div_yield"].values if "div_yield" in out.columns else np.zeros(len(out))
    sig = out["implied_vol"].values if "implied_vol" in out.columns else out["iv"].values

    mid_col = "lastPrice" if "lastPrice" in out.columns else "option_price"
    mid_arr = out[mid_col].values.astype(float)
    bid_arr = out["bid"].values.astype(float) if "bid" in out.columns else np.full(len(out), np.nan)
    ask_arr = out["ask"].values.astype(float) if "ask" in out.columns else np.full(len(out), np.nan)

    is_call = out["option_type"].values == "C"

    # contracts column: positive = long, negative = short
    contracts_arr = (
        out["contracts"].values.astype(float)
        if "contracts" in out.columns
        else np.ones(len(out))
    )
    is_short = contracts_arr < 0

    # Intrinsic & extrinsic
    F = out["forward"].values if "forward" in out.columns else S * np.exp((r - q) * T)
    intrinsic      = np.where(is_call, np.maximum(S - K, 0), np.maximum(K - S, 0))
    out["intrinsic"]   = intrinsic
    out["extrinsic"]   = np.maximum(mid_arr - intrinsic, 0)
    out["moneyness"]   = S / K
    # break-even is symmetric: for longs it's the min move needed to profit;
    # for shorts it's the max move the seller can absorb before losing money
    out["break_even"]  = np.where(is_call, K + mid_arr, K - mid_arr)
    # max_loss: long → premium paid; short call → unlimited; short put → strike - premium
    out["max_loss"] = np.where(
        is_short,
        np.where(is_call, np.inf, np.maximum(K - mid_arr, 0) * contract_size),
        mid_arr * contract_size,
    )
    out["premium_pct"] = mid_arr / S * 100.0

    # Greeks
    greeks_rows = [
        bs_greeks_full(opt, s, k, t, ri, qi, v)
        for opt, s, k, t, ri, qi, v in zip(
            out["option_type"].values, S, K, T, r, q, sig
        )
    ]
    for col in ("d1", "d2", "delta", "gamma", "vega", "theta_day", "rho", "prob_itm"):
        out[col] = [row[col] for row in greeks_rows]

    # P(profit) using mid breakeven
    # Long:  need underlying to move past break-even → same direction as ITM
    # Short: profit when underlying stays within collected premium → complementary probability
    sqrt_T   = np.sqrt(np.maximum(T, 1e-12))
    sig_safe = np.where(sig > 0, sig, 1e-8)
    be_mid   = np.where(is_call, K + mid_arr, K - mid_arr)
    be_mid   = np.where(be_mid > 0, be_mid, 1e-8)
    be_d2_mid = (np.log(S / be_mid) + (r - q - 0.5 * sig_safe ** 2) * T) / (sig_safe * sqrt_T)
    long_pp  = np.where(is_call, norm.cdf(be_d2_mid),  norm.cdf(-be_d2_mid))
    short_pp = np.where(is_call, norm.cdf(-be_d2_mid), norm.cdf(be_d2_mid))
    out["prob_profit"] = np.where(is_short, short_pp, long_pp)

    # P(profit) using ask / bid (conservative fill cost)
    # Longs pay ask; shorts receive bid (worst-case fill is bid for the seller)
    bid_entry = np.where(np.isfinite(bid_arr) & (bid_arr > 0), bid_arr, mid_arr)
    ask_entry = np.where(np.isfinite(ask_arr) & (ask_arr > 0), ask_arr, mid_arr)
    be_ask = np.where(is_call, K + ask_entry, K - ask_entry)
    be_ask = np.where(be_ask > 0, be_ask, 1e-8)
    be_d2_ask = (np.log(S / be_ask) + (r - q - 0.5 * sig_safe ** 2) * T) / (sig_safe * sqrt_T)
    be_bid = np.where(is_call, K + bid_entry, K - bid_entry)
    be_bid = np.where(be_bid > 0, be_bid, 1e-8)
    be_d2_bid = (np.log(S / be_bid) + (r - q - 0.5 * sig_safe ** 2) * T) / (sig_safe * sqrt_T)
    long_pp_ask  = np.where(is_call, norm.cdf(be_d2_ask),  norm.cdf(-be_d2_ask))
    short_pp_bid = np.where(is_call, norm.cdf(-be_d2_bid), norm.cdf(be_d2_bid))
    out["prob_profit_ask"] = np.where(is_short, short_pp_bid, long_pp_ask)

    # Position Greeks: signed by contracts (negative contracts = short = negative exposure)
    for g in ("delta", "gamma", "vega", "theta_day", "rho"):
        out[f"pos_{g}"] = out[g] * contract_size * contracts_arr

    return out


def portfolio_greeks(metrics_df) -> dict:
    """Aggregate position Greeks across all legs in a metrics DataFrame."""
    cols = ["pos_delta", "pos_gamma", "pos_vega", "pos_theta_day", "pos_rho"]
    totals = {c: metrics_df[c].sum() if c in metrics_df.columns else np.nan for c in cols}
    return {k.replace("pos_", ""): v for k, v in totals.items()}


def smile_prob_profit(
    chain_slice: pd.DataFrame,
    forward: float,
    T: float,
    disc_factor: float,
    break_even: float,
    is_call: bool,
    is_short: bool = False,
    n_grid: int = 400,
) -> float:
    """Probability of profit using the Breeden-Litzenberger risk-neutral PDF.

    Fits the OTM vol smile at the option's own expiry (no cross-maturity
    interpolation), builds a smooth call-price surface, then integrates
    the implied PDF over the profitable strike region.

    Returns NaN when fewer than 3 OTM quotes are available so the caller
    can fall back to the Black-Scholes approximation.

    Parameters
    ----------
    chain_slice : option chain rows for a **single** expiry; must have
                  columns ``option_type`` (C/P), ``strike``, ``implied_vol``
    forward     : forward price of the underlying at this expiry
    T           : time to expiry in years
    disc_factor : discount factor e^{-rT}
    break_even  : underlying price at which P&L = 0 at expiry
    is_call     : True for call, False for put
    is_short    : True if the position is short (contracts < 0)
    n_grid      : number of strike grid points for the PDF
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import PchipInterpolator

    # ── 1. Normalise chain slice ───────────────────────────────────────────────
    sl = chain_slice.copy()
    if "implied_vol" not in sl.columns and "iv" in sl.columns:
        sl = sl.rename(columns={"iv": "implied_vol"})
    if "option_type" not in sl.columns or "implied_vol" not in sl.columns:
        return np.nan

    sl["option_type"] = (sl["option_type"].astype(str).str.upper().str.strip()
                         .replace({"CALL": "C", "PUT": "P"}))
    calls = sl[(sl["option_type"] == "C") & (sl["strike"] >= forward)]
    puts  = sl[(sl["option_type"] == "P") & (sl["strike"] <  forward)]
    otm   = (pd.concat([puts, calls])
               .dropna(subset=["implied_vol", "strike"])
               .pipe(lambda d: d[d["implied_vol"].between(0.01, 5.0)])
               .sort_values("strike"))

    if len(otm) < 3:
        return np.nan

    # ── 2. Strike grid (data range + flat-wing buffer) ────────────────────────
    k_data = np.log(otm["strike"].values / forward)
    wing   = 0.10
    k_lo, k_hi = k_data.min() - wing, k_data.max() + wing
    k_grid     = np.linspace(k_lo, k_hi, n_grid)
    K_grid     = forward * np.exp(k_grid)

    # ── 3. IV on grid: SVI fit, then PCHIP fallback ───────────────────────────
    iv_grid = None
    if len(otm) >= 5:
        params = fit_svi_slice(otm["strike"].values, otm["implied_vol"].values, forward, T)
        if params is not None:
            iv_svi = eval_svi_iv(params, K_grid, forward, T)
            iv_lo  = float(eval_svi_iv(params, [forward * np.exp(k_lo)], forward, T)[0])
            iv_hi  = float(eval_svi_iv(params, [forward * np.exp(k_hi)], forward, T)[0])
            # flat wings outside the data range
            iv_grid = np.where(k_grid < k_data.min(), iv_lo,
                      np.where(k_grid > k_data.max(), iv_hi, iv_svi))

    if iv_grid is None:
        pchip   = PchipInterpolator(otm["strike"].values, otm["implied_vol"].values,
                                    extrapolate=False)
        iv_grid = pchip(K_grid).astype(float)
        valid   = np.where(np.isfinite(iv_grid))[0]
        if len(valid) == 0:
            return np.nan
        iv_grid[:valid[0]]     = iv_grid[valid[0]]
        iv_grid[valid[-1]+1:]  = iv_grid[valid[-1]]

    iv_grid = gaussian_filter1d(np.clip(iv_grid, 1e-4, 5.0), sigma=2)

    # ── 4. Call-price surface ─────────────────────────────────────────────────
    C = np.array([bs_price_forward(True, forward, K, iv, T, disc_factor)
                  for K, iv in zip(K_grid, iv_grid)])
    C = gaussian_filter1d(C, sigma=2)

    # ── 5. Breeden-Litzenberger: f(K) = (1/D) × ∂²C/∂K² ─────────────────────
    h     = K_grid[1] - K_grid[0]
    Cpp           = np.empty_like(C)
    Cpp[1:-1]     = (C[2:] - 2*C[1:-1] + C[:-2]) / (h * h)
    Cpp[0]        = Cpp[1]
    Cpp[-1]       = Cpp[-2]
    pdf   = np.clip(Cpp / max(disc_factor, 1e-16), 0.0, None)
    area  = np.trapz(pdf, K_grid)
    if area < 1e-12:
        return np.nan
    pdf /= area  # normalise: ∫pdf dK = 1

    # ── 6. Integrate over profitable region ───────────────────────────────────
    # Long call / short put → profit if S_T > break_even
    # Long put / short call → profit if S_T < break_even
    profit_above = (is_call != is_short)  # XOR
    mask = K_grid >= break_even if profit_above else K_grid <= break_even

    if not np.any(mask):
        return 0.0
    if np.all(mask):
        return 1.0
    return float(np.trapz(pdf[mask], K_grid[mask]))


def market_implied_pdf(chain_df, spot: float, r: float, T: float,
                       price_col: str = "option_price", n_grid: int = 500):
    """Breeden-Litzenberger market-implied risk-neutral PDF.

    Uses OTM puts (K ≤ spot) and OTM calls (K ≥ spot).
    Since ∂²C/∂K² = ∂²P/∂K², we can stitch together the OTM side of each.

    Returns
    -------
    K_fine  : np.ndarray of strikes
    pdf     : np.ndarray, normalised risk-neutral density
    """
    from scipy.interpolate import CubicSpline

    df = chain_df.copy()
    # normalise option_type to 'C'/'P'
    if "isCall" in df.columns and "option_type" not in df.columns:
        df["option_type"] = df["isCall"].map({True: "C", False: "P"})
    col = df["option_type"].str.upper().str.strip().replace({"CALL": "C", "PUT": "P"})
    df["option_type"] = col

    if price_col not in df.columns and "lastPrice" in df.columns:
        df[price_col] = df["lastPrice"]

    df = df.dropna(subset=["strike", price_col])
    df = df[df[price_col] > 0]

    strike_col = "strike" if "strike" in df.columns else "K"

    otm_puts  = df[(df["option_type"] == "P") & (df[strike_col] <= spot)].copy()
    otm_calls = df[(df["option_type"] == "C") & (df[strike_col] >= spot)].copy()

    if len(otm_puts) < 3 or len(otm_calls) < 3:
        raise ValueError("Insufficient OTM options to fit market-implied PDF.")

    puts_sorted  = otm_puts.sort_values(strike_col)
    calls_sorted = otm_calls.sort_values(strike_col)

    K_obs = np.concatenate([puts_sorted[strike_col].values, calls_sorted[strike_col].values])
    P_obs = np.concatenate([puts_sorted[price_col].values, calls_sorted[price_col].values])

    # deduplicate
    _, idx = np.unique(K_obs, return_index=True)
    K_obs, P_obs = K_obs[idx], P_obs[idx]

    cs = CubicSpline(K_obs, P_obs)
    K_fine = np.linspace(K_obs.min(), K_obs.max(), n_grid)
    d2_dK2 = cs(K_fine, 2)
    pdf = np.exp(r * T) * np.maximum(d2_dK2, 0.0)
    norm_factor = np.trapz(pdf, K_fine)
    if norm_factor <= 0:
        raise ValueError("PDF integrates to zero — check option price data.")
    return K_fine, pdf / norm_factor


def prob_profit_from_pdf(pdf_strikes, pdf_values, breakeven: float, option_type: str) -> float:
    """Integrate the market-implied PDF to get P(profit) for one option.

    Parameters
    ----------
    pdf_strikes : array of strike values from market_implied_pdf
    pdf_values  : array of density values from market_implied_pdf
    breakeven   : strike at which the option starts to be profitable
    option_type : 'C' or 'P'
    """
    K = np.asarray(pdf_strikes)
    q = np.asarray(pdf_values)
    if option_type.upper() == "C":
        mask = K >= breakeven
    else:
        mask = K <= breakeven
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(q[mask], K[mask]))


def fit_svi_slice(strikes, ivs, forward, T):
    """
    Fit a raw-SVI smile to one expiry slice.

    Use OTM options only (calls above forward, puts below) for best results.
    Guarantees a convex total-variance smile by construction.

    Parameters
    ----------
    strikes : array-like  liquid strikes
    ivs     : array-like  market implied vols (same units as Black-Scholes σ)
    forward : float       forward price for this expiry
    T       : float       years to expiry

    Returns
    -------
    params : ndarray (5,) = [a, b, rho, m, sigma]  or None on failure
    """
    from scipy.optimize import least_squares

    k = np.log(np.asarray(strikes, float) / forward)
    w = np.asarray(ivs, float) ** 2 * T          # total variance

    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if len(k) < 5:
        return None

    def _svi(k_, a, b, rho, m, sig):
        return a + b * (rho * (k_ - m) + np.sqrt((k_ - m) ** 2 + sig ** 2))

    # ATM total variance as starting level
    sort_idx = np.argsort(k)
    w_atm = float(np.interp(0.0, k[sort_idx], w[sort_idx]))
    x0 = [max(w_atm * 0.5, 1e-4), 0.1, -0.2, 0.0, 0.15]
    lo = [-np.inf, 1e-6, -0.9999, -2.0, 1e-4]
    hi = [np.inf,  5.0,  0.9999,  2.0, 5.0]

    try:
        res = least_squares(
            lambda p: _svi(k, *p) - w, x0,
            bounds=(lo, hi), method="trf", max_nfev=2000,
        )
        # Accept fit if cost is reasonable (normalised per point)
        if res.cost / max(len(k), 1) < 0.5:
            return res.x
    except Exception:
        pass
    return None


def eval_svi_iv(params, strikes, forward, T):
    """
    Evaluate SVI implied vol at arbitrary strikes, interpolating AND extrapolating.

    Parameters
    ----------
    params  : array-like (5,)  output of fit_svi_slice
    strikes : array-like
    forward : float
    T       : float years to expiry

    Returns
    -------
    ivs : ndarray (same length as strikes)
    """
    a, b, rho, m, sig = params
    k = np.log(np.asarray(strikes, float) / forward)
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))
    return np.sqrt(np.maximum(w, 1e-8) / T)


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
