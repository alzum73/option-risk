


"""
yield_curves.py
----------------
Robust yield curve classes for pricing and discounting, including live SOFR and €STR fetchers.

Features:
- Flat and piecewise yield curves
- Year fraction, zero, forward, and discount factor calculations
- SOFR (Fed) and €STR (ECB) rate retrieval
- Continuous compounding
- Plotting helpers for term structure visualization
"""

from __future__ import annotations
from datetime import date, timedelta
import math, re, requests, pandas as pd, matplotlib.pyplot as plt
from tracemalloc import start
from dataclasses import dataclass
from bisect import bisect_right

# ---------------------- #
#   Utility Functions    #
# ---------------------- #

def year_fraction(d1: date, d2: date, convention: str = "ACT/365F") -> float:
    """Compute year fraction between two dates."""
    days = (d2 - d1).days
    if convention.upper().startswith("ACT/365"):
        return days / 365.0
    elif convention.upper().startswith("ACT/360"):
        return days / 360.0
    else:
        raise ValueError("Unsupported day count convention")


def parse_tenor(tenor: str) -> timedelta:
    """Convert tenor like '3M', '1Y' into timedelta."""
    match = re.match(r"(\d+)([DWMY])", tenor.upper())
    if not match:
        raise ValueError(f"Invalid tenor format: {tenor}")
    n, unit = int(match.group(1)), match.group(2)
    if unit == "D": return timedelta(days=n)
    if unit == "W": return timedelta(weeks=n)
    if unit == "M": return timedelta(days=30 * n)
    if unit == "Y": return timedelta(days=365 * n)


# ---------------------- #
#   Base Yield Curve     #
# ---------------------- #

@dataclass
class YieldCurve:
    valuation_date: date
    day_count: str = "ACT/365F"
    compounding: str = "cont"

    def year_frac(self, d: date) -> float:
        return year_fraction(self.valuation_date, d, self.day_count)

    def discount_factor(self, d: date) -> float:
        raise NotImplementedError

    def zero_rate(self, d: date) -> float:
        """Return zero-coupon yield at date `d`."""
        t = self.year_frac(d)
        if t <= 0: return 0.0
        df = self.discount_factor(d)
        return -math.log(df) / t if self.compounding == "cont" else (1/df - 1)/t

    def forward_rate(self, d1: date, d2: date) -> float:
        """Compute instantaneous forward rate between d1 and d2."""
        df1, df2 = self.discount_factor(d1), self.discount_factor(d2)
        t1, t2 = self.year_frac(d1), self.year_frac(d2)
        return math.log(df1 / df2) / (t2 - t1)

    def plot(self, max_years: float = 5.0, points: int = 50):
        """Visualize zero rate term structure."""
        import numpy as np
        times = np.linspace(0.0, max_years, points)
        rates = []
        for t in times:
            d = self.valuation_date + timedelta(days=int(t * 365))
            rates.append(self.zero_rate(d) * 100)
        plt.plot(times, rates, label="Zero Curve")
        plt.xlabel("Years")
        plt.ylabel("Zero Rate (%)")
        plt.title("Yield Curve")
        plt.grid(True)
        plt.legend()
        plt.show()


# ---------------------- #
#   Flat Yield Curve     #
# ---------------------- #

@dataclass
class FlatYieldCurve(YieldCurve):
    rate: float = 0.0  # annual continuously compounded rate

    def discount_factor(self, d: date) -> float:
        t = self.year_frac(d)
        if self.compounding == "cont":
            return math.exp(-self.rate * t)
        else:
            return 1 / (1 + self.rate * t)


# ---------------------- #
#   Piecewise Zero Curve #
# ---------------------- #

@dataclass
class PiecewiseLinearZeroCurve(YieldCurve):
    pillars: list[tuple[str, float]] = None  # e.g. [("1M", 0.05), ("1Y", 0.045)]

    def __post_init__(self):
        self.times, self.rates = [], []
        for tenor, rate in self.pillars:
            d = self.valuation_date + parse_tenor(tenor)
            t = self.year_frac(d)
            self.times.append(t)
            self.rates.append(rate)
        self._max_t = self.times[-1]

    def _interp(self, t: float) -> float:
        if t <= 0: return self.rates[0]
        if t >= self._max_t: return self.rates[-1]
        for i in range(1, len(self.times)):
            if t <= self.times[i]:
                t1, t2 = self.times[i-1], self.times[i]
                r1, r2 = self.rates[i-1], self.rates[i]
                return r1 + (r2 - r1) * (t - t1) / (t2 - t1)

    def discount_factor(self, d: date) -> float:
        t = self.year_frac(d)
        r = self._interp(t)
        return math.exp(-r * t)


# ---------------------- #
# Risk-Free Fetchers     #
# ---------------------- #

class RiskFreeCurveFactory:
    """Fetches SOFR or €STR and returns a flat curve."""

    @staticmethod
    def fetch_sofr() -> float:
        """Fetch the most recent SOFR rate from the New York Fed."""
        start = "2025-11-05"
        url = f"https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate={start}&endDate={start}"
        r = requests.get(url).json()
        rate = float(r["refRates"][-1]["percentRate"]) / 100
        return rate

    @staticmethod
    def fetch_estr() -> float:
        """Fetch the most recent €STR rate from the ECB."""
        url = "https://data.ecb.europa.eu/api/v2/data/EST.B.EU000A2X2A25.ESTR.RT.H.A?format=csvdata"
        df = pd.read_csv(url)
        return float(df["OBS_VALUE"].iloc[-1]) / 100

    @classmethod
    def create(cls, currency: str = "USD") -> FlatYieldCurve:
        today = date.today()
        if currency.upper() == "USD":
            r = cls.fetch_sofr()
        elif currency.upper() in ("EUR", "EURO"):
            r = cls.fetch_estr()
        else:
            raise ValueError("Only USD (SOFR) and EUR (€STR) supported")
        return FlatYieldCurve(valuation_date=today, rate=r)
