"""Persistent SQLite store for option chain snapshots.

Each call to save_option_snapshot appends today's chain as a new snapshot.
Running the fetch notebook daily builds a local historical options database.

Yahoo Finance note
------------------
yfinance only provides *today's* option chains — there is no API parameter
to request a historical date.  For historical option data you need paid
services (Polygon.io, CBOE DataShop, OptionsDX, Intrinio, etc.).
Running save_option_snapshot daily accumulates your own history in the DB.

Storage layout
--------------
Single SQLite file (default: data/options.db).
Table: option_snapshots
  - One row per (ticker, snapshot_date, expiry, strike, option_type)
  - All chain columns are stored as-is; column names are normalised on write
  - Snapshots are deduplicated by the unique constraint on the 5-tuple above
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DB = "data/options.db"
_TABLE      = "option_snapshots"

# Map vendor-specific column names → canonical names used throughout skew
_COLUMN_ALIASES: dict[str, str] = {
    "lastPrice":  "option_price",
    "iv":         "implied_vol",
    "isCall":     "option_type",
    "spot":       "underlying_price",
    "K":          "strike",
    "opt_type":   "option_type",
    "type":       "option_type",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename vendor columns and convert boolean option_type to call/put."""
    out = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    if "option_type" in out.columns:
        col = out["option_type"]
        if col.dtype == bool or set(col.dropna().unique()).issubset({True, False}):
            out["option_type"] = col.map({True: "call", False: "put"})
        else:
            out["option_type"] = col.str.lower().str.strip()
    # Serialise timezone-aware datetimes so SQLite can store them
    for c in out.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]"]).columns:
        out[c] = out[c].astype(str)
    return out


def save_option_snapshot(
    df: pd.DataFrame,
    ticker: str,
    db_path: str = _DEFAULT_DB,
    source: str  = "yfinance",
    snapshot_date: str | None = None,
) -> int:
    """Append an option chain snapshot to the SQLite store.

    Adds ``ticker``, ``source``, and ``snapshot_date`` columns automatically.
    Duplicate rows (same ticker + snapshot_date + expiry + strike + option_type)
    are silently skipped.

    Parameters
    ----------
    df            : raw option chain DataFrame (yfinance or IBKR format)
    ticker        : underlying ticker symbol (e.g. 'NVDA')
    db_path       : path to the SQLite file; created automatically if absent
    source        : data source label stored in the DB ('yfinance' or 'ibkr')
    snapshot_date : override date string (YYYY-MM-DD); defaults to today

    Returns
    -------
    int : number of new rows actually inserted
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    out = _normalise(df.copy())
    out["ticker"]        = ticker.upper()
    out["source"]        = source
    out["snapshot_date"] = snapshot_date or date.today().isoformat()

    conn = sqlite3.connect(db_path)
    try:
        out.to_sql("_tmp", conn, if_exists="replace", index=False)

        # Ensure main table exists
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} AS
            SELECT * FROM _tmp WHERE 1=0
        """)

        # Add unique constraint column set if not already present
        try:
            conn.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_snapshot
                ON {_TABLE}(ticker, snapshot_date, expiry, strike, option_type)
            """)
        except sqlite3.OperationalError:
            pass  # index may already exist

        cols   = ", ".join(f'"{c}"' for c in out.columns)
        n_before = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
        conn.execute(f"""
            INSERT OR IGNORE INTO {_TABLE} ({cols})
            SELECT {cols} FROM _tmp
        """)
        n_after  = conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
        conn.execute("DROP TABLE IF EXISTS _tmp")
        conn.commit()
        return n_after - n_before
    finally:
        conn.close()


def load_option_snapshots(
    ticker: str,
    db_path: str        = _DEFAULT_DB,
    snapshot_date: str  | None = None,
    start_date:   str   | None = None,
    end_date:     str   | None = None,
) -> pd.DataFrame:
    """Load option snapshots from the SQLite store.

    Parameters
    ----------
    ticker        : underlying symbol (case-insensitive)
    db_path       : path to the SQLite file
    snapshot_date : exact fetch date (YYYY-MM-DD); returns that day only
    start_date    : earliest snapshot_date to include (YYYY-MM-DD)
    end_date      : latest  snapshot_date to include (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame  (empty if no data found)
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    clauses = [f"ticker = '{ticker.upper()}'"]
    if snapshot_date:
        clauses.append(f"snapshot_date = '{snapshot_date}'")
    if start_date:
        clauses.append(f"snapshot_date >= '{start_date}'")
    if end_date:
        clauses.append(f"snapshot_date <= '{end_date}'")

    where = " AND ".join(clauses)
    conn  = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {_TABLE} WHERE {where}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


_SKEW_TABLE = "skew_metrics"


def save_skew_metrics(
    df: pd.DataFrame,
    ticker: str,
    db_path: str = _DEFAULT_DB,
    calc_date: str | None = None,
) -> int:
    """Persist the per-expiry skew table (sigma_atm, RR25, BF25 …).

    Rows are deduplicated on (ticker, calc_date, expiry) so re-running the
    notebook for the same snapshot date is safe — existing rows are silently
    skipped.

    Parameters
    ----------
    df        : DataFrame containing at least expiry + any skew columns
    ticker    : underlying symbol
    db_path   : path to the SQLite file (same DB as option snapshots)
    calc_date : date string (YYYY-MM-DD) for this set of metrics; defaults to
                today.  Pass the snapshot_date so historical metrics are stored
                under the date the data was captured, not the date of analysis.

    Returns
    -------
    int : number of new rows actually inserted
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["ticker"]    = ticker.upper()
    out["calc_date"] = calc_date or date.today().isoformat()

    if "expiry" in out.columns:
        out["expiry"] = pd.to_datetime(out["expiry"]).dt.date.astype(str)

    conn = sqlite3.connect(db_path)
    try:
        out.to_sql("_tmp_skew", conn, if_exists="replace", index=False)

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_SKEW_TABLE} AS
            SELECT * FROM _tmp_skew WHERE 1=0
        """)
        try:
            conn.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_skew
                ON {_SKEW_TABLE}(ticker, calc_date, expiry)
            """)
        except sqlite3.OperationalError:
            pass

        cols     = ", ".join(f'"{c}"' for c in out.columns)
        n_before = conn.execute(f"SELECT COUNT(*) FROM {_SKEW_TABLE}").fetchone()[0]
        conn.execute(f"""
            INSERT OR IGNORE INTO {_SKEW_TABLE} ({cols})
            SELECT {cols} FROM _tmp_skew
        """)
        n_after = conn.execute(f"SELECT COUNT(*) FROM {_SKEW_TABLE}").fetchone()[0]
        conn.execute("DROP TABLE IF EXISTS _tmp_skew")
        conn.commit()
        return n_after - n_before
    finally:
        conn.close()


def load_skew_metrics(
    ticker: str,
    db_path: str = _DEFAULT_DB,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """Load historical skew metrics for a ticker.

    Parameters
    ----------
    ticker     : underlying symbol (case-insensitive)
    db_path    : path to the SQLite file
    start_date : earliest calc_date to include (YYYY-MM-DD)
    end_date   : latest  calc_date to include (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame  (empty if no data found)
    """
    if not Path(db_path).exists():
        return pd.DataFrame()

    clauses = [f"ticker = '{ticker.upper()}'"]
    if start_date:
        clauses.append(f"calc_date >= '{start_date}'")
    if end_date:
        clauses.append(f"calc_date <= '{end_date}'")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f"SELECT * FROM {_SKEW_TABLE} WHERE {' AND '.join(clauses)}", conn
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def list_snapshots(db_path: str = _DEFAULT_DB) -> pd.DataFrame:
    """Return a summary of all snapshots stored in the database.

    Returns
    -------
    pd.DataFrame with columns: ticker, source, snapshot_date, row_count
    """
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["ticker", "source", "snapshot_date", "row_count"])

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f"""
            SELECT ticker, source, snapshot_date, COUNT(*) AS row_count
            FROM   {_TABLE}
            GROUP  BY ticker, source, snapshot_date
            ORDER  BY ticker, snapshot_date DESC
            """,
            conn,
        )
    except Exception:
        df = pd.DataFrame(columns=["ticker", "source", "snapshot_date", "row_count"])
    finally:
        conn.close()
    return df
