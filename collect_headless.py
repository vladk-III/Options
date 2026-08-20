"""
Headless daily options data collector -- runs in GitHub Actions on a
schedule (see .github/workflows/collect.yml), no notebook needed.

Each run:
  1. Checks the real US/Eastern clock and exits immediately if it isn't
     close to 10:30am ET (handles the two DST-offset cron triggers without
     double-collecting -- see the workflow file for why there are two).
  2. Loads whatever's already in options_data.csv (checked out from the repo).
  3. Collects a fresh snapshot (price + options features + Greeks) for
     every tracked ticker, upserting into start/intermediate/end for the
     current week -- same rules as the Colab dashboard.
  4. Saves the updated options_data.csv, which the workflow then commits
     back to the repo.

No credentials or secrets needed at all -- just this script and a repo
with Actions enabled (and "Read and write permissions" turned on so the
workflow can push its own commits -- see SETUP.md).
"""
import os
import time
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------
# CONFIG (kept in sync with dashboard.py's Colab app)
# ---------------------------------------------------------
TICKER_YF_SYMBOL = {
    "SPY": "SPY", "QQQ": "QQQ", "SPX": "^GSPC", "TSLA": "TSLA", "NVDA": "NVDA",
    "IBIT": "IBIT", "AAPL": "AAPL", "AMZN": "AMZN", "TLT": "TLT", "INTC": "INTC",
    "IWM": "IWM", "MU": "MU", "MSTR": "MSTR", "MSFT": "MSFT", "GLD": "GLD",
    "VIX": "^VIX", "ETHA": "ETHA", "META": "META", "SLV": "SLV", "NFLX": "NFLX",
    "PFE": "PFE", "SOFI": "SOFI", "COIN": "COIN", "GOOGL": "GOOGL", "PLTR": "PLTR",
    "SHOP": "SHOP", "SQ": "SQ", "XOM": "XOM", "AMD": "AMD", "BAC": "BAC",
}
TICKERS = list(TICKER_YF_SYMBOL.keys())
REQUEST_DELAY_SECONDS = 0.4
DEFAULT_RISK_FREE_RATE = 0.045

FEATURE_COLS = [
    "avg_call_iv", "avg_put_iv", "iv_skew", "call_volume", "put_volume",
    "put_call_volume_ratio", "call_open_interest", "put_open_interest",
    "put_call_oi_ratio", "num_contracts",
    "avg_call_delta", "avg_put_delta", "avg_gamma",
    "avg_call_theta", "avg_put_theta", "avg_vega",
    "avg_call_rho", "avg_put_rho",
]
OPTIONS_COLUMNS = ["ticker", "week_start", "snapshot_type", "snapshot_date", "price"] + FEATURE_COLS
WEEK_POSITION_LABELS = {"start": "Start of Week", "intermediate": "Midweek", "end": "End of Week"}

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "options_data.csv")

# How close to 10:30am ET we need to be for the run to actually proceed
# (the workflow fires at both possible UTC offsets to survive DST, and
# this window keeps only the one that's actually ~10:30 local from doing
# anything).
TARGET_HOUR, TARGET_MINUTE = 10, 30
WINDOW_MINUTES = 20


# ---------------------------------------------------------
# TIME GATE
# ---------------------------------------------------------
def should_run_now():
    # A manual trigger (the "Run workflow" button) should always run,
    # regardless of what time it is -- the whole point of testing manually
    # is to see it work right now. Only the scheduled cron runs are
    # restricted to the ~10:30am ET window (see FORCE_RUN in the workflow).
    if os.environ.get("FORCE_RUN") == "true":
        print("Manually triggered -- running regardless of time.")
        return True

    current = now_et()
    if current.weekday() >= 5:
        print(f"{current}: weekend -- skipping.")
        return False
    target = current.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    if abs((current - target).total_seconds()) > WINDOW_MINUTES * 60:
        print(f"{current}: not within {WINDOW_MINUTES} min of {TARGET_HOUR}:{TARGET_MINUTE:02d} ET -- skipping.")
        return False
    return True


# ---------------------------------------------------------
# WEEK / SNAPSHOT LABELING (same rules as the Colab dashboard)
# ---------------------------------------------------------
def now_et():
    return datetime.datetime.now(ZoneInfo("America/New_York"))


def today_et():
    """The current date in US/Eastern time (handles EST/EDT automatically).
    GitHub Actions runners default to UTC, so using this instead of
    datetime.date.today() everywhere keeps snapshot_date, week_start, and
    the start/midweek/end labeling all anchored to the US market's actual
    calendar day, not the runner's."""
    return now_et().date()


def get_week_start(d=None):
    d = d or today_et()
    return d - datetime.timedelta(days=d.weekday())


def determine_snapshot_type(today=None):
    d = today or today_et()
    wd = d.weekday()
    if wd in (0, 1):
        return "start"
    if wd == 2:
        return "intermediate"
    return "end"


# ---------------------------------------------------------
# DATA FETCH (yfinance + self-computed Black-Scholes Greeks)
# ---------------------------------------------------------
def fetch_current_price(yf_symbol):
    try:
        hist = yf.Ticker(yf_symbol).history(period="5d")
    except Exception:
        return None
    if hist is None or hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


def fetch_risk_free_rate():
    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return DEFAULT_RISK_FREE_RATE


def black_scholes_greeks(S, K, T, r, sigma, option_type):
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    valid = (T > 0) & (sigma > 0) & np.isfinite(sigma) & (K > 0) & (S > 0)

    delta = np.full_like(K, np.nan, dtype=float)
    gamma = np.full_like(K, np.nan, dtype=float)
    theta = np.full_like(K, np.nan, dtype=float)
    vega = np.full_like(K, np.nan, dtype=float)
    rho = np.full_like(K, np.nan, dtype=float)

    if not np.any(valid):
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    Kv, sv = K[valid], sigma[valid]
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / Kv) + (r + 0.5 * sv ** 2) * T) / (sv * sqrtT)
    d2 = d1 - sv * sqrtT
    pdf_d1 = norm.pdf(d1)

    gamma[valid] = pdf_d1 / (S * sv * sqrtT)
    vega[valid] = S * pdf_d1 * sqrtT / 100.0

    if option_type == "call":
        delta[valid] = norm.cdf(d1)
        theta[valid] = (-S * pdf_d1 * sv / (2 * sqrtT) - r * Kv * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        rho[valid] = Kv * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        delta[valid] = norm.cdf(d1) - 1
        theta[valid] = (-S * pdf_d1 * sv / (2 * sqrtT) + r * Kv * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
        rho[valid] = -Kv * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


STRIKE_OFFSETS = [-2, -1, 0, 1, 2]  # 0 = closest strike to spot, then 2 below / 2 above
STRIKE_COLUMNS = [
    "ticker", "week_start", "snapshot_type", "snapshot_date", "option_type", "strike_offset",
    "strike", "last_price", "bid", "ask", "volume", "open_interest",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
]
STRIKE_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strike_level_data.csv")


def pick_expiration(expirations):
    """Skips same-day (0DTE) expirations when a later one is available --
    using a 0DTE expiration makes time-to-expiry 0 for the whole chain,
    which zeroes out every Greek (this is why SPY/QQQ/IWM -- all very
    liquid 0DTE names -- were showing blank Greeks before this fix)."""
    today = today_et()
    for exp in expirations:
        if datetime.date.fromisoformat(exp) > today:
            return exp
    return expirations[0] if expirations else None


def extract_near_strikes(price, calls, puts, call_greeks, put_greeks):
    """Returns per-contract rows (call AND put) for the strikes immediately
    around the current price: the closest strike, plus 2 below and 2 above
    it (5 strike levels x 2 sides = up to 10 rows)."""
    call_strikes = calls["strike"].to_numpy() if len(calls) else np.array([])
    put_strikes = puts["strike"].to_numpy() if len(puts) else np.array([])
    all_strikes = sorted(set(call_strikes.tolist()) | set(put_strikes.tolist()))
    if not all_strikes:
        return []

    closest_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - price))

    def contract_row(strikes_arr, side_df, greeks, strike):
        matches = np.where(np.isclose(strikes_arr, strike))[0]
        if len(matches) == 0:
            return None
        i = int(matches[0])
        r = side_df.iloc[i]
        return {
            "last_price": float(r.get("lastPrice", np.nan)),
            "bid": float(r.get("bid", np.nan)),
            "ask": float(r.get("ask", np.nan)),
            "volume": float(r.get("volume", np.nan)),
            "open_interest": float(r.get("openInterest", np.nan)),
            "implied_volatility": float(r.get("impliedVolatility", np.nan)),
            "delta": float(greeks["delta"][i]) if greeks is not None else np.nan,
            "gamma": float(greeks["gamma"][i]) if greeks is not None else np.nan,
            "theta": float(greeks["theta"][i]) if greeks is not None else np.nan,
            "vega": float(greeks["vega"][i]) if greeks is not None else np.nan,
            "rho": float(greeks["rho"][i]) if greeks is not None else np.nan,
        }

    rows = []
    for offset in STRIKE_OFFSETS:
        idx = closest_idx + offset
        if idx < 0 or idx >= len(all_strikes):
            continue
        strike = all_strikes[idx]
        for option_type, strikes_arr, side_df, greeks in [
            ("call", call_strikes, calls, call_greeks),
            ("put", put_strikes, puts, put_greeks),
        ]:
            data = contract_row(strikes_arr, side_df, greeks, strike)
            if data is None:
                continue
            row = {"option_type": option_type, "strike_offset": offset, "strike": strike}
            row.update(data)
            rows.append(row)
    return rows


def fetch_option_snapshot(yf_symbol, risk_free_rate):
    """Returns (price, aggregate_features_dict, near_strike_rows).
    aggregate_features_dict is the whole-chain summary (same shape as
    before); near_strike_rows is per-contract detail for the 5 strikes
    around spot (see extract_near_strikes)."""
    price = fetch_current_price(yf_symbol)
    if price is None:
        return None, None, []
    try:
        tk = yf.Ticker(yf_symbol)
        expirations = tk.options
        if not expirations:
            return price, None, []
        expiration = pick_expiration(expirations)
        chain = tk.option_chain(expiration)
        calls, puts = chain.calls.copy(), chain.puts.copy()
    except Exception:
        return price, None, []

    for df in (calls, puts):
        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)

    T = max((datetime.date.fromisoformat(expiration) - today_et()).days, 0) / 365.0

    call_greeks = black_scholes_greeks(price, calls["strike"].to_numpy(), T, risk_free_rate,
                                        calls["impliedVolatility"].to_numpy(), "call") if len(calls) else None
    put_greeks = black_scholes_greeks(price, puts["strike"].to_numpy(), T, risk_free_rate,
                                       puts["impliedVolatility"].to_numpy(), "put") if len(puts) else None

    def nanmean(arr):
        return float(np.nanmean(arr)) if arr is not None and len(arr) and not np.all(np.isnan(arr)) else np.nan

    call_iv = calls["impliedVolatility"].mean() if len(calls) else np.nan
    put_iv = puts["impliedVolatility"].mean() if len(puts) else np.nan
    call_vol = calls["volume"].sum() if len(calls) else 0
    put_vol = puts["volume"].sum() if len(puts) else 0
    call_oi = calls["openInterest"].sum() if len(calls) else 0
    put_oi = puts["openInterest"].sum() if len(puts) else 0

    gammas = np.concatenate([call_greeks["gamma"], put_greeks["gamma"]]) if call_greeks is not None and put_greeks is not None else None
    vegas = np.concatenate([call_greeks["vega"], put_greeks["vega"]]) if call_greeks is not None and put_greeks is not None else None

    features = {
        "avg_call_iv": call_iv, "avg_put_iv": put_iv,
        "iv_skew": (put_iv - call_iv) if pd.notna(call_iv) and pd.notna(put_iv) else np.nan,
        "call_volume": float(call_vol), "put_volume": float(put_vol),
        "put_call_volume_ratio": (put_vol / call_vol) if call_vol else np.nan,
        "call_open_interest": float(call_oi), "put_open_interest": float(put_oi),
        "put_call_oi_ratio": (put_oi / call_oi) if call_oi else np.nan,
        "num_contracts": len(calls) + len(puts),
        "avg_call_delta": nanmean(call_greeks["delta"]) if call_greeks is not None else np.nan,
        "avg_put_delta": nanmean(put_greeks["delta"]) if put_greeks is not None else np.nan,
        "avg_gamma": nanmean(gammas),
        "avg_call_theta": nanmean(call_greeks["theta"]) if call_greeks is not None else np.nan,
        "avg_put_theta": nanmean(put_greeks["theta"]) if put_greeks is not None else np.nan,
        "avg_vega": nanmean(vegas),
        "avg_call_rho": nanmean(call_greeks["rho"]) if call_greeks is not None else np.nan,
        "avg_put_rho": nanmean(put_greeks["rho"]) if put_greeks is not None else np.nan,
    }

    near_strike_rows = extract_near_strikes(price, calls, puts, call_greeks, put_greeks)
    return price, features, near_strike_rows


# ---------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------
def load_existing():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=OPTIONS_COLUMNS)
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        return pd.DataFrame(columns=OPTIONS_COLUMNS)
    for c in ["price"] + FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_all(df):
    df[OPTIONS_COLUMNS].to_csv(CSV_PATH, index=False)


def load_existing_strikes():
    if not os.path.exists(STRIKE_CSV_PATH):
        return pd.DataFrame(columns=STRIKE_COLUMNS)
    df = pd.read_csv(STRIKE_CSV_PATH)
    if df.empty:
        return pd.DataFrame(columns=STRIKE_COLUMNS)
    numeric_cols = ["strike", "last_price", "bid", "ask", "volume", "open_interest",
                     "implied_volatility", "delta", "gamma", "theta", "vega", "rho"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_all_strikes(df):
    df[STRIKE_COLUMNS].to_csv(STRIKE_CSV_PATH, index=False)


# ---------------------------------------------------------
# COLLECTION (upsert, same rules as the Colab dashboard)
# ---------------------------------------------------------
def upsert_snapshot(df, strikes_df, ticker, risk_free_rate):
    """Updates both the aggregate weekly row (df) and the near-strike
    detail rows (strikes_df) for this ticker's current snapshot."""
    yf_symbol = TICKER_YF_SYMBOL.get(ticker, ticker)
    snapshot_type = determine_snapshot_type()
    price, features, strike_rows = fetch_option_snapshot(yf_symbol, risk_free_rate)
    if price is None:
        print(f"{ticker}: could not fetch data right now.")
        return df, strikes_df

    week_start = get_week_start().isoformat()
    today_str = today_et().isoformat()

    row = {"ticker": ticker, "week_start": week_start, "snapshot_type": snapshot_type,
           "snapshot_date": today_str, "price": price}
    row.update(features or {c: np.nan for c in FEATURE_COLS})

    mask = (df["ticker"] == ticker) & (df["week_start"] == week_start) & (df["snapshot_type"] == snapshot_type)
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
        action = "refreshed"
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        action = "recorded"

    # near-strike detail rows: same upsert key plus option_type + strike_offset
    for sr in strike_rows:
        srow = {"ticker": ticker, "week_start": week_start, "snapshot_type": snapshot_type,
                "snapshot_date": today_str}
        srow.update(sr)
        smask = ((strikes_df["ticker"] == ticker) & (strikes_df["week_start"] == week_start) &
                  (strikes_df["snapshot_type"] == snapshot_type) &
                  (strikes_df["option_type"] == sr["option_type"]) &
                  (strikes_df["strike_offset"] == sr["strike_offset"]))
        if smask.any():
            for k, v in srow.items():
                strikes_df.loc[smask, k] = v
        else:
            strikes_df = pd.concat([strikes_df, pd.DataFrame([srow])], ignore_index=True)

    n_strikes = len(set((r["strike_offset"] for r in strike_rows)))
    print(f"{ticker}: {action} {WEEK_POSITION_LABELS[snapshot_type]} snapshot (${price:.2f}), "
          f"{len(strike_rows)} near-strike contract rows across {n_strikes} strike levels.")
    return df, strikes_df


def main():
    if not should_run_now():
        return

    df = load_existing()
    strikes_df = load_existing_strikes()
    risk_free_rate = fetch_risk_free_rate()
    for ticker in TICKERS:
        df, strikes_df = upsert_snapshot(df, strikes_df, ticker, risk_free_rate)
        time.sleep(REQUEST_DELAY_SECONDS)

    save_all(df)
    save_all_strikes(strikes_df)
    print(f"\nDone. {len(df)} weekly summary rows ({CSV_PATH}), "
          f"{len(strikes_df)} near-strike detail rows ({STRIKE_CSV_PATH}).")


if __name__ == "__main__":
    main()
