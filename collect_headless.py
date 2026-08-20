"""
Headless daily options data collector -- runs in GitHub Actions on a
schedule (see .github/workflows/collect.yml), no notebook needed.

Each run:
  1. Checks the real US/Eastern clock and exits immediately if it isn't
     close to 10:30am ET (handles the two DST-offset cron triggers without
     double-collecting -- see the workflow file for why there are two).
  2. Loads whatever's already in the Google Sheet.
  3. Collects a fresh snapshot (price + options features + Greeks) for
     every tracked ticker, upserting into start/intermediate/end for the
     current week exactly like the Colab dashboard does.
  4. Writes the updated data back to the Google Sheet AND saves it as
     options_data.csv in this repo, which the workflow then commits --
     so you can pull the CSV directly from GitHub whenever you want,
     independent of the Sheet.

Required GitHub Actions secrets (see SETUP.md):
  GCP_SA_KEY  -- the Google service account's JSON key, as a single secret
  SHEET_ID    -- the target Google Sheet's ID (from its URL)
"""
import os
import sys
import time
import json
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
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
SHEET_WORKSHEET_NAME = "options_data"
SHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

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
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        print(f"{now_et}: weekend -- skipping.")
        return False
    target = now_et.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
    if abs((now_et - target).total_seconds()) > WINDOW_MINUTES * 60:
        print(f"{now_et}: not within {WINDOW_MINUTES} min of {TARGET_HOUR}:{TARGET_MINUTE:02d} ET -- skipping.")
        return False
    return True


# ---------------------------------------------------------
# WEEK / SNAPSHOT LABELING (same rules as the Colab dashboard)
# ---------------------------------------------------------
def get_week_start(d=None):
    d = d or datetime.date.today()
    return d - datetime.timedelta(days=d.weekday())


def determine_snapshot_type(today=None):
    d = today or datetime.date.today()
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


def fetch_option_snapshot(yf_symbol, risk_free_rate):
    price = fetch_current_price(yf_symbol)
    if price is None:
        return None, None
    try:
        tk = yf.Ticker(yf_symbol)
        expirations = tk.options
        if not expirations:
            return price, None
        expiration = expirations[0]
        chain = tk.option_chain(expiration)
        calls, puts = chain.calls.copy(), chain.puts.copy()
    except Exception:
        return price, None

    for df in (calls, puts):
        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)

    T = max((datetime.date.fromisoformat(expiration) - datetime.date.today()).days, 0) / 365.0

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
    return price, features


# ---------------------------------------------------------
# GOOGLE SHEETS I/O
# ---------------------------------------------------------
def get_worksheet():
    creds = Credentials.from_service_account_file("service_account.json", scopes=SHEET_SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ["SHEET_ID"])
    try:
        ws = sh.worksheet(SHEET_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_WORKSHEET_NAME, rows=2000, cols=len(OPTIONS_COLUMNS) + 2)
        ws.update([OPTIONS_COLUMNS])
    return ws


def load_existing(ws):
    values = ws.get_all_values()
    if len(values) < 2:
        return pd.DataFrame(columns=OPTIONS_COLUMNS)
    header, data_rows = values[0], values[1:]
    df = pd.DataFrame(data_rows, columns=header)
    for c in ["price"] + FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_all(ws, df):
    df_out = df[OPTIONS_COLUMNS].copy()
    values = [OPTIONS_COLUMNS] + df_out.astype(object).where(pd.notna(df_out), "").values.tolist()
    ws.clear()
    ws.update(values)


# ---------------------------------------------------------
# COLLECTION (upsert, same rules as the Colab dashboard)
# ---------------------------------------------------------
def upsert_snapshot(df, ticker, risk_free_rate):
    yf_symbol = TICKER_YF_SYMBOL.get(ticker, ticker)
    snapshot_type = determine_snapshot_type()
    price, features = fetch_option_snapshot(yf_symbol, risk_free_rate)
    if price is None:
        print(f"{ticker}: could not fetch data right now.")
        return df

    week_start = get_week_start().isoformat()
    today_str = datetime.date.today().isoformat()
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

    print(f"{ticker}: {action} {WEEK_POSITION_LABELS[snapshot_type]} snapshot (${price:.2f}).")
    return df


def main():
    if not should_run_now():
        return

    ws = get_worksheet()
    df = load_existing(ws)
    if df.empty:
        df = pd.DataFrame(columns=OPTIONS_COLUMNS)

    risk_free_rate = fetch_risk_free_rate()
    for ticker in TICKERS:
        df = upsert_snapshot(df, ticker, risk_free_rate)
        time.sleep(REQUEST_DELAY_SECONDS)

    save_all(ws, df)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nDone. {len(df)} total rows. Sheet and {CSV_PATH} updated.")


if __name__ == "__main__":
    main()
