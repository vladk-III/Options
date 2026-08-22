"""
Headless daily options data collector -- runs in GitHub Actions on a
schedule (see .github/workflows/collect.yml), no notebook needed.

Two prediction targets, by ticker group:
  - DAILY_TICKERS (the 11 names with Mon/Wed/Fri or full daily listed
    expirations): target = NEXT TRADING DAY's price. For these, the
    "aligned" expiration fetched is whichever one actually expires on
    that next trading day (or the nearest one after it).
  - Everyone else: target = this week's Friday close, as before. The
    "weekly" expiration IS the aligned one for this group.

For every ticker, three roles are fetched where available -- "aligned"
(daily-tickers only), "weekly" (nearest Friday), "monthly" (~30 DTE) --
deduped by actual date (if two roles land on the same expiration, that's
one fetch, tagged with both role names). Each fetched expiration gets:
  - a FROZEN "initial_length_bucket" label (0-1/2-4/5-9/10-19/20-45/45+
    calendar days), assigned the first time that exact (ticker,
    expiration_date) pair is ever seen and never recomputed after -- by
    checking whether strike_level_data.csv already has a row for that
    pair, so the label doesn't drift as the same contract approaches
    expiry across multiple collection days.
  - live (recomputed daily) trading_days_to_expiration and
    trading_days_to_target numbers.
  - near-strike detail (2 strikes above/below spot, both calls and puts,
    with self-computed Black-Scholes Greeks) in strike_level_data.csv.

Three files, all upserted in place (re-running the same day refreshes
existing rows rather than duplicating):
  - options_data.csv        -- weekly aggregate summary (unchanged shape
                                from before), sourced from the "weekly"
                                role expiration specifically.
  - strike_level_data.csv   -- per-contract detail, see above -- this is
                                also where the frozen initial-length-bucket
                                labels effectively live (no separate file).
  - daily_data.csv          -- simple (ticker, date) -> price ledger,
                                DAILY_TICKERS only -- the target series
                                for next-day prediction.

No credentials or secrets needed -- just this script, a repo with
Actions enabled, and "Read and write permissions" turned on (see
SETUP.md).
"""
import os
import sys
import time
import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------
# CONFIG (kept in sync with dashboard.py's Colab app, where applicable)
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

# Names with same-week (daily or Mon/Wed/Fri) listed expirations, as of
# early 2026 -- predicted target for these is NEXT TRADING DAY's price,
# not end-of-week. Update this set if the exchange adds/removes names.
DAILY_TICKERS = {"SPY", "QQQ", "IWM", "AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA", "IBIT"}

REQUEST_DELAY_SECONDS = 0.6  # bumped from 0.4 -- more requests/run now, more spacing to be polite to Yahoo
DEFAULT_RISK_FREE_RATE = 0.045
MIN_T_YEARS = 6.5 / (24 * 365)  # ~6.5 trading hours, floor for a same-day target expiration (e.g. Friday-target on a Friday)

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

STRIKE_OFFSETS = [-2, -1, 0, 1, 2]  # 0 = closest strike to spot, then 2 below / 2 above
STRIKE_COLUMNS = [
    "ticker", "snapshot_date", "week_start", "week_position", "role", "expiration_date",
    "initial_length_bucket", "trading_days_to_expiration", "trading_days_to_target",
    "option_type", "strike_offset", "strike", "last_price", "bid", "ask", "volume",
    "open_interest", "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
]
STRIKE_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strike_level_data.csv")

DAILY_COLUMNS = ["ticker", "snapshot_date", "price"]
DAILY_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_data.csv")

MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE = 9, 30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE = 16, 0


# ---------------------------------------------------------
# TIME / CALENDAR HELPERS
# ---------------------------------------------------------
def now_et():
    return datetime.datetime.now(ZoneInfo("America/New_York"))


def today_et():
    return now_et().date()


def get_week_start(d=None):
    d = d or today_et()
    return d - datetime.timedelta(days=d.weekday())


def determine_snapshot_type(d=None):
    """Mon/Tue -> start, Wed -> intermediate, Thu/Fri/weekend -> end."""
    d = d or today_et()
    wd = d.weekday()
    if wd in (0, 1):
        return "start"
    if wd == 2:
        return "intermediate"
    return "end"


def next_trading_day(d):
    nd = d + datetime.timedelta(days=1)
    while nd.weekday() >= 5:
        nd += datetime.timedelta(days=1)
    return nd


def upcoming_friday(d):
    days_ahead = (4 - d.weekday()) % 7  # Friday = 4; if d IS Friday, returns d itself
    return d + datetime.timedelta(days=days_ahead)


def trading_days_between(start_date, end_date):
    """Number of weekday steps from start_date (exclusive) to end_date
    (inclusive). 0 if end_date <= start_date."""
    if end_date <= start_date:
        return 0
    count = 0
    d = start_date
    while d < end_date:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return count


def classify_length_bucket(calendar_days_to_expiration):
    d = max(calendar_days_to_expiration, 0)
    if d <= 1:
        return "0-1_day"
    if d <= 4:
        return "2-4_day"
    if d <= 9:
        return "5-9_day"
    if d <= 19:
        return "10-19_day"
    if d <= 45:
        return "20-45_day"
    return "45+_day"


# ---------------------------------------------------------
# WATCHDOG / TIME GATE
# ---------------------------------------------------------
def already_collected_today():
    if not os.path.exists(CSV_PATH):
        return False
    try:
        df = pd.read_csv(CSV_PATH, usecols=["snapshot_date"])
    except Exception:
        return False
    if df.empty:
        return False
    return (df["snapshot_date"] == today_et().isoformat()).any()


def should_run_now():
    """No more narrow time-window targeting -- real-world testing showed
    GitHub's scheduled-run delay can run well over an hour, which a tight
    window can't absorb. Instead: this fires many times a day (see
    collect.yml's every-30-min cron), and the FIRST invocation that lands
    during market hours on a given day does the actual collection --
    every later invocation that same day sees today's data already
    exists and just exits immediately (cheap, no API calls). If nothing
    has collected by market close, the run fails on purpose so GitHub's
    failure-notification email becomes your alert."""
    if os.environ.get("FORCE_RUN") == "true":
        print("Manually triggered -- running regardless of time.")
        return True

    current = now_et()
    if current.weekday() >= 5:
        print(f"{current}: weekend -- skipping.")
        return False

    if already_collected_today():
        print(f"{current}: already collected today -- skipping.")
        return False

    market_open = current.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = current.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)

    if market_open <= current <= market_close:
        return True

    if current > market_close:
        print(f"ALERT: it's {current} (past market close) and nothing has been collected today. "
              f"Failing this run on purpose so GitHub emails you about it -- "
              f"trigger 'Run workflow' manually to collect today's data.")
        sys.exit(1)

    print(f"{current}: before market open -- skipping.")
    return False


# ---------------------------------------------------------
# PRICE / RATE FETCH
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


# ---------------------------------------------------------
# BLACK-SCHOLES GREEKS (self-computed -- no paid data source needed)
# ---------------------------------------------------------
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


def compute_T(expiration_date, today):
    days = (expiration_date - today).days
    if days <= 0:
        return MIN_T_YEARS  # same-day target (e.g. Friday-role on a Friday) -- approximate, not zero
    return days / 365.0


def extract_near_strikes(price, calls, puts, call_greeks, put_greeks):
    """Per-contract rows (call AND put) for the closest strike to spot,
    plus 2 below and 2 above it (up to 10 rows: 5 strike levels x 2 sides)."""
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


def fetch_chain_for_expiration(yf_symbol, expiration_str, spot_price, risk_free_rate, today):
    """Fetches ONE specific expiration's chain. Returns (aggregate_features
    dict, near_strike_rows list), or (None, []) if the fetch fails."""
    try:
        tk = yf.Ticker(yf_symbol)
        chain = tk.option_chain(expiration_str)
        calls, puts = chain.calls.copy(), chain.puts.copy()
    except Exception:
        return None, []

    for df in (calls, puts):
        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)

    expiration_date = datetime.date.fromisoformat(expiration_str)
    T = compute_T(expiration_date, today)

    call_greeks = black_scholes_greeks(spot_price, calls["strike"].to_numpy(), T, risk_free_rate,
                                        calls["impliedVolatility"].to_numpy(), "call") if len(calls) else None
    put_greeks = black_scholes_greeks(spot_price, puts["strike"].to_numpy(), T, risk_free_rate,
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

    near_strike_rows = extract_near_strikes(spot_price, calls, puts, call_greeks, put_greeks)
    return features, near_strike_rows


# ---------------------------------------------------------
# EXPIRATION SELECTION
# ---------------------------------------------------------
def find_expiration_matching_or_nearest_after(expirations, target_date, floor_date):
    """Exact match on target_date if listed; else nearest expiration >=
    floor_date, closest to target_date. None if nothing qualifies."""
    exact = [e for e in expirations if e == target_date]
    if exact:
        return target_date
    candidates = [e for e in expirations if e >= floor_date]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - target_date).days))


def nearest_expiration_to_dte(expirations, today, target_dte):
    if not expirations:
        return None
    return min(expirations, key=lambda e: abs((e - today).days - target_dte))


def get_target_expirations(ticker, expirations, today):
    """Returns {role_name: expiration_date} for this ticker -- 'aligned'
    only for DAILY_TICKERS, 'weekly' and 'monthly' for everyone."""
    if not expirations:
        return {}
    roles = {}

    if ticker in DAILY_TICKERS:
        target = next_trading_day(today)
        aligned = find_expiration_matching_or_nearest_after(expirations, target, floor_date=target)
        if aligned:
            roles["aligned"] = aligned

    friday_target = upcoming_friday(today)
    weekly = find_expiration_matching_or_nearest_after(expirations, friday_target, floor_date=today)
    if weekly:
        roles["weekly"] = weekly

    monthly = nearest_expiration_to_dte(expirations, today, target_dte=30)
    if monthly:
        roles["monthly"] = monthly

    return roles


# ---------------------------------------------------------
# FROZEN INITIAL-LENGTH-BUCKET LOOKUP
# ---------------------------------------------------------
# No separate registry file -- the frozen label is derived by checking
# whether strike_level_data.csv already has a row for this exact
# (ticker, expiration_date). If so, that row's initial_length_bucket is
# reused as-is (frozen); if this is the first time this expiration has
# ever been seen, it's computed fresh from today's calendar-days-to-expiry.
def get_or_freeze_bucket(strikes_df, ticker, expiration_date, today):
    mask = (strikes_df["ticker"] == ticker) & (strikes_df["expiration_date"] == expiration_date.isoformat())
    existing = strikes_df[mask]
    if not existing.empty:
        return existing.iloc[0]["initial_length_bucket"]
    calendar_dte = (expiration_date - today).days
    return classify_length_bucket(calendar_dte)


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
                     "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
                     "trading_days_to_expiration", "trading_days_to_target"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_all_strikes(df):
    df[STRIKE_COLUMNS].to_csv(STRIKE_CSV_PATH, index=False)


def load_existing_daily():
    if not os.path.exists(DAILY_CSV_PATH):
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = pd.read_csv(DAILY_CSV_PATH)
    if df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def save_all_daily(df):
    df[DAILY_COLUMNS].to_csv(DAILY_CSV_PATH, index=False)


# ---------------------------------------------------------
# PER-TICKER COLLECTION
# ---------------------------------------------------------
def process_ticker(ticker, risk_free_rate, weekly_df, strikes_df, daily_df):
    yf_symbol = TICKER_YF_SYMBOL.get(ticker, ticker)
    today = today_et()
    today_str = today.isoformat()

    price = fetch_current_price(yf_symbol)
    if price is None:
        return f"{ticker}: could not fetch price right now.", weekly_df, strikes_df, daily_df

    try:
        tk = yf.Ticker(yf_symbol)
        raw_expirations = [datetime.date.fromisoformat(e) for e in tk.options]
    except Exception:
        raw_expirations = []

    role_map = get_target_expirations(ticker, raw_expirations, today)
    date_to_roles = {}
    for role, exp_date in role_map.items():
        date_to_roles.setdefault(exp_date, []).append(role)

    week_start = get_week_start(today).isoformat()
    week_position = WEEK_POSITION_LABELS[determine_snapshot_type(today)]
    target_date = next_trading_day(today) if ticker in DAILY_TICKERS else upcoming_friday(today)

    weekly_features = None
    n_strike_rows_total = 0

    for exp_date, roles in date_to_roles.items():
        role_label = "+".join(sorted(roles))
        features, strike_rows = fetch_chain_for_expiration(yf_symbol, exp_date.isoformat(), price, risk_free_rate, today)
        if features is None:
            continue
        if "weekly" in roles:
            weekly_features = features

        bucket = get_or_freeze_bucket(strikes_df, ticker, exp_date, today)
        trading_days_to_exp = trading_days_between(today, exp_date)
        trading_days_to_tgt = trading_days_between(today, target_date)

        for sr in strike_rows:
            srow = {
                "ticker": ticker, "snapshot_date": today_str, "week_start": week_start,
                "week_position": week_position, "role": role_label, "expiration_date": exp_date.isoformat(),
                "initial_length_bucket": bucket, "trading_days_to_expiration": trading_days_to_exp,
                "trading_days_to_target": trading_days_to_tgt,
            }
            srow.update(sr)
            smask = ((strikes_df["ticker"] == ticker) & (strikes_df["snapshot_date"] == today_str) &
                      (strikes_df["expiration_date"] == exp_date.isoformat()) &
                      (strikes_df["option_type"] == sr["option_type"]) &
                      (strikes_df["strike_offset"] == sr["strike_offset"]))
            if smask.any():
                for k, v in srow.items():
                    strikes_df.loc[smask, k] = v
            else:
                strikes_df = pd.concat([strikes_df, pd.DataFrame([srow])], ignore_index=True)
        n_strike_rows_total += len(strike_rows)

    # weekly aggregate row (options_data.csv), sourced from the "weekly" role expiration
    snapshot_type = determine_snapshot_type(today)
    wrow = {"ticker": ticker, "week_start": week_start, "snapshot_type": snapshot_type,
            "snapshot_date": today_str, "price": price}
    wrow.update(weekly_features or {c: np.nan for c in FEATURE_COLS})
    wmask = (weekly_df["ticker"] == ticker) & (weekly_df["week_start"] == week_start) & (weekly_df["snapshot_type"] == snapshot_type)
    if wmask.any():
        for k, v in wrow.items():
            weekly_df.loc[wmask, k] = v
    else:
        weekly_df = pd.concat([weekly_df, pd.DataFrame([wrow])], ignore_index=True)

    # daily price ledger, DAILY_TICKERS only -- the target series for next-day prediction
    if ticker in DAILY_TICKERS:
        dmask = (daily_df["ticker"] == ticker) & (daily_df["snapshot_date"] == today_str)
        drow = {"ticker": ticker, "snapshot_date": today_str, "price": price}
        if dmask.any():
            for k, v in drow.items():
                daily_df.loc[dmask, k] = v
        else:
            daily_df = pd.concat([daily_df, pd.DataFrame([drow])], ignore_index=True)

    msg = (f"{ticker}: {week_position} weekly row (${price:.2f}); "
           f"{len(date_to_roles)} expiration(s) fetched, {n_strike_rows_total} near-strike rows.")
    return msg, weekly_df, strikes_df, daily_df


def main():
    if not should_run_now():
        return

    weekly_df = load_existing()
    strikes_df = load_existing_strikes()
    daily_df = load_existing_daily()

    risk_free_rate = fetch_risk_free_rate()
    for ticker in TICKERS:
        msg, weekly_df, strikes_df, daily_df = process_ticker(
            ticker, risk_free_rate, weekly_df, strikes_df, daily_df)
        print(msg)
        time.sleep(REQUEST_DELAY_SECONDS)

    save_all(weekly_df)
    save_all_strikes(strikes_df)
    save_all_daily(daily_df)
    print(f"\nDone. {len(weekly_df)} weekly rows, {len(strikes_df)} near-strike rows, "
          f"{len(daily_df)} daily price rows.")


if __name__ == "__main__":
    main()
