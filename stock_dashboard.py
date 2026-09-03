"""
Stock Screener Dashboard (Streamlit) — v2
==========================================

Changes from v1:
  - Can screen the full S&P 500 (fetched live from Wikipedia), not just
    a manual ticker list. Batched downloads + a ticker cap protect
    against the page hanging or Yahoo rate-limiting you.
  - CAGR is now shown for both 5-year and 10-year windows vs gold,
    not a single blended lookback.
  - Two-stage display: a FUNDAMENTAL screen (must-beat-gold filters)
    runs on the whole universe first; the (usually much smaller) set
    of stocks that pass then gets the full technical readout
    (RSI / CDC ActionZone / volume) for you to eyeball and decide —
    with the automatic combined ALERT column kept exactly as before.
  - Volume is now checked two ways: RVOL (vs its own 20-day average)
    AND a 1-year volume percentile (is today's volume unusually high
    vs the last ~252 trading days, not just the last 20).
  - New "Backtest" tab: pick one ticker, replay the same ALERT logic
    over history, and see what happened to price N days after each
    past signal (5 / 20 / 60 trading days forward).

REQUIREMENTS:
    pip install streamlit yfinance pandas numpy lxml

LOCAL TEST:
    streamlit run stock_dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Gold-Beating Stock Screener", layout="wide")

# ============================================================
# CONSTANTS
# ============================================================

RETURN_FREQ = "W"
CDC_SMOOTH, CDC_FAST, CDC_SLOW = 2, 12, 26
RSI_PERIOD = 14
BATCH_SIZE = 40          # tickers per yf.download call
VOL_PCT_WINDOW = 252     # ~1 trading year


# ============================================================
# SIDEBAR — universe + thresholds
# ============================================================

st.sidebar.header("⚙️ Universe")

universe_mode = st.sidebar.radio(
    "Ticker source", ["Custom list", "Full S&P 500 (capped)"], index=0
)

if universe_mode == "Custom list":
    tickers_input = st.sidebar.text_area(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSM",
    )
    RAW_TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
else:
    MAX_TICKERS = st.sidebar.slider(
        "Max tickers to screen (caps run time / avoids hangs)",
        min_value=25, max_value=500, value=100, step=25,
    )
    st.sidebar.caption(
        f"Screening {MAX_TICKERS} of 500 — full 500 can take several minutes "
        "and may hit Yahoo Finance rate limits."
    )
    RAW_TICKERS = None  # resolved after sp500 list is fetched

GOLD_PROXY = st.sidebar.text_input("Gold proxy ticker", value="GLD")
VIX_TICKER = "^VIX"
RISK_FREE_RATE = st.sidebar.number_input(
    "Risk-free rate (annualized)", value=0.05, step=0.005, format="%.3f"
)

st.sidebar.subheader("Fundamental filters (must-beat-gold)")
CAGR_WINDOW_FOR_FILTER = st.sidebar.selectbox(
    "Use which CAGR window to filter eligibility", ["5y", "10y"], index=0
)
CORRELATION_MAX = st.sidebar.slider("Max correlation vs gold", -1.0, 1.0, 0.2, 0.05)
SHARPE_MIN = st.sidebar.slider("Min Sharpe ratio", 0.0, 3.0, 1.0, 0.1)

st.sidebar.subheader("Technical / timing filters (for ALERT + your own read)")
RVOL_THRESHOLD = st.sidebar.slider("Min relative volume (RVOL, vs 20d avg)", 0.5, 5.0, 1.5, 0.1)
VOL_PCT_THRESHOLD = st.sidebar.slider("Min volume percentile (vs past 1y)", 0.0, 1.0, 0.8, 0.05)
RSI_BUY_THRESHOLD = st.sidebar.slider("RSI buy zone (below)", 10, 70, 40, 1)
VIX_FEAR_THRESHOLD = st.sidebar.slider("VIX fear threshold", 10, 50, 25, 1)

REFRESH = st.sidebar.button("🔄 Refresh data now")
if REFRESH:
    st.cache_data.clear()


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_tickers() -> list:
    """Pull the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(tickers)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_single(ticker: str, period: str = "10y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history_batch(tickers: tuple, period: str = "10y") -> dict:
    """
    Download many tickers in batches (yf.download handles multiple
    tickers in one HTTP round-trip, far faster than one call per
    ticker). Returns {ticker: DataFrame with Close/Volume}.
    """
    results = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        try:
            data = yf.download(
                chunk, period=period, auto_adjust=True,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception:
            continue
        for t in chunk:
            try:
                if len(chunk) == 1:
                    sub = data
                else:
                    sub = data[t]
                sub = sub.dropna(how="all")
                if not sub.empty and "Close" in sub and sub["Close"].notna().sum() > 50:
                    results[t] = sub
            except Exception:
                continue
    return results


# ============================================================
# METRICS
# ============================================================

def compute_cagr(prices: pd.Series, years: float = None) -> float:
    prices = prices.dropna()
    if len(prices) < 2:
        return np.nan
    if years is not None:
        cutoff = prices.index[-1] - pd.Timedelta(days=int(years * 365.25))
        prices = prices[prices.index >= cutoff]
        if len(prices) < 2:
            return np.nan
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1


def compute_correlation(stock_prices, benchmark_prices, freq=RETURN_FREQ, years=2) -> float:
    cutoff = stock_prices.index[-1] - pd.Timedelta(days=int(years * 365.25))
    s = stock_prices[stock_prices.index >= cutoff]
    b = benchmark_prices[benchmark_prices.index >= cutoff]
    stock_ret = s.resample(freq).last().pct_change().dropna()
    bench_ret = b.resample(freq).last().pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def compute_sharpe(prices: pd.Series, risk_free_rate: float, years=2) -> float:
    cutoff = prices.index[-1] - pd.Timedelta(days=int(years * 365.25))
    p = prices[prices.index >= cutoff]
    daily_ret = p.pct_change().dropna()
    if daily_ret.std() == 0 or daily_ret.empty:
        return np.nan
    excess_daily_rf = risk_free_rate / 252
    return (daily_ret.mean() - excess_daily_rf) / daily_ret.std() * np.sqrt(252)


def compute_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_cdc_action_zone(prices: pd.Series) -> pd.DataFrame:
    ap = prices.ewm(span=CDC_SMOOTH).mean()
    fast = ap.ewm(span=CDC_FAST).mean()
    slow = ap.ewm(span=CDC_SLOW).mean()
    bull = fast > slow
    above_fast = prices > fast
    zone = pd.Series(index=prices.index, dtype=object)
    zone[bull & above_fast] = "Green"
    zone[bull & ~above_fast] = "LightGreen"
    zone[~bull & ~above_fast] = "Red"
    zone[~bull & above_fast] = "LightRed"
    buy_trigger = (zone == "Green") & (zone.shift(1) != "Green")
    return pd.DataFrame({"fast_ema": fast, "slow_ema": slow, "zone": zone,
                          "buy_trigger": buy_trigger})


def compute_rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / volume.rolling(window).mean()


def compute_volume_percentile(volume: pd.Series, window: int = VOL_PCT_WINDOW) -> pd.Series:
    """Today's volume rank (0-1) within the trailing `window` days —
    answers 'is volume higher than usual over the past year', not
    just vs the last 20 days like RVOL does."""
    return volume.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                                         raw=False)


# ============================================================
# FUNDAMENTAL SCREEN (runs on full universe — cheap)
# ============================================================

def fundamental_screen_row(ticker: str, hist: pd.DataFrame, gold_prices: pd.Series) -> dict:
    close = hist["Close"].dropna()
    cagr_5y = compute_cagr(close, years=5)
    cagr_10y = compute_cagr(close, years=10)
    gold_cagr_5y = compute_cagr(gold_prices, years=5)
    gold_cagr_10y = compute_cagr(gold_prices, years=10)

    corr = compute_correlation(close, gold_prices)
    sharpe = compute_sharpe(close, RISK_FREE_RATE)

    filter_cagr = cagr_5y if CAGR_WINDOW_FOR_FILTER == "5y" else cagr_10y
    filter_gold_cagr = gold_cagr_5y if CAGR_WINDOW_FOR_FILTER == "5y" else gold_cagr_10y

    beats_gold = (not np.isnan(filter_cagr)) and (filter_cagr > filter_gold_cagr)
    corr_ok = (not np.isnan(corr)) and (corr < CORRELATION_MAX)
    sharpe_ok = (not np.isnan(sharpe)) and (sharpe > SHARPE_MIN)
    eligible = beats_gold and corr_ok and sharpe_ok

    return {
        "Ticker": ticker,
        "Eligible": eligible,
        "CAGR 5y": cagr_5y, "CAGR 10y": cagr_10y,
        "Gold CAGR 5y": gold_cagr_5y, "Gold CAGR 10y": gold_cagr_10y,
        "Beats Gold": beats_gold,
        "Corr vs Gold": corr, "Corr OK": corr_ok,
        "Sharpe": sharpe, "Sharpe OK": sharpe_ok,
    }


# ============================================================
# TECHNICAL READOUT (runs only on eligible tickers — small set)
# ============================================================

def technical_row(ticker: str, hist: pd.DataFrame, vix_level: float) -> dict:
    close, volume = hist["Close"].dropna(), hist["Volume"].dropna()
    rsi = compute_rsi(close).iloc[-1]
    cdc = compute_cdc_action_zone(close)
    current_zone = cdc["zone"].iloc[-1]
    buy_trigger_today = bool(cdc["buy_trigger"].iloc[-1])
    rvol = compute_rvol(volume).iloc[-1]
    vol_pct = compute_volume_percentile(volume).iloc[-1]

    rsi_buy_zone = rsi < RSI_BUY_THRESHOLD
    in_trend_rvol = rvol > RVOL_THRESHOLD
    in_trend_pct = (not np.isnan(vol_pct)) and (vol_pct > VOL_PCT_THRESHOLD)
    in_trend_volume = in_trend_rvol and in_trend_pct

    alert = buy_trigger_today and rsi_buy_zone and in_trend_volume

    return {
        "Ticker": ticker,
        "ALERT": "🔔 BUY" if alert else "",
        "RSI": rsi, "RSI Buy Zone": rsi_buy_zone,
        "CDC Zone": current_zone, "CDC Buy Today": buy_trigger_today,
        "RVOL": rvol, "Vol Percentile (1y)": vol_pct, "In Trend": in_trend_volume,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(hist: pd.DataFrame, horizons=(5, 20, 60)) -> pd.DataFrame:
    """Replay the ALERT logic historically on one ticker and measure
    forward returns after each past signal."""
    close, volume = hist["Close"].dropna(), hist["Volume"].dropna()
    rsi = compute_rsi(close)
    cdc = compute_cdc_action_zone(close)
    rvol = compute_rvol(volume)
    vol_pct = compute_volume_percentile(volume)

    signal = (
        cdc["buy_trigger"]
        & (rsi < RSI_BUY_THRESHOLD)
        & (rvol > RVOL_THRESHOLD)
        & (vol_pct > VOL_PCT_THRESHOLD)
    )
    signal_dates = close.index[signal.reindex(close.index, fill_value=False)]

    rows = []
    for d in signal_dates:
        pos = close.index.get_loc(d)
        row = {"Signal Date": d.date(), "Price at Signal": close.iloc[pos]}
        for h in horizons:
            if pos + h < len(close):
                fwd_ret = close.iloc[pos + h] / close.iloc[pos] - 1
                row[f"+{h}d Return"] = fwd_ret
            else:
                row[f"+{h}d Return"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================

st.title("🪙 Gold-Beating Stock Screener")
st.caption(
    "Stage 1 screens the whole universe on fundamentals only (must beat "
    "gold's CAGR, stay low-correlation, clear a Sharpe bar). Stage 2 shows "
    "full technical detail for the stocks that pass, so you make the final "
    "call — with an automatic ALERT flagged when RSI + CDC ActionZone + "
    "volume all line up too."
)

tab_screen, tab_backtest = st.tabs(["📊 Screener", "🔁 Backtest"])

with tab_screen:
    with st.spinner("Fetching gold benchmark and VIX..."):
        try:
            gold_prices = fetch_history_single(GOLD_PROXY, "10y")["Close"]
            vix_level = fetch_history_single(VIX_TICKER, "5d")["Close"].iloc[-1]
        except Exception as e:
            st.error(f"Failed to fetch benchmark data: {e}")
            st.stop()

    if universe_mode == "Full S&P 500 (capped)":
        with st.spinner("Fetching S&P 500 constituent list..."):
            try:
                all_sp500 = fetch_sp500_tickers()
            except Exception as e:
                st.error(f"Failed to fetch S&P 500 list: {e}")
                st.stop()
        RAW_TICKERS = all_sp500[:MAX_TICKERS]

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Gold CAGR ({CAGR_WINDOW_FOR_FILTER})",
                f"{compute_cagr(gold_prices, years=5 if CAGR_WINDOW_FOR_FILTER=='5y' else 10)*100:.1f}%")
    col2.metric("Current VIX", f"{vix_level:.1f}",
                "FEAR ZONE" if vix_level > VIX_FEAR_THRESHOLD else "normal")
    col3.metric("Universe size", f"{len(RAW_TICKERS)} tickers")

    st.divider()
    st.subheader("Stage 1 — Fundamental screen (full universe)")

    with st.spinner(f"Downloading price history for {len(RAW_TICKERS)} tickers "
                     f"(batched, cached 15 min)..."):
        hist_map = fetch_history_batch(tuple(RAW_TICKERS), "10y")

    missing = set(RAW_TICKERS) - set(hist_map.keys())
    if missing:
        st.caption(f"Skipped {len(missing)} ticker(s) with insufficient data: "
                   + ", ".join(list(missing)[:20]) + ("..." if len(missing) > 20 else ""))

    fund_rows = []
    progress = st.progress(0.0, text="Screening fundamentals...")
    tickers_with_data = list(hist_map.keys())
    for i, t in enumerate(tickers_with_data):
        try:
            fund_rows.append(fundamental_screen_row(t, hist_map[t], gold_prices))
        except Exception:
            pass
        progress.progress((i + 1) / max(len(tickers_with_data), 1))
    progress.empty()

    if not fund_rows:
        st.info("No results.")
        st.stop()

    fund_df = pd.DataFrame(fund_rows).sort_values("Eligible", ascending=False)
    eligible_tickers = fund_df[fund_df["Eligible"]]["Ticker"].tolist()

    st.write(f"**{len(eligible_tickers)} of {len(fund_df)}** tickers passed the "
             f"fundamental filter (beats gold {CAGR_WINDOW_FOR_FILTER} CAGR, "
             f"correlation < {CORRELATION_MAX}, Sharpe > {SHARPE_MIN}).")

    st.dataframe(
        fund_df.style.format({
            "CAGR 5y": "{:.1%}", "CAGR 10y": "{:.1%}",
            "Gold CAGR 5y": "{:.1%}", "Gold CAGR 10y": "{:.1%}",
            "Corr vs Gold": "{:.2f}", "Sharpe": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Stage 2 — Technical detail (eligible tickers only)")

    if not eligible_tickers:
        st.info("No tickers passed the fundamental filter — nothing to show "
                 "technicals for. Try loosening the thresholds in the sidebar.")
    else:
        tech_rows = [technical_row(t, hist_map[t], vix_level) for t in eligible_tickers]
        tech_df = pd.DataFrame(tech_rows).sort_values("ALERT", ascending=False)

        alerts = tech_df[tech_df["ALERT"] != ""]
        if not alerts.empty:
            st.success(f"🔔 {len(alerts)} buy alert(s) today: "
                       + ", ".join(alerts["Ticker"].tolist()))
        else:
            st.info("No combined buy alerts right now — RSI, CDC zone and "
                    "volume all need to align at once, on top of already "
                    "passing fundamentals.")

        st.dataframe(
            tech_df.style.format({
                "RSI": "{:.1f}", "RVOL": "{:.2f}", "Vol Percentile (1y)": "{:.0%}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Data via Yahoo Finance (typically 15-20 min delayed). CDC ActionZone is "
        "a best-effort reimplementation — verify against TradingView. This is a "
        "decision-support tool, not investment advice."
    )

with tab_backtest:
    st.subheader("Backtest the ALERT logic on one ticker")
    bt_ticker = st.text_input("Ticker to backtest", value="NVDA").strip().upper()
    bt_period = st.selectbox("History length", ["5y", "10y", "max"], index=1)
    run_bt = st.button("Run backtest")

    if run_bt and bt_ticker:
        with st.spinner(f"Fetching history and running backtest for {bt_ticker}..."):
            try:
                bt_hist = fetch_history_single(bt_ticker, bt_period)
                bt_results = run_backtest(bt_hist)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                bt_results = None

        if bt_results is not None:
            if bt_results.empty:
                st.info("No historical signals fired for this ticker with the "
                        "current thresholds — try loosening RSI / RVOL / volume "
                        "percentile in the sidebar.")
            else:
                st.write(f"**{len(bt_results)} historical signal(s)** found.")
                st.dataframe(
                    bt_results.style.format({
                        "Price at Signal": "{:.2f}",
                        "+5d Return": "{:.1%}", "+20d Return": "{:.1%}", "+60d Return": "{:.1%}",
                    }),
                    use_container_width=True, hide_index=True,
                )

                st.markdown("**Summary**")
                summary_cols = st.columns(3)
                for i, h in enumerate((5, 20, 60)):
                    col_name = f"+{h}d Return"
                    valid = bt_results[col_name].dropna()
                    if len(valid) > 0:
                        win_rate = (valid > 0).mean()
                        avg_ret = valid.mean()
                        summary_cols[i].metric(
                            f"{h}-day forward",
                            f"{avg_ret:.1%} avg",
                            f"{win_rate:.0%} win rate ({len(valid)} signals)",
                        )
                    else:
                        summary_cols[i].metric(f"{h}-day forward", "n/a")

        st.caption(
            "Backtest replays today's exact ALERT rule (CDC just turned green + "
            "RSI below threshold + RVOL and 1y volume percentile both above "
            "threshold) at every point in history. Past signals do not "
            "guarantee future ones will behave the same way — this is a "
            "sanity check on the rule, not a promise of future returns."
        )
