"""
Stock Screener Dashboard (Streamlit)
=====================================

Web version of stock_screener.py — deploy this once (free, via Streamlit
Community Cloud) and you get a URL you can open from Safari on your
iPhone any time, no need to keep your own computer running.

LOCAL TEST:
    pip install streamlit yfinance pandas numpy
    streamlit run stock_dashboard.py

DEPLOY (so it's reachable from your iPhone) — see deployment steps
in the chat message alongside this file.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Gold-Beating Stock Screener", layout="wide")

# ============================================================
# SIDEBAR — editable thresholds (mirrors CONFIG in the CLI version)
# ============================================================

st.sidebar.header("⚙️ Screening Criteria")

tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSM",
)
TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

GOLD_PROXY = st.sidebar.text_input("Gold proxy ticker", value="GLD")
VIX_TICKER = "^VIX"
LOOKBACK_PERIOD = st.sidebar.selectbox(
    "Lookback period", ["1y", "2y", "3y", "5y"], index=1
)
RISK_FREE_RATE = st.sidebar.number_input(
    "Risk-free rate (annualized)", value=0.05, step=0.005, format="%.3f"
)

st.sidebar.subheader("Fundamental filters")
CORRELATION_MAX = st.sidebar.slider("Max correlation vs gold", -1.0, 1.0, 0.2, 0.05)
SHARPE_MIN = st.sidebar.slider("Min Sharpe ratio", 0.0, 3.0, 1.0, 0.1)
CAGR_MUST_BEAT_GOLD = st.sidebar.checkbox("Must beat gold CAGR", value=True)

st.sidebar.subheader("Technical / timing filters")
RVOL_THRESHOLD = st.sidebar.slider("Min relative volume (RVOL)", 0.5, 5.0, 1.5, 0.1)
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = st.sidebar.slider("RSI buy zone (below)", 10, 70, 40, 1)
VIX_FEAR_THRESHOLD = st.sidebar.slider("VIX fear threshold", 10, 50, 25, 1)

CDC_SMOOTH, CDC_FAST, CDC_SLOW = 2, 12, 26
RETURN_FREQ = "W"

REFRESH = st.sidebar.button("🔄 Refresh data now")


# ============================================================
# CORE LOGIC (same math as the CLI version)
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)  # cache 15 min so repeat opens are fast
def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return df


def compute_cagr(prices: pd.Series) -> float:
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    if n_years <= 0:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1


def compute_correlation(stock_prices, benchmark_prices, freq=RETURN_FREQ) -> float:
    stock_ret = stock_prices.resample(freq).last().pct_change().dropna()
    bench_ret = benchmark_prices.resample(freq).last().pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def compute_sharpe(prices: pd.Series, risk_free_rate: float) -> float:
    daily_ret = prices.pct_change().dropna()
    if daily_ret.std() == 0 or daily_ret.empty:
        return np.nan
    excess_daily_rf = risk_free_rate / 252
    return (daily_ret.mean() - excess_daily_rf) / daily_ret.std() * np.sqrt(252)


def compute_annualized_vol(prices: pd.Series) -> float:
    return prices.pct_change().dropna().std() * np.sqrt(252)


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


def screen_ticker(ticker: str, gold_prices: pd.Series, vix_level: float) -> dict:
    hist = fetch_history(ticker, LOOKBACK_PERIOD)
    close, volume = hist["Close"], hist["Volume"]

    cagr = compute_cagr(close)
    gold_cagr = compute_cagr(gold_prices)
    corr = compute_correlation(close, gold_prices)
    sharpe = compute_sharpe(close, RISK_FREE_RATE)
    vol_annual = compute_annualized_vol(close)

    rsi = compute_rsi(close).iloc[-1]
    cdc = compute_cdc_action_zone(close)
    current_zone = cdc["zone"].iloc[-1]
    buy_trigger_today = bool(cdc["buy_trigger"].iloc[-1])
    rvol = compute_rvol(volume).iloc[-1]

    passes_cagr = (cagr > gold_cagr) if CAGR_MUST_BEAT_GOLD else True
    passes_corr = (not np.isnan(corr)) and (corr < CORRELATION_MAX)
    passes_sharpe = (not np.isnan(sharpe)) and (sharpe > SHARPE_MIN)
    fundamentally_eligible = passes_cagr and passes_corr and passes_sharpe

    in_trend_volume = rvol > RVOL_THRESHOLD
    rsi_buy_zone = rsi < RSI_BUY_THRESHOLD

    alert = fundamentally_eligible and buy_trigger_today and rsi_buy_zone and in_trend_volume

    return {
        "Ticker": ticker, "ALERT": "🔔 BUY" if alert else "",
        "Eligible": "✅" if fundamentally_eligible else "❌",
        "CAGR": cagr, "Gold CAGR": gold_cagr, "Beats Gold": passes_cagr,
        "Corr vs Gold": corr, "Corr OK": passes_corr,
        "Sharpe": sharpe, "Sharpe OK": passes_sharpe,
        "Ann. Vol": vol_annual,
        "RSI": rsi, "RSI Buy Zone": rsi_buy_zone,
        "CDC Zone": current_zone, "CDC Buy Today": buy_trigger_today,
        "RVOL": rvol, "In Trend": in_trend_volume,
    }


# ============================================================
# UI
# ============================================================

st.title("🪙 Gold-Beating Stock Screener")
st.caption(
    "Screens stocks against your thesis: must beat gold's CAGR, "
    "stay low-correlation with gold, clear a Sharpe bar, and only "
    "alert when RSI + CDC ActionZone + volume all line up."
)

if REFRESH:
    st.cache_data.clear()

with st.spinner("Fetching gold benchmark and VIX..."):
    try:
        gold_prices = fetch_history(GOLD_PROXY, LOOKBACK_PERIOD)["Close"]
        vix_level = fetch_history(VIX_TICKER, "5d")["Close"].iloc[-1]
    except Exception as e:
        st.error(f"Failed to fetch benchmark data: {e}")
        st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Gold CAGR (lookback)", f"{compute_cagr(gold_prices)*100:.1f}%")
col2.metric("Current VIX", f"{vix_level:.1f}",
            "FEAR ZONE" if vix_level > VIX_FEAR_THRESHOLD else "normal")
col3.metric("Last updated", datetime.now().strftime("%Y-%m-%d %H:%M"))

st.divider()

rows = []
errors = []
progress = st.progress(0.0, text="Screening tickers...")
for i, t in enumerate(TICKERS):
    try:
        rows.append(screen_ticker(t, gold_prices, vix_level))
    except Exception as e:
        errors.append(f"{t}: {e}")
    progress.progress((i + 1) / len(TICKERS), text=f"Screening {t}...")
progress.empty()

if errors:
    st.warning("Some tickers failed:\n" + "\n".join(errors))

if not rows:
    st.info("No results to show.")
    st.stop()

df = pd.DataFrame(rows).sort_values("ALERT", ascending=False)

alerts = df[df["ALERT"] != ""]
if not alerts.empty:
    st.success(f"🔔 {len(alerts)} buy alert(s) today: "
               + ", ".join(alerts["Ticker"].tolist()))
else:
    st.info("No combined buy alerts right now — fundamentals, RSI, CDC zone "
            "and volume all need to align at once.")

st.dataframe(
    df.style.format({
        "CAGR": "{:.1%}", "Gold CAGR": "{:.1%}", "Corr vs Gold": "{:.2f}",
        "Sharpe": "{:.2f}", "Ann. Vol": "{:.1%}", "RSI": "{:.1f}", "RVOL": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Data via Yahoo Finance (typically 15-20 min delayed). CDC ActionZone is "
    "a best-effort reimplementation — verify against TradingView. This is a "
    "decision-support tool, not investment advice."
)
