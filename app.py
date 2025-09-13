# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import math

# --- Safe import of norm (scipy optional) ---
try:
    from scipy.stats import norm
except Exception:
    class _NormFallback:
        @staticmethod
        def cdf(x):
            # normal CDF via error function
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        @staticmethod
        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    norm = _NormFallback()

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(page_title="Analyst Terminal — Valuation & Options", page_icon="💹", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #e6e6e6; }
    h1,h2,h3 { color: #39FF14; font-weight:700; }
    .block-container { padding: 1rem 2rem; }
    .metric-card { background: #111316; padding: 10px; border-radius: 8px; border: 1px solid #222; }
    .css-1d391kg { background-color: #0b0c0e !important; }
    .stDataFrame { background-color: #121416; }
    a { color: #7ef9a4; }
    </style>
    """, unsafe_allow_html=True
)

st.title("💹 Analyst Terminal — Equity Valuation & Options")
st.caption("Real-time modelling, DCF, comps, Black–Scholes options, and news. Built for IB/PE/AM prep. — Navjot Dhah")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Search & settings")
ticker = st.sidebar.text_input("Enter ticker (example: AAPL, WYNN, MSFT)", value="WYNN").upper().strip()
use_live = st.sidebar.checkbox("Use live yfinance data", value=True)
# DCF defaults
default_wacc = st.sidebar.number_input("Default WACC (%)", value=9.0, step=0.1)/100.0
default_tg = st.sidebar.number_input("Default Terminal growth (%)", value=2.5, step=0.1)/100.0
projection_years = st.sidebar.selectbox("Projection years", [5,7,10], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If yfinance misses fields, use manual overrides shown in each section.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=300)
def fetch_yf(t):
    """Fetch common yfinance payloads (safe)."""
    tk = yf.Ticker(t)
    try:
        info = tk.info
    except Exception:
        info = {}
    try:
        fin = tk.financials
    except Exception:
        fin = pd.DataFrame()
    try:
        bs = tk.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        cf = tk.cashflow
    except Exception:
        cf = pd.DataFrame()
    try:
        hist = tk.history(period="5y")
    except Exception:
        hist = pd.DataFrame()
    return info, fin, bs, cf, hist

def safe_number(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def style_numeric(df):
    if df is None or df.empty:
        return df
    df_t = df.T
    numeric_cols = df_t.select_dtypes(include=[np.number]).columns
    fmt = {c: "{:,.0f}" for c in numeric_cols}
    return df_t.style.format(fmt)

def find_row_value(df, keywords):
    """Find first row in df index containing any of keywords (case-insensitive)."""
    if df is None or df.empty:
        return None
    idx = df.index
    for k in keywords:
        for label in idx:
            if k.lower() in str(label).lower():
                try:
                    return df.loc[label].iloc[0]
                except Exception:
                    continue
    return None

def dcf_from_fcf(last_fcf, growth, discount, tg, years):
    """Simple DCF (explicit years + Gordon terminal)."""
    proj = [last_fcf * (1 + growth)**i for i in range(1, years+1)]
    pv = sum([proj[i] / ((1 + discount)**(i+1)) * (1+discount) for i in range(len(proj))])  # safer indexing
    # better to compute: pv = sum(proj[i]/(1+discount)^(i+1)) but we'll compute cleanly below
    pv = sum([proj[i] / ((1 + discount)**(i+1)) for i in range(len(proj))])  # adjust to correct discounting
    if discount <= tg:
        terminal = np.nan
    else:
        terminal_nom = proj[-1] * (1 + tg) / (discount - tg)
        terminal = terminal_nom / ((1 + discount)**years)
    enterprise = pv + (terminal if not np.isnan(terminal) else 0)
    return {
        "proj_nominal": proj,
        "proj_pv": [proj[i] / ((1 + discount)**(i+1)) for i in range(len(proj))],
        "terminal_pv": terminal,
        "enterprise_value": enterprise
    }

def black_scholes_price(S, K, T, r, sigma, option="call"):
    """Black-Scholes with fallback norm implemented."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S-K) if option=="call" else max(0.0, K-S)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    else:
        return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def get_yahoo_news(ticker, limit=6):
    """Small Yahoo search-based news fetch."""
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        resp = requests.get(url, timeout=6).json()
        items = resp.get("news", []) or resp.get("items", []) or []
        # limit and simplify
        out = []
        for it in items[:limit]:
            title = it.get("title") or it.get("headline")
            link = it.get("link") or it.get("url")
            pub = it.get("publisher") or it.get("provider") or it.get("source")
            ts = it.get("providerPublishTime") or it.get("pubDate") or None
            if ts and isinstance(ts, (int,float)):
                time = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
            else:
                time = ""
            out.append({"title": title, "link": link, "source": pub, "time": time})
        return out
    except Exception:
        return []

# -------------------------
# Fetch & display header
# -------------------------
info, fin, bs, cf, hist = fetch_yf(ticker) if use_live else ({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

company_name = info.get("shortName") or info.get("longName") or ticker
st.header(f"{company_name} — {ticker}")

# Top metrics
price = safe_number(info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan))
market_cap = safe_number(info.get("marketCap"))
shares_out = safe_number(info.get("sharesOutstanding") or info.get("floatShares") or np.nan)
ev, mc, debt, cash = (np.nan, np.nan, np.nan, np.nan)
if info:
    try:
        ev = safe_number(info.get("enterpriseValue"))
    except:
        ev = (market_cap or 0) + (safe_number(info.get("totalDebt")) or 0) - (safe_number(info.get("totalCash")) or 0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price (approx)", f"${price:,.2f}" if not np.isnan(price) else "N/A")
col2.metric("Market Cap", f"${market_cap:,.0f}" if not np.isnan(market_cap) else "N/A")
col3.metric("Enterprise Value", f"${ev:,.0f}" if not np.isnan(ev) else "N/A")
col4.metric("Shares Outstanding", f"{int(shares_out):,}" if not np.isnan(shares_out) else "N/A")
col5.metric("Sector / Industry", f"{info.get('sector','N/A')} / {info.get('industry','N/A')}")

st.markdown("---")

# -------------------------
# Price chart
# -------------------------
st.subheader("Price chart (candles)")
if not hist.empty:
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                         open=hist['Open'],
                                         high=hist['High'],
                                         low=hist['Low'],
                                         close=hist['Close'])])
    fig.update_layout(template="plotly_dark", height=420, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Price history not available.")

# -------------------------
# Financial statements display
# -------------------------
st.subheader("Financial Statements (yfinance)")
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Income Statement**")
    if fin is None or fin.empty:
        st.info("Income statement not available.")
    else:
        st.dataframe(style_numeric(fin), use_container_width=True)
with f2:
    st.markdown("**Balance Sheet**")
    if bs is None or bs.empty:
        st.info("Balance sheet not available.")
    else:
        st.dataframe(style_numeric(bs), use_container_width=True)
with f3:
    st.markdown("**Cash Flow**")
    if cf is None or cf.empty:
        st.info("Cash flow not available.")
    else:
        st.dataframe(style_numeric(cf), use_container_width=True)

st.markdown("---")

# -------------------------
# DCF interactive
# -------------------------
st.subheader("DCF Valuation (interactive)")

# Attempt derive FCF
last_fcf = None
# common labels to search for
ocf_labels = ["Total Cash From Operating Activities", "Net cash provided by operating activities", "Operating cash flow", "Net cash from operating activities", "Net cash provided by operating activities"]
capex_labels = ["Capital Expenditures", "Capital Expenditure", "Purchase of property", "Purchases of property"]

ocf_val = find_row_value(cf, ["operat", "cash from operating", "net cash provided"])
capex_val = find_row_value(cf, ["capital expend", "purchase of property", "payments for property"])
if ocf_val is not None:
    ocf_val = safe_number(ocf_val)
if capex_val is not None:
    capex_val = safe_number(capex_val)
if ocf_val is not None:
    # capex might be negative, so FCF = OCF + CAPEX
    cap = capex_val if capex_val is not None else 0.0
    last_fcf = ocf_val + cap

# fallback manual input if not found
if last_fcf is None or np.isnan(last_fcf):
    last_fcf = st.number_input("Manual: most recent unlevered FCF (USD)", value=500_000_000.0, step=1000000.0, format="%.0f")
else:
    st.write(f"Derived last FCF (best-effort): **${last_fcf:,.0f}** — you can override manually below if needed.")
last_fcf = st.number_input("Use this FCF (you may override)", value=float(last_fcf), format="%.0f", step=1000000.0)

g = st.slider("Explicit FCF CAGR (annual %)", min_value=-10.0, max_value=30.0, value=5.0)/100.0
d = st.slider("Discount rate / WACC (%)", min_value=0.1, max_value=30.0, value=float(default_wacc*100))/100.0
tg = st.slider("Terminal growth (%)", min_value=-2.0, max_value=6.0, value=float(default_tg*100))/100.0
years = st.selectbox("Projection years", [3,5,7,10], index=1)

result = dcf_from_fcf(last_fcf, g, d, tg, years)
ev_calc = result["enterprise_value"]
terminal_pv = result["terminal_pv"]
proj_pv = result["proj_pv"]
equity_val = ev_calc - (safe_number(info.get("totalDebt") or 0)) + (safe_number(info.get("totalCash") or 0))
implied_price = equity_val / shares_out if (shares_out and shares_out>0) else np.nan

st.metric("Enterprise value (DCF)", f"${ev_calc:,.0f}")
st.metric("Equity value (net debt adj)", f"${equity_val:,.0f}")
st.metric("Implied price per share", f"${implied_price:,.2f}" if not np.isnan(implied_price) else "N/A")

# DCF components plot
fig_dcf = go.Figure()
fig_dcf.add_trace(go.Bar(x=[f"Y{i}" for i in range(1, years+1)], y=result["proj_pv"], name="Discounted FCF", marker_color="#00CC96"))
fig_dcf.add_trace(go.Bar(x=["Terminal"], y=[terminal_pv if terminal_pv is not None else 0], name="Terminal PV", marker_color="#f5c518"))
fig_dcf.update_layout(template="plotly_dark", barmode="stack", title="DCF PV contributions")
st.plotly_chart(fig_dcf, use_container_width=True)

st.markdown("---")

# -------------------------
# Options pricing (Black-Scholes)
# -------------------------
st.subheader("Options Pricing (Black–Scholes)")

col1, col2, col3, col4, col5 = st.columns(5)
S_default = price if not np.isnan(price) else 100.0
S = col1.number_input("Underlying Price (S)", value=float(S_default))
K = col2.number_input("Strike (K)", value=float(S_default))
days = col3.number_input("Days to Expiry", min_value=1, max_value=3650, value=30)
r = col4.number_input("Risk-free rate (annual %)", value=0.5)/100.0
# estimate sigma (historic 30-day annualized) if hist exists
if not hist.empty:
    hist_ret = hist["Close"].pct_change().dropna()
    sigma_est = hist_ret.rolling(21).std().dropna().iloc[-1] * np.sqrt(252) if len(hist_ret)>21 else hist_ret.std()*np.sqrt(252)
    sigma_est = float(sigma_est) if not np.isnan(sigma_est) else 0.25
else:
    sigma_est = 0.25
sigma = col5.number_input("Volatility (annual σ)", value=float(sigma_est), format="%.4f")
T = days / 365.0

call_val = black_scholes_price(S, K, T, r, sigma, option="call")
put_val = black_scholes_price(S, K, T, r, sigma, option="put")
st.write(f"Call price ≈ **${call_val:,.2f}** — Put price ≈ **${put_val:,.2f}**")
st.caption("Black–Scholes is for European-style options; use with caution for American options.")

st.markdown("---")

# -------------------------
# News feed
# -------------------------
st.subheader("Company News (Yahoo feed)")
news_items = get_yahoo_news(ticker, limit=8)
if news_items:
    for n in news_items:
        tlink = n.get("link") or n.get("url") or "#"
        title = n.get("title") or "No title"
        source = n.get("source") or ""
        time = n.get("time") or ""
        st.markdown(f"- [{title}]({tlink}) <small>({source}) {time}</small>", unsafe_allow_html=True)
else:
    st.info("No news found or Yahoo API blocked. Try again later or add a custom news source.")

st.markdown("---")
st.write("⚠️ **Caveats:** This is an interview / educational tool. Verify all values against official filings for professional use. For IB/PE-level models you should reconstruct a 3-statement model, compute WACC precisely, and model financing & dilution for projects.")
