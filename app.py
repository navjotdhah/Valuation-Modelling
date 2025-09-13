# app.py
"""
Pro Valuation Terminal (Bloomberg-lite)
Features:
- Realtime price + charts
- Financial statements (yfinance)
- Linked 3-statement projection (simplified)
- WACC calculation (CAPM + cost of debt)
- DCF valuation + sensitivity + Monte Carlo (basic)
- Project financing & dilution modeling
- Comparable company multiples and implied price
- Options (Black-Scholes + Greeks)
- News feed: Yahoo scraping (no key) or NewsAPI (optional key)
- Export to Excel
Author: Navjot Dhah
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import requests
import math
from datetime import datetime, date, timedelta

# Attempt to import scipy.norm; fallback to math.erf-based cdf
try:
    from scipy.stats import norm
except Exception:
    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        @staticmethod
        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    norm = _NormFallback()

# -------------------------
# Page and CSS
# -------------------------
st.set_page_config(page_title="Pro Valuation Terminal", page_icon="💹", layout="wide")
st.markdown("""
<style>
body { background-color: #0b0d10; color: #e8e8e8; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h1,h2,h3 { color: #f5c518; font-weight:700; }
.metric-card { background: #0f1113; padding: 10px; border-radius: 8px; border: 1px solid #222; }
.sidebar .stButton > button { background-color:#f5c518 !important; color:#000 !important; }
a { color: #7ef9a4; }
</style>
""", unsafe_allow_html=True)

st.title("💹 Pro Valuation Terminal — Bloomberg-lite")
st.caption("Built by Navjot Dhah — DCF, 3-statement, WACC, comps, options & real-time news")

# -------------------------
# Sidebar - global controls
# -------------------------
st.sidebar.header("Global settings")
ticker_input = st.sidebar.text_input("Ticker (e.g. AAPL, WYNN):", value="WYNN").upper().strip()
history_period = st.sidebar.selectbox("Price history range", ["6mo","1y","3y","5y","10y"], index=1)
auto_refresh = st.sidebar.checkbox("Auto-refresh market data on start", value=True)
newsapi_key = st.sidebar.text_input("Optional NewsAPI.org key (optional)", value="")
use_newsapi = st.sidebar.checkbox("Use NewsAPI (if key provided)", value=False)

# Modeling defaults
st.sidebar.markdown("---")
st.sidebar.subheader("Model defaults")
default_rf = st.sidebar.number_input("Risk-free rate (%)", value=4.0, step=0.1) / 100.0
default_mrp = st.sidebar.number_input("Market risk premium (%)", value=6.0, step=0.1) / 100.0
default_tax = st.sidebar.number_input("Corporate tax rate (%)", value=21.0, step=0.5) / 100.0
st.sidebar.markdown("Tip: For production-grade workflows provide your own NewsAPI key and pin dependency versions in requirements.txt.")

# -------------------------
# Utilities & helpers
# -------------------------
@st.cache_data(ttl=300)
def fetch_ticker_data(ticker):
    t = yf.Ticker(ticker)
    # safe calls
    try:
        info = t.info
    except Exception:
        info = {}
    try:
        hist = t.history(period=history_period)
    except Exception:
        hist = pd.DataFrame()
    try:
        fin = t.financials
    except Exception:
        fin = pd.DataFrame()
    try:
        bs = t.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        cf = t.cashflow
    except Exception:
        cf = pd.DataFrame()
    # try option chain (may be missing)
    try:
        options_dates = t.options
    except Exception:
        options_dates = []
    return {"ticker": t, "info": info, "hist": hist, "financials": fin, "balance_sheet": bs, "cashflow": cf, "options_dates": options_dates}

def safe_float(x, fallback=np.nan):
    try:
        return float(x)
    except Exception:
        return fallback

def style_numeric(df):
    if df is None or df.empty:
        return df
    df_t = df.T
    numeric_cols = df_t.select_dtypes(include=[np.number]).columns
    fmt_map = {c: "{:,.0f}" for c in numeric_cols}
    return df_t.style.format(fmt_map)

def black_scholes(S, K, T, r, sigma, option="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S-K) if option=="call" else max(0.0, K-S)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    # Greeks (approx)
    try:
        delta = norm.cdf(d1) if option=="call" else (norm.cdf(d1)-1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        theta = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * norm.cdf(d2 if option=="call" else -d2)
    except Exception:
        delta = gamma = vega = theta = np.nan
    return price, {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def yahoo_news_search(ticker, limit=10):
    # lightweight Yahoo search endpoint (no key) — can be flaky
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}&quotesCount=1&newsCount={limit}"
        resp = requests.get(url, timeout=6).json()
        news_list = resp.get("news", []) or resp.get("items", [])
        out = []
        for n in news_list[:limit]:
            out.append({
                "title": n.get("title"),
                "link": n.get("link") or n.get("url"),
                "provider": n.get("provider") or n.get("publisher"),
                "time": n.get("providerPublishTime")
            })
        return out
    except Exception:
        return []

def newsapi_search(q, api_key, limit=10):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": q, "apiKey": api_key, "pageSize": limit, "sortBy": "publishedAt", "language": "en"}
        resp = requests.get(url, params=params, timeout=6).json()
        articles = resp.get("articles", [])[:limit]
        out = []
        for a in articles:
            out.append({"title": a.get("title"), "link": a.get("url"), "provider": a.get("source", {}).get("name"), "time": a.get("publishedAt")})
        return out
    except Exception:
        return []

# -------------------------
# Fetch & display header data
# -------------------------
if not ticker_input:
    st.error("Enter a ticker in the left sidebar.")
    st.stop()

data = fetch_ticker_data(ticker_input)
info = data["info"]
hist = data["hist"]
fin = data["financials"]
bs = data["balance_sheet"]
cf = data["cashflow"]
yf_t = data["ticker"]  # yfinance Ticker object
options_dates = data["options_dates"]

company_name = info.get("shortName") or info.get("longName") or ticker_input
st.header(f"{company_name}  —  {ticker_input}")

# Top metrics row
price = safe_float(info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan))
market_cap = safe_float(info.get("marketCap"))
shares_out = safe_float(info.get("sharesOutstanding") or info.get("floatShares") or np.nan)
enterprise_value = safe_float(info.get("enterpriseValue") or (market_cap + safe_float(info.get("totalDebt") or 0) - safe_float(info.get("totalCash") or 0)))
beta = safe_float(info.get("beta"))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price (approx)", f"${price:,.2f}" if not np.isnan(price) else "N/A")
col2.metric("Market Cap", f"${market_cap:,.0f}" if not np.isnan(market_cap) else "N/A")
col3.metric("Enterprise Value", f"${enterprise_value:,.0f}" if not np.isnan(enterprise_value) else "N/A")
col4.metric("Shares Outstanding", f"{int(shares_out):,}" if not np.isnan(shares_out) else "N/A")
col5.metric("Beta (yf)", f"{beta:.2f}" if not np.isnan(beta) else "N/A")

st.markdown("---")

# -------------------------
# Price panel: candles + indicators
# -------------------------
st.subheader("Price & Technicals")
if hist is None or hist.empty:
    st.info("Price history not available via yfinance.")
else:
    # compute moving averages
    hist = hist.copy()
    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], name="MA50", line=dict(color="#FFD54F")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], name="MA200", line=dict(color="#90CAF9")))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    # volume chart
    fig2 = go.Figure([go.Bar(x=hist.index, y=hist["Volume"], marker_color="#4db6ac")])
    fig2.update_layout(template="plotly_dark", title="Volume", height=180)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -------------------------
# Tabs for organization
# -------------------------
tabs = st.tabs(["Financials", "3-Statement Model", "Valuation (DCF)", "Comps", "Options", "News & Feed", "Export"])

# -------------------------
# Financials tab
# -------------------------
with tabs[0]:
    st.subheader("Financial Statements (yfinance)")
    st.write("If statements are missing or look odd, use manual overrides below.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Income Statement**")
        if fin is None or fin.empty:
            st.info("Income statement not available via yfinance.")
        else:
            st.dataframe(style_numeric(fin), use_container_width=True)
    with c2:
        st.markdown("**Balance Sheet**")
        if bs is None or bs.empty:
            st.info("Balance sheet not available via yfinance.")
        else:
            st.dataframe(style_numeric(bs), use_container_width=True)
    with c3:
        st.markdown("**Cash Flow Statement**")
        if cf is None or cf.empty:
            st.info("Cash flow not available via yfinance.")
        else:
            st.dataframe(style_numeric(cf), use_container_width=True)

# -------------------------
# 3-statement model tab
# -------------------------
with tabs[1]:
    st.subheader("Linked 3-Statement Model (simplified)")
    st.markdown("This builds a simplified projection across the three statements. For IB-grade models you'd reconstruct each line from filings.")

    proj_years = st.slider("Projection years", min_value=3, max_value=10, value=5)
    # base revenue & last-year numbers fallback
    last_revenue = None
    try:
        if not fin.empty:
            # yfinance financials columns are year-end columns with latest first
            last_revenue = safe_float(fin.loc["Total Revenue"].iloc[0])
    except Exception:
        last_revenue = None
    last_revenue = st.number_input("Most recent revenue (USD) (auto if available)", value=last_revenue if last_revenue else 1_000_000_000.0, step=1e6, format="%.0f")
    rev_growth = st.number_input("Revenue growth (annual %)", value=5.0, step=0.1)/100.0
    ebit_margin = st.number_input("EBIT margin (%)", value=15.0, step=0.1)/100.0
    dep_pct = st.number_input("Depreciation & Amortization as % of revenue", value=3.0, step=0.1)/100.0
    capex_pct = st.number_input("CapEx as % of revenue", value=4.0, step=0.1)/100.0
    nwc_pct = st.number_input("Change in NWC as % of revenue", value=0.5, step=0.1)/100.0
    tax_rate_model = st.number_input("Tax rate (%)", value=default_tax*100 if default_tax else 21.0)/100.0

    # build projection
    proj_rows = []
    revenue = last_revenue
    for i in range(1, proj_years+1):
        revenue = revenue * (1+rev_growth)
        ebit = revenue * ebit_margin
        dep = revenue * dep_pct
        capex = revenue * capex_pct
        change_nwc = revenue * nwc_pct
        tax = max(0.0, ebit) * tax_rate_model  # simplistic
        nopat = ebit - tax
        fcf = nopat + dep - capex - change_nwc
        proj_rows.append({"Year": datetime.now().year + i, "Revenue": revenue, "EBIT": ebit, "Depreciation": dep, "CapEx": -capex, "ΔNWC": -change_nwc, "NOPAT": nopat, "FCF": fcf})
    proj_df = pd.DataFrame(proj_rows)
    st.dataframe(proj_df.style.format({"Revenue":"${:,.0f}", "EBIT":"${:,.0f}", "Depreciation":"${:,.0f}", "CapEx":"${:,.0f}", "NOPAT":"${:,.0f}", "FCF":"${:,.0f}"}), use_container_width=True)

    st.markdown("The projection above flows into the DCF in the Valuation tab. You can export these schedules in Export tab.")

# -------------------------
# Valuation (DCF) tab
# -------------------------
with tabs[2]:
    st.subheader("DCF Valuation (from projected FCF)")
    st.markdown("Adjust discount rate (WACC) components, run sensitivity and basic Monte Carlo.")

    # attempt to compute market-value weights
    # get market cap and debt values
    market_cap_val = market_cap
    total_debt = safe_float(info.get("totalDebt") or info.get("totalDebt", np.nan))
    total_cash = safe_float(info.get("totalCash") or info.get("totalCash", np.nan))
    # user input overrides
    st.markdown("**WACC inputs** (editable)")
    rf = st.number_input("Risk-free rate (%)", value=default_rf*100)/100.0
    mrp = st.number_input("Market risk premium (%)", value=default_mrp*100)/100.0
    beta_input = st.number_input("Equity beta", value=beta if not np.isnan(beta) else 1.0, step=0.01)
    cost_of_equity = rf + beta_input * mrp
    pretax_cost_of_debt = st.number_input("Pre-tax cost of debt (%)", value=4.0)/100.0
    corp_tax = st.number_input("Tax rate (%)", value=default_tax*100)/100.0
    cost_of_debt_aftertax = pretax_cost_of_debt * (1 - corp_tax)

    # weights
    equity_mv = market_cap_val if not np.isnan(market_cap_val) else (shares_out * price if (not np.isnan(shares_out) and not np.isnan(price)) else np.nan)
    debt_mv = total_debt if not np.isnan(total_debt) else 0.0
    w_e = equity_mv / (equity_mv + debt_mv) if (not np.isnan(equity_mv) and (equity_mv + debt_mv)>0) else 0.8
    w_d = 1 - w_e
    wacc = w_e * cost_of_equity + w_d * cost_of_debt_aftertax
    st.write(f"Computed WACC = {wacc*100:.2f}% (weights: E={w_e*100:.1f}%, D={w_d*100:.1f}%)")

    # connect to last proj_df built in 3-statement tab
    try:
        fcfs = proj_df["FCF"].tolist()
    except Exception:
        # fallback: if no proj_df, ask user to input last FCF and growth
        last_fcf_manual = st.number_input("Most recent unlevered FCF (USD)", value=500_000_000.0, step=1e6)
        g_manual = st.number_input("Explicit FCF growth (annual %)", value=5.0)/100.0
        years_manual = st.number_input("Projection years", min_value=3, max_value=10, value=5)
        fcfs = [last_fcf_manual * (1+g_manual)**i for i in range(1, years_manual+1)]

    # discounting
    disc_rate = st.number_input("Discount rate (WACC override %)", value=wacc*100 if not np.isnan(wacc) else 9.0)/100.0
    term_g = st.number_input("Terminal growth rate (%)", value=2.0)/100.0
    explicit_n = len(fcfs)
    pv_fcfs = [fcfs[i]/((1+disc_rate)**(i+1)) for i in range(len(fcfs))]

    if disc_rate <= term_g:
        st.error("Discount rate must be greater than terminal growth to compute terminal value.")
        terminal_pv = np.nan
    else:
        terminal_nominal = fcfs[-1] * (1+term_g) / (disc_rate - term_g)
        terminal_pv = terminal_nominal / ((1+disc_rate)**explicit_n)

    ev_dcf = sum(pv_fcfs) + (terminal_pv if not np.isnan(terminal_pv) else 0)
    equity_value_dcf = ev_dcf - debt_mv + total_cash
    implied_price_dcf = equity_value_dcf / shares_out if (not np.isnan(shares_out) and shares_out>0) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Enterprise value (DCF)", f"${ev_dcf:,.0f}")
    c2.metric("Equity value (net debt adj)", f"${equity_value_dcf:,.0f}")
    c3.metric("Implied price per share (DCF)", f"${implied_price_dcf:,.2f}" if not np.isnan(implied_price_dcf) else "N/A")

    # sensitivity table (WACC vs term_g)
    st.markdown("### Sensitivity: Implied price (WACC vs Terminal growth)")
    wacc_vals = np.round(np.linspace(max(0.01, disc_rate-0.03), disc_rate+0.03, 7), 4)
    tg_vals = np.round(np.linspace(max(-0.01, term_g-0.01), term_g+0.01, 7), 4)
    sens_matrix = []
    for tg in tg_vals:
        row = []
        for wa in wacc_vals:
            try:
                if wa <= tg:
                    row.append(np.nan)
                else:
                    pv = sum([fcfs[i]/((1+wa)**(i+1)) for i in range(len(fcfs))])
                    term = (fcfs[-1]*(1+tg)/(wa-tg))/((1+wa)**explicit_n)
                    ev = pv + term
                    eq = ev - debt_mv + total_cash
                    price = eq / shares_out if (not np.isnan(shares_out) and shares_out) else np.nan
                    row.append(price)
            except Exception:
                row.append(np.nan)
        sens_matrix.append(row)
    sens_df = pd.DataFrame(sens_matrix, index=[f"{tg*100:.2f}%" for tg in tg_vals], columns=[f"{wa*100:.2f}%" for wa in wacc_vals])
    fig_heat = go.Figure(data=go.Heatmap(z=sens_df.values, x=sens_df.columns, y=sens_df.index, colorscale="Viridis"))
    fig_heat.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Monte Carlo (basic) — sample growth & WACC
    st.markdown("### Monte Carlo (basic) — optional")
    run_mc = st.checkbox("Run Monte Carlo", value=False)
    if run_mc:
        mc_sims = st.number_input("Simulations", value=3000, min_value=100, step=100)
        g_mu = st.number_input("MC: mean FCF growth (%)", value=rev_growth*100)/100.0
        g_sigma = st.number_input("MC: sigma for growth (%)", value=5.0)/100.0
        wacc_mu = disc_rate
        wacc_sigma = st.number_input("MC: sigma for WACC (%)", value=0.01)/100.0
        rng = np.random.default_rng()
        mc_prices = []
        for i in range(int(mc_sims)):
            g_sim = rng.normal(g_mu, g_sigma)
            w_sim = abs(rng.normal(wacc_mu, wacc_sigma))
            # simple MC: re-project fcfs using g_sim and discount using w_sim
            proj_sim = [fcfs[0]*(1+g_sim)**(k+1) for k in range(len(fcfs))]
            pv_sim = sum([proj_sim[k]/((1+w_sim)**(k+1)) for k in range(len(proj_sim))])
            if w_sim<=term_g:
                term_sim = np.nan
            else:
                term_nom_sim = proj_sim[-1]*(1+term_g)/(w_sim-term_g)
                term_sim = term_nom_sim/((1+w_sim)**explicit_n)
            ev_sim = pv_sim + (term_sim if not np.isnan(term_sim) else 0)
            eq_sim = ev_sim - debt_mv + total_cash
            price_sim = eq_sim / shares_out if (not np.isnan(shares_out) and shares_out) else np.nan
            mc_prices.append(price_sim)
        mc_arr = np.array([p for p in mc_prices if not np.isnan(p)])
        if mc_arr.size>0:
            st.write(f"MC results — mean: ${np.nanmean(mc_arr):,.2f}, median: ${np.nanmedian(mc_arr):,.2f}, 5th pct: ${np.nanpercentile(mc_arr,5):,.2f}, 95th pct: ${np.nanpercentile(mc_arr,95):,.2f}")
            fig_mc = px.histogram(mc_arr, nbins=50, title="Monte Carlo implied price distribution", marginal="box", template="plotly_dark")
            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("Monte Carlo returned no valid sims. Adjust inputs.")

# -------------------------
# Comparables tab
# -------------------------
with tabs[3]:
    st.subheader("Comparable Companies (quick comps)")
    peers_input = st.text_input("Enter peer tickers (comma separated)", value="MGM,LVS,MLCO")
    peers = [p.strip().upper() for p in peers_input.split(",") if p.strip()]
    rows = []
    for p in peers:
        try:
            p_t = yf.Ticker(p)
            p_info = p_t.info
            p_ev = safe_float(p_info.get("enterpriseValue"))
            p_ebitda = safe_float(p_info.get("ebitda"))
            p_price = safe_float(p_info.get("currentPrice"))
            p_mc = safe_float(p_info.get("marketCap"))
            rows.append({"Ticker": p, "EV": p_ev, "EBITDA": p_ebitda, "EV/EBITDA": (p_ev/p_ebitda if p_ev and p_ebitda and p_ebitda>0 else np.nan), "Price": p_price, "MarketCap": p_mc})
        except Exception:
            rows.append({"Ticker": p})
    if rows:
        peer_df = pd.DataFrame(rows).set_index("Ticker")
        st.dataframe(peer_df.style.format({c: "{:,.2f}" for c in peer_df.select_dtypes(include=[np.number]).columns}), use_container_width=True)
        # implied price from median EV/EBITDA
        med = peer_df["EV/EBITDA"].dropna().median() if "EV/EBITDA" in peer_df.columns else np.nan
        target_ebitda = safe_float(info.get("ebitda"))
        if not np.isnan(med) and target_ebitda and target_ebitda>0:
            implied_ev = med * target_ebitda
            implied_eq = implied_ev - debt_mv + total_cash
            implied_price = implied_eq / shares_out if (not np.isnan(shares_out) and shares_out) else np.nan
            st.markdown(f"**Implied price from peer median EV/EBITDA ({med:.2f}x)**: ${implied_price:,.2f}")
        else:
            st.info("Insufficient data to compute implied price from comps.")

# -------------------------
# Options tab
# -------------------------
with tabs[4]:
    st.subheader("Options & Black–Scholes")
    st.markdown("Use Black–Scholes (European) for a quick estimate; Greeks are provided.")
    S_default = price if not np.isnan(price) else 100.0
    S = st.number_input("Underlying price (S)", value=float(S_default))
    K = st.number_input("Strike (K)", value=float(S_default))
    days = st.number_input("Days to expiry", min_value=1, max_value=3650, value=30)
    r_bs = st.number_input("Risk-free rate (%)", value=default_rf*100)/100.0
    # estimate sigma from hist (30d annualized)
    if not hist.empty:
        ret = hist["Close"].pct_change().dropna()
        sigma_est = float(ret.rolling(21).std().dropna().iloc[-1] * math.sqrt(252)) if len(ret)>21 else float(ret.std()*math.sqrt(252))
    else:
        sigma_est = 0.30
    sigma = st.number_input("Volatility (annual, σ)", value=float(sigma_est), format="%.4f")
    T = days/365.0
    call_price, call_greeks = black_scholes(S, K, T, r_bs, sigma, option="call")
    put_price, put_greeks = black_scholes(S, K, T, r_bs, sigma, option="put")
    st.metric("Call (BS)", f"${call_price:,.2f}")
    st.metric("Put (BS)", f"${put_price:,.2f}")
    st.write("Greeks (call): ", {k: (f"{v:.4f}" if v is not None else "N/A") for k,v in call_greeks.items()})

    # option chain viewing (if available)
    if options_dates:
        st.markdown("**Available option expiry dates**")
        selected_date = st.selectbox("Select expiry", options=options_dates)
        if selected_date:
            try:
                opt_chain = yf_t.option_chain(selected_date)
                calls = opt_chain.calls
                puts = opt_chain.puts
                st.write("Top 10 calls (by volume)")
                st.dataframe(calls.sort_values("volume", ascending=False).head(10), use_container_width=True)
                st.write("Top 10 puts (by volume)")
                st.dataframe(puts.sort_values("volume", ascending=False).head(10), use_container_width=True)
            except Exception as e:
                st.info("Option chain not available for this ticker or date.")

# -------------------------
# News & Feed tab (real-time)
# -------------------------
with tabs[5]:
    st.subheader("Real-time News & Feed")
    st.markdown("Choose NewsAPI for robust full-article results (requires API key) or use lightweight Yahoo feed (no key).")
    if use_newsapi and newsapi_key:
        st.write("Using NewsAPI.org")
        articles = newsapi_search(ticker_input, newsapi_key, limit=15)
    else:
        st.write("Using Yahoo Finance lightweight feed")
        articles = yahoo_news_search(ticker_input, limit=15)
    if articles:
        for a in articles:
            title = a.get("title") or "No title"
            link = a.get("link") or a.get("url") or "#"
            provider = a.get("provider") or a.get("source") or ""
            tstamp = a.get("time") or ""
            # if numeric timestamp convert
            if isinstance(tstamp, (int,float)):
                try:
                    tstamp = datetime.fromtimestamp(int(tstamp)).strftime("%Y-%m-%d %H:%M")
                except:
                    tstamp = ""
            st.markdown(f"- [{title}]({link}) — {provider} {tstamp}")
    else:
        st.info("No news found. If you need robust news, add a NewsAPI key in the sidebar.")

# -------------------------
# Export tab
# -------------------------
with tabs[6]:
    st.subheader("Export model & data")
    st.markdown("Download Excel with: summary, projection, and statements.")
    # build excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        # summary
        summary = pd.DataFrame({
            "Metric":["Ticker","Price","MarketCap","EnterpriseValue","SharesOutstanding","ComputedWACC","ImpliedPriceDCF"],
            "Value":[ticker_input, price, market_cap, enterprise_value, shares_out, wacc if 'wacc' in locals() else np.nan, implied_price_dcf if 'implied_price_dcf' in locals() else np.nan]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
        # write statements if present
        try:
            if not fin.empty:
                fin.to_excel(writer, sheet_name="IncomeStatement")
            if not bs.empty:
                bs.to_excel(writer, sheet_name="BalanceSheet")
            if not cf.empty:
                cf.to_excel(writer, sheet_name="CashFlow")
            # project schedule if present
            try:
                proj_df.to_excel(writer, sheet_name="Projection", index=False)
            except Exception:
                pass
        except Exception:
            pass
        writer.save()
        data = buf.getvalue()
    st.download_button("Download Excel", data=data, file_name=f"{ticker_input}_valuation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.write("Pro Valuation Terminal — interactive analyst toolkit. Customize assumptions and always validate model outputs against filings when making actionable decisions.")
