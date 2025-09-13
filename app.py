# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
from numpy import log, sqrt, exp

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="Equity Valuation Terminal",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bloomberg-like look
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    h1, h2, h3 {
        color: #39FF14; /* Bloomberg neon green */
    }
    .stDataFrame, .stTable {
        background-color: #1e222b;
        border: 1px solid #444;
    }
    .metric-box {
        background-color: #1e222b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Helper Functions
# ==============================
def dcf_valuation(ticker):
    """Run a simple DCF model for valuation."""
    stock = yf.Ticker(ticker)
    cf = stock.cashflow
    if cf is None or cf.empty:
        return None
    
    try:
        fcf = cf.loc["Total Cash From Operating Activities"] - cf.loc["Capital Expenditures"]
        fcf = fcf.dropna().values
    except:
        return None

    if len(fcf) < 3:
        return None

    last_fcf = fcf[0]
    growth_rate = np.mean([(fcf[i] / fcf[i+1]) - 1 for i in range(len(fcf)-1)])
    discount_rate = 0.1
    perpetuity_growth = 0.02
    projection_years = 5

    projected_fcfs = []
    for i in range(1, projection_years+1):
        projected_fcfs.append(last_fcf * ((1 + growth_rate) ** i))

    discount_factors = [(1 / ((1 + discount_rate) ** i)) for i in range(1, projection_years+1)]
    npv_fcfs = sum([projected_fcfs[i] * discount_factors[i] for i in range(projection_years)])

    terminal_value = projected_fcfs[-1] * (1 + perpetuity_growth) / (discount_rate - perpetuity_growth)
    terminal_value_discounted = terminal_value * discount_factors[-1]

    intrinsic_value = npv_fcfs + terminal_value_discounted
    shares_outstanding = stock.info.get("sharesOutstanding", 1)
    intrinsic_per_share = intrinsic_value / shares_outstanding if shares_outstanding else None

    return intrinsic_per_share

def black_scholes(S, K, T, r, sigma, option="call"):
    """Black-Scholes pricing model."""
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def get_news(ticker):
    """Pull company news from Yahoo Finance API."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    try:
        response = requests.get(url).json()
        if "news" in response:
            return response["news"][:5]
    except:
        return []
    return []

# ==============================
# Sidebar
# ==============================
st.sidebar.title("💹 Equity Valuation Terminal")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, WYNN)", "WYNN")

# ==============================
# Main Section
# ==============================
st.title(f"Equity Valuation & Analysis: {ticker}")

stock = yf.Ticker(ticker)

# Company Info
with st.expander("📘 Company Profile"):
    st.write(stock.info.get("longBusinessSummary", "No description available."))

# Stock Price Chart
hist = stock.history(period="1y")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name="Price"
))
fig.update_layout(title="Stock Price (1Y)", template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# Financials
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Income Statement")
    st.dataframe(stock.financials.fillna(""))
with col2:
    st.subheader("Balance Sheet")
    st.dataframe(stock.balance_sheet.fillna(""))
with col3:
    st.subheader("Cash Flow Statement")
    st.dataframe(stock.cashflow.fillna(""))

# Valuation (DCF)
st.subheader("📊 Valuation (DCF)")
intrinsic_value = dcf_valuation(ticker)
if intrinsic_value:
    current_price = stock.history(period="1d")["Close"].iloc[-1]
    st.metric(label="Intrinsic Value (DCF)", value=f"${intrinsic_value:,.2f}")
    st.metric(label="Current Price", value=f"${current_price:,.2f}")
    if intrinsic_value > current_price:
        st.success("Stock appears **undervalued** ✅")
    else:
        st.error("Stock appears **overvalued** ❌")
else:
    st.warning("Not enough data for DCF valuation.")

# Options Pricing (Black-Scholes)
st.subheader("📝 Black-Scholes Option Pricing")
col1, col2, col3, col4, col5 = st.columns(5)
S = col1.number_input("Stock Price", value=float(hist["Close"][-1]))
K = col2.number_input("Strike Price", value=float(hist["Close"][-1]))
T = col3.number_input("Time to Maturity (Years)", value=1.0)
r = col4.number_input("Risk-Free Rate", value=0.05)
sigma = col5.number_input("Volatility (σ)", value=0.2)

call_price = black_scholes(S, K, T, r, sigma, "call")
put_price = black_scholes(S, K, T, r, sigma, "put")

st.write(f"**Call Option Value:** ${call_price:,.2f}")
st.write(f"**Put Option Value:** ${put_price:,.2f}")

# News Section
st.subheader("📰 Latest News")
news = get_news(ticker)
if news:
    for item in news:
        st.markdown(f"- [{item['title']}]({item['link']})")
else:
    st.write("No news found.")
