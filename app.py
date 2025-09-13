# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
import requests

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Equity Valuation Terminal",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Equity Valuation & Modelling Platform")
st.markdown("An interactive tool for fundamental analysis, DCF valuation, comparables, options pricing, and real-time news.")

# -----------------------------
# Data Fetcher (Serializable)
# -----------------------------
@st.cache_data
def fetch_ticker_data(ticker):
    """Fetch serializable financial data for a ticker."""
    t = yf.Ticker(ticker)

    info = t.info if isinstance(t.info, dict) else {}
    hist = t.history(period="1y").reset_index()

    def safe_reset(df):
        try:
            return df.reset_index()
        except Exception:
            return pd.DataFrame()

    return {
        "info": info,
        "history": hist,
        "financials": safe_reset(t.financials),
        "balance_sheet": safe_reset(t.balance_sheet),
        "cashflow": safe_reset(t.cashflow),
        "earnings": safe_reset(t.earnings)
    }

# -----------------------------
# Helper Functions
# -----------------------------
def dcf_valuation(fcf, growth_rate, discount_rate, terminal_growth, years=5):
    fcf_projections = [fcf * ((1 + growth_rate) ** i) for i in range(1, years + 1)]
    discounted_fcfs = [fcf_projections[i] / ((1 + discount_rate) ** (i + 1)) for i in range(years)]
    terminal_value = fcf_projections[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    discounted_terminal = terminal_value / ((1 + discount_rate) ** years)
    intrinsic_value = sum(discounted_fcfs) + discounted_terminal
    return intrinsic_value, fcf_projections

def black_scholes(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

@st.cache_data
def fetch_news(ticker):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
    try:
        res = requests.get(url, timeout=5).json()
        return res.get("news", [])
    except Exception:
        return []

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔎 Search a Stock")
ticker_input = st.sidebar.text_input("Enter ticker (e.g., AAPL, MSFT, WYNN)", "WYNN").upper()

discount_rate = st.sidebar.slider("Discount Rate (WACC)", 0.05, 0.15, 0.09)
growth_rate = st.sidebar.slider("FCF Growth Rate", 0.01, 0.15, 0.05)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.01, 0.05, 0.025)

# -----------------------------
# Main Logic
# -----------------------------
if ticker_input:
    data = fetch_ticker_data(ticker_input)
    info = data["info"]

    st.subheader(f"📌 Company Overview: {info.get('longName', ticker_input)}")
    st.write(info.get("longBusinessSummary", "No company description available."))

    # Stock Price Chart
    st.subheader("📈 Stock Price (1Y)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["history"]["Date"],
        y=data["history"]["Close"],
        mode="lines",
        name="Close Price"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Financial Statements
    st.subheader("🧾 Financial Statements")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Income Statement**")
        st.dataframe(data["financials"])
    with col2:
        st.markdown("**Balance Sheet**")
        st.dataframe(data["balance_sheet"])
    with col3:
        st.markdown("**Cash Flow**")
        st.dataframe(data["cashflow"])

    # DCF Valuation
    st.subheader("💵 Discounted Cash Flow Valuation")
    try:
        fcf = data["cashflow"].loc[data["cashflow"]["index"] == "Total Cash From Operating Activities"].iloc[0, 1]
        capex = data["cashflow"].loc[data["cashflow"]["index"] == "Capital Expenditures"].iloc[0, 1]
        fcf = fcf + capex
        intrinsic_value, fcf_proj = dcf_valuation(fcf, growth_rate, discount_rate, terminal_growth)
        st.success(f"Estimated Intrinsic Value (Enterprise): **${intrinsic_value:,.0f}**")

        proj_df = pd.DataFrame({
            "Year": list(range(1, len(fcf_proj) + 1)),
            "Projected FCF": fcf_proj
        })
        st.bar_chart(proj_df.set_index("Year"))
    except Exception:
        st.warning("Unable to compute DCF (missing FCF or CapEx data).")

    # Black-Scholes Options
    st.subheader("📌 Options Valuation (Black-Scholes)")
    S = data["history"]["Close"].iloc[-1]
    K = S * 1.05
    T = 0.5
    r = 0.03
    sigma = data["history"]["Close"].pct_change().std() * np.sqrt(252)
    call_price = black_scholes(S, K, T, r, sigma, option="call")
    put_price = black_scholes(S, K, T, r, sigma, option="put")
    st.write(f"**Call Option (K={K:.2f}):** ${call_price:.2f}")
    st.write(f"**Put Option (K={K:.2f}):** ${put_price:.2f}")

    # News Feed
    st.subheader("📰 Latest News")
    news = fetch_news(ticker_input)
    if news:
        for article in news[:5]:
            st.markdown(f"- [{article['title']}]({article['link']})")
    else:
        st.write("No recent news found.")

