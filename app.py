# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import norm

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Equity Valuation & Modelling",
    page_icon="💹",
    layout="wide"
)

st.title("💹 Equity Valuation & Modelling Platform")
st.markdown("Built by **Navjot Dhah** | Professional financial modelling for IB / PE / AM roles")

# ------------------------------
# Sidebar: User Input
# ------------------------------
st.sidebar.header("Company Search")
ticker = st.sidebar.text_input("Enter Ticker (e.g., AAPL, MSFT, WYNN):", "AAPL")
period = st.sidebar.selectbox("Historical Period:", ["1y", "3y", "5y", "10y"], index=1)

# ------------------------------
# Fetch Data
# ------------------------------
try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    fin = stock.financials
    bal = stock.balance_sheet
    cf = stock.cashflow
    info = stock.info
except Exception as e:
    st.error(f"Error fetching data for {ticker}: {e}")
    st.stop()

# ------------------------------
# Price Chart
# ------------------------------
st.subheader(f"Stock Price Chart: {ticker}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
fig.update_layout(
    template="plotly_dark", 
    yaxis_title="Price (USD)", 
    xaxis_title="Date", 
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Display Financials
# ------------------------------
st.subheader("📊 Financial Statements")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Income Statement**")
    st.dataframe(fin.fillna(""))
with col2:
    st.markdown("**Balance Sheet**")
    st.dataframe(bal.fillna(""))
with col3:
    st.markdown("**Cash Flow Statement**")
    st.dataframe(cf.fillna(""))

# ------------------------------
# Linked 3-Statement Model (Simplified Projection)
# ------------------------------
st.subheader("📈 3-Statement Model Projection")
years = st.slider("Projection Years:", 3, 10, 5)

# Base revenue assumption from last reported year
try:
    base_rev = fin.loc["Total Revenue"].iloc[0]
except:
    base_rev = 1e9

growth_rate = st.number_input("Revenue Growth Rate (%)", value=5.0) / 100
ebit_margin = st.number_input("EBIT Margin (%)", value=15.0) / 100
tax_rate = st.number_input("Tax Rate (%)", value=21.0) / 100
dep_pct = st.number_input("Depreciation % of Revenue", value=5.0) / 100
capex_pct = st.number_input("CapEx % of Revenue", value=6.0) / 100
nwc_pct = st.number_input("Change in NWC % of Revenue", value=2.0) / 100

proj = []
rev = base_rev
for yr in range(1, years+1):
    rev = rev * (1 + growth_rate)
    ebit = rev * ebit_margin
    tax = ebit * tax_rate
    nopat = ebit - tax
    dep = rev * dep_pct
    capex = rev * capex_pct
    nwc = rev * nwc_pct
    fcf = nopat + dep - capex - nwc
    proj.append([datetime.now().year + yr, rev, ebit, nopat, fcf])

proj_df = pd.DataFrame(proj, columns=["Year", "Revenue", "EBIT", "NOPAT", "FCF"])
st.dataframe(proj_df.style.format("{:,.0f}"))

# ------------------------------
# WACC Calculation
# ------------------------------
st.subheader("⚖️ Weighted Average Cost of Capital (WACC)")

rf = st.number_input("Risk-Free Rate (%)", value=4.0) / 100
beta = st.number_input("Beta (from CAPM)", value=1.1)
mkt_return = st.number_input("Expected Market Return (%)", value=9.0) / 100
cost_of_equity = rf + beta * (mkt_return - rf)

pretax_cost_debt = st.number_input("Pre-Tax Cost of Debt (%)", value=5.0) / 100
tax_rate_for_wacc = st.number_input("Corporate Tax Rate (%)", value=21.0) / 100
cost_of_debt = pretax_cost_debt * (1 - tax_rate_for_wacc)

equity_val = st.number_input("Equity Value (Market Cap, $B)", value=info.get("marketCap", 1e10)/1e9) * 1e9
debt_val = st.number_input("Total Debt ($B)", value=bal.loc["Total Debt"].iloc[0]/1e9 if "Total Debt" in bal.index else 10.0) * 1e9

w_e = equity_val / (equity_val + debt_val)
w_d = debt_val / (equity_val + debt_val)
wacc = w_e * cost_of_equity + w_d * cost_of_debt

st.metric("WACC", f"{wacc*100:.2f}%")

# ------------------------------
# DCF Valuation
# ------------------------------
st.subheader("📉 DCF Valuation")
discount_factors = [(1/(1+wacc)**i) for i in range(1, years+1)]
dcf = (proj_df["FCF"] * discount_factors).sum()

terminal_growth = st.number_input("Terminal Growth Rate (%)", value=2.0) / 100
terminal_value = proj_df["FCF"].iloc[-1] * (1+terminal_growth) / (wacc - terminal_growth)
terminal_value_pv = terminal_value / ((1+wacc)**years)

ev = dcf + terminal_value_pv
equity_value = ev - debt_val
intrinsic_price = equity_value / info.get("sharesOutstanding", 1)

st.metric("Enterprise Value", f"${ev/1e9:.2f}B")
st.metric("Equity Value", f"${equity_value/1e9:.2f}B")
st.metric("Intrinsic Price / Share", f"${intrinsic_price:.2f}")

# ------------------------------
# Financing & Dilution
# ------------------------------
st.subheader("🏗️ Project Financing & Dilution Analysis")

project_cost = st.number_input("Project Cost ($B)", value=3.0) * 1e9
equity_raise = st.slider("Equity Financing Portion (%)", 0, 100, 50) / 100
debt_raise = 1 - equity_raise

new_equity = project_cost * equity_raise
new_debt = project_cost * debt_raise

issue_price = st.number_input("Equity Issue Price ($)", value=intrinsic_price)
new_shares = new_equity / issue_price
dilution = new_shares / info.get("sharesOutstanding", 1)

st.metric("New Debt", f"${new_debt/1e9:.2f}B")
st.metric("New Equity Raised", f"${new_equity/1e9:.2f}B")
st.metric("Share Dilution", f"{dilution*100:.2f}%")

# ------------------------------
# Black-Scholes Option Pricing
# ------------------------------
st.subheader("📊 Black-Scholes Option Pricing")

S = st.number_input("Current Stock Price (S)", value=float(hist["Close"].iloc[-1]))
K = st.number_input("Strike Price (K)", value=S*1.05)
T = st.number_input("Time to Maturity (Years)", value=1.0)
sigma = st.number_input("Volatility (σ)", value=0.3)
r = rf

d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

st.metric("Call Option Value", f"${call_price:.2f}")
st.metric("Put Option Value", f"${put_price:.2f}")

# ------------------------------
# News Feed
# ------------------------------
st.subheader("📰 Latest News")
if "longBusinessSummary" in info:
    st.write(info["longBusinessSummary"])
else:
    st.info("No news summary available.")
