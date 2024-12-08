import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from streamlit_autorefresh import st_autorefresh

# Automatically refresh every 60 second
count = st_autorefresh(interval=60 * 1000, key="refresh")

# Title
st.title("💼 Investment Tracker Dashboard")
st.markdown("Track your investments with smooth, interactive charts and detailed breakdowns.")

# Fetch USD to AUD exchange rate using yfinance
def get_exchange_rate():
    try:
        forex_data = yf.Ticker("AUDUSD=X")
        exchange_rate = forex_data.history(period="1d")["Close"].iloc[-1]
        return 1 / exchange_rate  # Flip the rate to USD -> AUD
    except Exception as e:
        return 1.5  # Default to 1.5 if fetching fails

exchange_rate = get_exchange_rate()
st.markdown(f"### 💱 Exchange Rate: {exchange_rate:.4f} USD to AUD")

# User's current holdings
holdings = {
    "Ticker": ["AMZN", "DBX", "PYPL", "TSLA", "VOO", "VOOG"],
    "Avg Buy Price": [185.15, 24.02, 64.64, 237.91, 502.54, 332.98],
    "Shares": [3.0005, 1, 0.9997, 4.0016, 10.6181, 5.9919]
}

portfolio_df = pd.DataFrame(holdings)

# Fetch current prices
def fetch_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return price
    except Exception as e:
        return 0  # Default to 0 if fetching fails

portfolio_df["Current Price (USD)"] = portfolio_df["Ticker"].apply(fetch_current_price)
portfolio_df["Current Price (AUD)"] = portfolio_df["Current Price (USD)"] * exchange_rate

# Calculate P&L in AUD
portfolio_df["Total Invested (AUD)"] = portfolio_df["Avg Buy Price"] * portfolio_df["Shares"] * exchange_rate
portfolio_df["Current Value (AUD)"] = portfolio_df["Current Price (AUD)"] * portfolio_df["Shares"]
portfolio_df["P&L (AUD)"] = portfolio_df["Current Value (AUD)"] - portfolio_df["Total Invested (AUD)"]

# Portfolio Summary Calculations
total_invested = portfolio_df["Total Invested (AUD)"].sum()
total_value = portfolio_df["Current Value (AUD)"].sum()
total_pnl = portfolio_df["P&L (AUD)"].sum()
percentage_change = ((total_value - total_invested) / total_invested) * 100

# Dashboard Layout with Button Cards
st.markdown("### Dashboard Metrics")
col1, col2, col3 = st.columns(3)

def button_style(content, value, superscript=None):
    """
    Create a styled HTML button-like card for metrics.
    """
    superscript_html = f"<span style='color: {'green' if percentage_change > 0 else 'red'}; font-size: 0.8em; vertical-align: super;'>({superscript:.1f}%)</span>" if superscript is not None else ""
    return f"""
    <button style="
        background-color: #f8f9fa; 
        border: 2px solid #e7e7e7; 
        border-radius: 10px; 
        padding: 15px; 
        text-align: center; 
        font-size: 1.2em; 
        font-weight: bold; 
        cursor: pointer; 
        width: 100%; 
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
        {content}<br>${value:,.2f} {superscript_html}
    </button>
    """

with col1:
    if st.button("Total Invested Breakdown"):
        st.markdown("#### Total Invested Breakdown")
        st.table(
            portfolio_df[["Ticker", "Total Invested (AUD)"]]
            .reset_index(drop=True)  # Remove index column
            .style.format({"Total Invested (AUD)": "${:,.2f}"})
        )
    st.markdown(button_style("💰 Total Invested", total_invested), unsafe_allow_html=True)

with col2:
    if st.button("Current Value Breakdown"):
        st.markdown("#### Current Value Breakdown")
        st.table(
            portfolio_df[["Ticker", "Current Value (AUD)"]]
            .reset_index(drop=True)  # Remove index column
            .style.format({"Current Value (AUD)": "${:,.2f}"})
        )
    st.markdown(button_style("💵 Current Value", total_value, superscript=percentage_change), unsafe_allow_html=True)

with col3:
    if st.button("Total P&L Breakdown"):
        st.markdown("#### Total P&L Breakdown")
        st.table(
            portfolio_df[["Ticker", "P&L (AUD)"]]
            .reset_index(drop=True)  # Remove index column
            .style.format({"P&L (AUD)": "${:,.2f}"})
        )
    st.markdown(button_style("📈 Total P&L", total_pnl), unsafe_allow_html=True)

# Historical Data for Plots
st.subheader("📊 Portfolio Performance")
start_date = "2024-11-01"

def fetch_historical_data(ticker, start_date):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, interval="1h")  # Fetch hourly data
        history["Close (AUD)"] = history["Close"] * exchange_rate
        return history
    except Exception as e:
        return pd.DataFrame()

# Total Portfolio Performance
historical_values = pd.DataFrame()
for index, row in portfolio_df.iterrows():
    ticker_history = fetch_historical_data(row["Ticker"], start_date)
    if not ticker_history.empty:
        historical_values[row["Ticker"]] = ticker_history["Close (AUD)"] * row["Shares"]

# Sum up the total portfolio value
historical_values["Total Value"] = historical_values.sum(axis=1)

# Smooth the data using cubic spline interpolation
x_original = np.arange(len(historical_values.index))
y_original = historical_values["Total Value"].values
x_smooth = np.linspace(x_original.min(), x_original.max(), 1000)  # More points for smoother lines
spline = make_interp_spline(x_original, y_original)
y_smooth = spline(x_smooth)

# Plot 1: Total Portfolio Performance using Plotly
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pd.date_range(start=historical_values.index.min(), periods=len(x_smooth), freq="H"),
    y=y_smooth,
    mode="lines",
    line=dict(color="green", width=2),
    name="Total Portfolio Value",
    hovertemplate="<b>Date:</b> %{x}<br><b>Portfolio Value:</b> $%{y:,.2f}<extra></extra>"
))
fig1.update_layout(
    title="Total Portfolio Performance (Smoothed)",
    xaxis_title="Date",
    yaxis_title="Portfolio Value (AUD)",
    xaxis=dict(showline=True, showgrid=False, linecolor="black", tickangle=45),
    yaxis=dict(showline=True, showgrid=False, linecolor="black"),
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper
    hovermode="x unified",
)
st.plotly_chart(fig1)

# Individual Percentage Performance
percentage_change_df = pd.DataFrame()
for ticker in portfolio_df["Ticker"]:
    ticker_data = fetch_historical_data(ticker, start_date)
    if not ticker_data.empty:
        percentage_change_df[ticker] = (ticker_data["Close (AUD)"] / ticker_data["Close (AUD)"].iloc[0] - 1) * 100

# Smooth the percentage change using cubic spline interpolation
fig2 = go.Figure()
for ticker in percentage_change_df.columns:
    x_original = np.arange(len(percentage_change_df.index))
    y_original = percentage_change_df[ticker].values
    x_smooth = np.linspace(x_original.min(), x_original.max(), 1000)
    spline = make_interp_spline(x_original, y_original)
    y_smooth = spline(x_smooth)

    fig2.add_trace(go.Scatter(
        x=pd.date_range(start=percentage_change_df.index.min(), periods=len(x_smooth), freq="H"),
        y=y_smooth,
        mode="lines",
        line=dict(width=2),
        name=ticker,
        hovertemplate="<b>Date:</b> %{x}<br><b>Percentage Change:</b> %{y:.2f}%<extra></extra>"
    ))
fig2.update_layout(
    title="Individual Percentage Performance (Smoothed)",
    xaxis_title="Date",
    yaxis_title="Percentage Change (%)",
    xaxis=dict(showline=True, showgrid=False, linecolor="black", tickangle=45),
    yaxis=dict(showline=True, showgrid=False, linecolor="black"),
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper
    hovermode="x unified",
)
st.plotly_chart(fig2)
