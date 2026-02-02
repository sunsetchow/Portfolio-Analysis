import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.market_data import MarketData
from src.portfolio_analyzer import PortfolioAnalyzer
from src.optimizer import PortfolioOptimizer

# Page Config
st.set_page_config(page_title="Portfolio Analysis Tool", layout="wide")

# Title
st.title("📊 Portfolio Analysis Tool")

# Sidebar - Configuration
st.sidebar.header("Configuration")

# Date Range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))

# Portfolio Input
st.sidebar.subheader("Portfolio Assets")

# Default values
default_tickers = ["SPY", "AGG", "GLD"]
default_weights = [0.5, 0.4, 0.1]

if "portfolio" not in st.session_state:
    st.session_state.portfolio = [
        {"ticker": t, "weight": w} for t, w in zip(default_tickers, default_weights)
    ]

def add_asset():
    st.session_state.portfolio.append({"ticker": "AAPL", "weight": 0.0})

def remove_asset(index):
    st.session_state.portfolio.pop(index)

# Helper functions for state updates
def normalize_weights():
    total = sum(item["weight"] for item in st.session_state.portfolio)
    if total > 0:
        for i, item in enumerate(st.session_state.portfolio):
            new_weight = item["weight"] / total
            item["weight"] = new_weight
            st.session_state[f"weight_{i}"] = new_weight

def update_portfolio_from_state():
    # Syncs widget changes back to portfolio list
    # This might be needed if generic keys are used, but here we are using direct keys.
    pass

# Dynamic Input Fields
for i, asset in enumerate(st.session_state.portfolio):
    # Ensure keys exist in session state to avoid "value set via API" warning
    if f"ticker_{i}" not in st.session_state:
        st.session_state[f"ticker_{i}"] = asset["ticker"]
    if f"weight_{i}" not in st.session_state:
        st.session_state[f"weight_{i}"] = asset["weight"]

    cols = st.sidebar.columns([3, 2, 1])
    with cols[0]:
        # Widget handles state via 'key'
        st.text_input(f"Ticker {i+1}", key=f"ticker_{i}")
        # Read updated value from state
        asset["ticker"] = st.session_state[f"ticker_{i}"].upper()
    with cols[1]:
        # Widget handles state via 'key'
        st.number_input(f"Weight", key=f"weight_{i}", step=0.05)
        # Read updated value from state
        asset["weight"] = st.session_state[f"weight_{i}"]
    with cols[2]:
        # Spacer to align with inputs that have labels
        st.write("") 
        st.write("")
        if st.button("❌", key=f"del_{i}"):
            remove_asset(i)
            st.rerun()

if st.sidebar.button("➕ Add Asset"):
    add_asset()
    st.rerun()

# Normalize Weights Button
st.sidebar.button("Normalize Weights to 100%", on_click=normalize_weights)

# Processing
tickers = [item["ticker"] for item in st.session_state.portfolio]
weights = {item["ticker"]: item["weight"] for item in st.session_state.portfolio}

total_weight = sum(weights.values())
st.sidebar.markdown(f"**Total Allocation:** {total_weight:.1%}")

if total_weight == 0:
    st.error("Total allocation is 0%. Please add weights.")
    st.stop()

if st.sidebar.button("Run Analysis", type="primary"):
    st.session_state.run_analysis = True

# Caching data fetch to avoid API calls on every rerun/interaction
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(start, end, tickers):
    md = MarketData(start, end)
    return md.fetch_data(tickers)

if st.session_state.get("run_analysis", False):
    with st.spinner("Fetching data and calculating metrics..."):
        # Fetch Data
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        prices = get_market_data(start_str, end_str, tickers)
        
        if prices.empty:
            st.error("Failed to fetch data. Please check ticker symbols.")
        else:
            # Analyze
            analyzer = PortfolioAnalyzer(prices)
            metrics = analyzer.calculate_metrics(weights)
            corr = analyzer.get_correlation_matrix()
            
            # --- Results Display ---
            
            # 1. Metrics Scorecard
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Annual Return", f"{metrics['Annual Return']:.2%}")
            col2.metric("Volatility", f"{metrics['Annual Volatility']:.2%}")
            col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            col4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            
            # 2. Tabs for Charts
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📉 Drawdown", "🔥 Correlation", "⚙️ Optimization"])
            
            with tab1:
                st.subheader("Cumulative Returns")
                # Calculate cumulative return series manually for plotting
                returns = prices.pct_change().dropna()
                weight_vec = [weights[t] for t in returns.columns] # alignment
                portfolio_ret = returns.dot(weight_vec)
                portfolio_cum = (1 + portfolio_ret).cumprod()
                
                # Plotly Chart
                fig = px.line(portfolio_cum, x=portfolio_cum.index, y=portfolio_cum.values, labels={'x': 'Date', 'y': 'Growth of $1'})
                fig.add_trace(go.Scatter(x=portfolio_cum.index, y=portfolio_cum.values, mode='lines', name='Portfolio'))
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                st.subheader("Underwater Plot")
                peak = portfolio_cum.cummax()
                drawdown = (portfolio_cum - peak) / peak
                
                fig_dd = px.area(drawdown, x=drawdown.index, y=drawdown.values, title="Drawdown Over Time")
                fig_dd.update_traces(fillcolor='red', line_color='red')
                st.plotly_chart(fig_dd, use_container_width=True)
                
            with tab3:
                st.subheader("Correlation Matrix")
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
                st.plotly_chart(fig_corr, use_container_width=True)

            with tab4:
                st.subheader("Mean-Variance Optimization")
                
                optimizer = PortfolioOptimizer(prices)
                
                col_opt1, col_opt2 = st.columns(2)
                
                opt_weights = None
                
                with col_opt1:
                    if st.button("🚀 Maximize Sharpe Ratio"):
                        try:
                            opt_weights, perf = optimizer.optimize_max_sharpe()
                            st.success(f"Optimized! Sharpe: {perf[2]:.2f}")
                            st.metric("Expected Return", f"{perf[0]:.2%}")
                            st.metric("Expected Volatility", f"{perf[1]:.2%}")
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")

                with col_opt2:
                    if st.button("🛡️ Minimize Volatility"):
                        try:
                            opt_weights, perf = optimizer.optimize_min_volatility()
                            st.success(f"Optimized! Volatility: {perf[1]:.2%}")
                            st.metric("Expected Return", f"{perf[0]:.2%}")
                            st.metric("Expected Volatility", f"{perf[1]:.2%}")
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")

                if st.button("📈 Maximize Return (Target High Risk)"):
                    try:
                        opt_weights, perf = optimizer.optimize_max_return()
                        st.success(f"Optimized! Return: {perf[0]:.2%}")
                        st.metric("Expected Return", f"{perf[0]:.2%}")
                        st.metric("Expected Volatility", f"{perf[1]:.2%}")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                            
                if opt_weights:
                    st.markdown("### Optimized Allocation")
                    # Display as table
                    df_opt = pd.DataFrame.from_dict(opt_weights, orient='index', columns=['Target Weight'])
                    df_opt['Current Weight'] = pd.Series(weights)
                    df_opt['Difference'] = df_opt['Target Weight'] - df_opt['Current Weight']
                    st.dataframe(df_opt.style.format("{:.1%}"))
                    
                    # Store optimized weights in session state temporarily to apply them
                    st.session_state.optimized_weights = opt_weights
                    
                if "optimized_weights" in st.session_state and st.button("Apply Optimized Weights"):
                    for ticker, w in st.session_state.optimized_weights.items():
                         # Find index of ticker in portfolio list
                         for i, item in enumerate(st.session_state.portfolio):
                             if item["ticker"] == ticker:
                                 # Update session state keys directly
                                 st.session_state[f"weight_{i}"] = w
                                 # Update list
                                 item["weight"] = w
                    st.success("Weights updated! Go to sidebar to verify.")
                    st.rerun()
