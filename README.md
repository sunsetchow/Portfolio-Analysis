# Portfolio Analysis Tool

A Python-based interactive dashboard to analyze and optimize investment portfolios.

## Features
- **Historical Data**: Fetches adjusted close prices from Financial Modeling Prep (FMP).
- **Portfolio Metrics**: Calculates Annual Return, Volatility, Sharpe Ratio, and Max Drawdown.
- **Visualization**: Interactive charts for Cumulative Returns, Drawdown loops, and Correlation Heatmaps.
- **Optimization**:
    - **Maximize Sharpe Ratio**: Best risk-adjusted return.
    - **Minimize Volatility**: Safest portfolio.
    - **Maximize Return**: Target high return on the efficient frontier.

## Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## Configuration
The app uses an API Key for Financial Modeling Prep.
- By default, it looks for `st.secrets["FMP_API_KEY"]` or falls back to a provided default key.
- For local use, you can replace the key in `src/market_data.py` or set up `.streamlit/secrets.toml`.
