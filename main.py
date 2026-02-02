import sys
import os
import pandas as pd
from src.market_data import MarketData
from src.portfolio_analyzer import PortfolioAnalyzer
from src.visualizer import Visualizer

def main():
    # Configuration
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01"
    
    # Define a simple 60/40 portfolio
    # SPY: S&P 500
    # AGG: US Aggregate Bond
    # GLD: Gold (adding for diversification checks)
    tickers = ["SPY", "AGG", "GLD"]
    weights = {
        "SPY": 0.5,
        "AGG": 0.4,
        "GLD": 0.1
    }
    
    print("--- Starting Portfolio Analysis Tool ---")
    
    # 1. Fetch Data
    md = MarketData(START_DATE, END_DATE)
    prices = md.fetch_data(tickers)
    
    if prices.empty:
        print("No data fetched. Exiting.")
        return
        
    print(f"Data fetched successfully. Rows: {len(prices)}")
    
    # 2. Analyze
    analyzer = PortfolioAnalyzer(prices)
    metrics = analyzer.calculate_metrics(weights)
    
    print("\n--- Portfolio Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    corr = analyzer.get_correlation_matrix()
    print("\n--- Correlation Matrix ---")
    print(corr)
    
    # 3. Visualize
    viz = Visualizer(prices)
    
    # Create output directory for plots
    os.makedirs("output", exist_ok=True)
    
    viz.plot_cumulative_returns(weights, benchmark_ticker="SPY", save_path="output/cumulative_returns.png")
    viz.plot_correlation_heatmap(save_path="output/correlation_heatmap.png")
    viz.plot_drawdown(weights, save_path="output/drawdown.png")
    
    print("\nAnalysis complete. Plots saved to 'output/' directory.")

if __name__ == "__main__":
    main()
