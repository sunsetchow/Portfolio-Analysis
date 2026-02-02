import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, price_data: pd.DataFrame):
        self.prices = price_data
        self.returns = self.prices.pct_change().dropna()
        self.cumulative_returns = (1 + self.returns).cumprod()
    
    def plot_cumulative_returns(self, weights: dict, benchmark_ticker: str = None, save_path: str = None):
        """
        Plots the cumulative return of the portfolio vs benchmark.
        """
        plt.figure(figsize=(10, 6))
        
        # Calculate Portfolio Cumulative Return
        assets = [t for t in weights.keys() if t in self.returns.columns]
        weight_vec = np.array([weights[a] for a in assets])
        weight_vec = weight_vec / np.sum(weight_vec)
        
        portfolio_daily = self.returns[assets].dot(weight_vec)
        portfolio_cum = (1 + portfolio_daily).cumprod()
        
        plt.plot(portfolio_cum.index, portfolio_cum, label="Portfolio", linewidth=2)
        
        # Benchmark
        if benchmark_ticker and benchmark_ticker in self.cumulative_returns.columns:
            plt.plot(self.cumulative_returns.index, self.cumulative_returns[benchmark_ticker], label=benchmark_ticker, linestyle="--", alpha=0.7)
            
        plt.title("Cumulative Portfolio Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show() # Note: In agentic environment, show() might not work, usually better to save.
            
    def plot_correlation_heatmap(self, save_path: str = None):
        """
        Plots a correlation heatmap of the assets.
        """
        corr = self.returns.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Asset Correlation Matrix")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved heatmap to {save_path}")
        else:
            plt.show()

    def plot_drawdown(self, weights: dict, save_path: str = None):
        """
        Plots the underwater (drawdown) chart.
        """
        assets = [t for t in weights.keys() if t in self.returns.columns]
        weight_vec = np.array([weights[a] for a in assets])
        weight_vec = weight_vec / np.sum(weight_vec)
        
        portfolio_daily = self.returns[assets].dot(weight_vec)
        cum_ret = (1 + portfolio_daily).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        
        plt.figure(figsize=(10, 4))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        plt.title("Portfolio Drawdown")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved drawdown plot to {save_path}")
