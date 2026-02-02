import pandas as pd
import numpy as np

class PortfolioAnalyzer:
    def __init__(self, price_data: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        :param price_data: DataFrame of asset prices (rows=dates, cols=assets)
        :param risk_free_rate: Annual risk-free rate (decimal, e.g., 0.04 for 4%)
        """
        self.prices = price_data
        self.risk_free_rate = risk_free_rate
        # Calculate daily returns
        self.returns = self.prices.pct_change().dropna()
        
    def calculate_metrics(self, weights: dict) -> dict:
        """
        Calculate portfolio performance metrics for a given set of weights.
        :param weights: Dictionary {ticker: weight}, e.g., {'SPY': 0.6, 'AGG': 0.4}
        """
        # Align weights with available data
        assets = [t for t in weights.keys() if t in self.returns.columns]
        weight_vec = np.array([weights[a] for a in assets])
        
        # Normalize weights if they don't sum to 1 (optional, strictly speaking they should)
        if np.sum(weight_vec) == 0:
            return {}
        weight_vec = weight_vec / np.sum(weight_vec)
        
        # Subset returns
        portfolio_returns_daily = self.returns[assets].dot(weight_vec)
        
        # Annualized metrics (assuming 252 trading days)
        annual_return = portfolio_returns_daily.mean() * 252
        annual_volatility = portfolio_returns_daily.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Max Drawdown
        cumulative_returns = (1 + portfolio_returns_daily).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # VaR 95%
        var_95 = np.percentile(portfolio_returns_daily, 5)
        
        return {
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "VaR 95% (Daily)": var_95
        }

    def get_correlation_matrix(self):
        return self.returns.corr()

if __name__ == "__main__":
    # Mock data for testing
    dates = pd.date_range("2020-01-01", periods=100)
    data = pd.DataFrame({
        "SPY": np.random.normal(100, 1, 100),
        "AGG": np.random.normal(100, 0.5, 100)
    }, index=dates)
    
    analyzer = PortfolioAnalyzer(data)
    metrics = analyzer.calculate_metrics({"SPY": 0.6, "AGG": 0.4})
    print(metrics)
