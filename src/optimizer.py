import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.plotting import plot_efficient_frontier
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, price_data: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        :param price_data: DataFrame of asset prices (rows=dates, cols=assets)
        :param risk_free_rate: Annual risk-free rate
        """
        self.prices = price_data
        self.risk_free_rate = risk_free_rate
        
        # Calculate expected returns and sample covariance
        self.mu = expected_returns.mean_historical_return(price_data)
        self.S = risk_models.sample_cov(price_data)

    def optimize_max_sharpe(self):
        """
        Returns weights for maximum Sharpe ratio.
        """
        ef = EfficientFrontier(self.mu, self.S)
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        return cleaned_weights, performance

    def optimize_min_volatility(self):
        """
        Returns weights for minimum volatility.
        """
        ef = EfficientFrontier(self.mu, self.S)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        return cleaned_weights, performance

    def optimize_max_return(self):
        """
        Returns weights for maximum return (approximated by max risk).
        Since pure 'maximize return' is often 100% in one asset, valid only with constraints.
        Here we target the return of the highest-returning asset as a target on efficient frontier.
        """
        # Find max return of individual assets
        max_ind_ret = self.mu.max()
        
        ef = EfficientFrontier(self.mu, self.S)
        # Target a return slightly below max to ensure solution exists
        try:
            weights = ef.efficient_return(target_return=max_ind_ret * 0.99)
        except:
             # Fallback if solver struggles at the very edge 
            weights = ef.efficient_return(target_return=max_ind_ret * 0.95)
            
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        return cleaned_weights, performance

    def get_efficient_frontier_points(self, num_points=100):
        """
        Returns risk, return points for plotting the efficient frontier.
        """
        ef = EfficientFrontier(self.mu, self.S)
        
        # Calculate min volatility and max return
        min_vol_ef = EfficientFrontier(self.mu, self.S)
        min_vol_ef.min_volatility()
        min_ret = min_vol_ef.portfolio_performance()[0]
        
        max_ret = self.mu.max()
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret * 0.99, num_points)
        
        volatilities = []
        returns = []
        
        for r in target_returns:
            try:
                ef = EfficientFrontier(self.mu, self.S)
                ef.efficient_return(r)
                perf = ef.portfolio_performance()
                volatilities.append(perf[1])
                returns.append(perf[0])
            except:
                pass
                
        return volatilities, returns
