import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

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

    def optimize_max_sharpe(self, fixed_weights=None):
        """
        Returns weights for maximum Sharpe ratio.
        :param fixed_weights: dict of {ticker: weight} to lock specific assets
        """
        ef = EfficientFrontier(self.mu, self.S)
        
        if fixed_weights:
            for ticker, weight in fixed_weights.items():
                if ticker in self.mu.index:
                    idx = self.mu.index.get_loc(ticker)
                    # Constraint: w[idx] == weight
                    ef.add_constraint(lambda w, i=idx, v=weight: w[i] == v)
                    
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        return cleaned_weights, performance

    def optimize_min_volatility(self, fixed_weights=None):
        """
        Returns weights for minimum volatility.
        :param fixed_weights: dict of {ticker: weight} to lock specific assets
        """
        ef = EfficientFrontier(self.mu, self.S)
        
        if fixed_weights:
            for ticker, weight in fixed_weights.items():
                if ticker in self.mu.index:
                    idx = self.mu.index.get_loc(ticker)
                    ef.add_constraint(lambda w, i=idx, v=weight: w[i] == v)

        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
        return cleaned_weights, performance

    def optimize_max_return(self, fixed_weights=None):
        """
        Returns weights for maximum return.
        With constraints: Allocates remaining capital to the highest-return unlocked asset.
        """
        ef = EfficientFrontier(self.mu, self.S)
        
        if fixed_weights:
            # Analytical solution for Max Return with constraints:
            # Fill fixed weights, then put ALL remaining weight into the highest return unlocked asset.
            
            # Start with 0 for all
            current_weights = {t: 0.0 for t in self.mu.index}
            current_weights.update(fixed_weights)
            
            used_weight = sum(fixed_weights.values())
            remaining_weight = 1.0 - used_weight
            
            if remaining_weight > 1e-6:
                # Find unlocked assets
                unlocked_assets = [t for t in self.mu.index if t not in fixed_weights]
                if unlocked_assets:
                    # Find the one with max return
                    best_asset = self.mu[unlocked_assets].idxmax()
                    current_weights[best_asset] += remaining_weight
            
            # Use PyPortfolioOpt to calculate metrics for this manually constructed portfolio
            ef.set_weights(current_weights)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)
            return cleaned_weights, performance

        # Unconstrained (existing logic)
        max_ind_ret = self.mu.max()
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
