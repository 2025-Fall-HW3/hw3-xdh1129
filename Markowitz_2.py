"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        if self.exclude not in self.price.columns:
            raise ValueError("Excluded asset not found in price data.")

        # Strategy hyper-parameters
        assets = assets.tolist() if isinstance(assets, pd.Index) else list(assets)
        defensive_assets = [a for a in ["XLP", "XLV", "XLU"] if a in assets]
        short_window = max(63, self.lookback)
        long_window = max(126, short_window)
        vol_window = max(20, self.lookback)
        trend_window = 200
        max_assets = max(3, min(5, len(assets)))

        # Pre-compute rolling statistics
        short_momentum = self.price[assets].pct_change(short_window)
        long_momentum = self.price[assets].pct_change(long_window)
        rolling_vol = (
            self.returns[assets].rolling(vol_window).std().replace(0, np.nan)
        )
        spy_ma = self.price[self.exclude].rolling(trend_window).mean()

        start_idx = max(short_window, long_window, vol_window, trend_window)
        for idx in range(start_idx, len(self.price)):
            date = self.price.index[idx]

            short_mom = short_momentum.iloc[idx]
            long_mom = long_momentum.iloc[idx]
            vol = rolling_vol.iloc[idx]

            momentum_signal = (0.6 * short_mom + 0.4 * long_mom).clip(lower=0).fillna(0)

            risk_score = momentum_signal / vol
            risk_score.replace([np.inf, -np.inf], 0, inplace=True)
            risk_score = risk_score.fillna(0)

            # Keep the strongest ideas only
            selected = risk_score.nlargest(max_assets)
            momentum_weights = pd.Series(0.0, index=assets)
            if selected.sum() > 0:
                momentum_weights[selected.index] = selected / selected.sum()

            defensive_weights = pd.Series(0.0, index=assets)
            if defensive_assets:
                defensive_weights[defensive_assets] = 1.0 / len(defensive_assets)
            else:
                defensive_weights.loc[:] = 1.0 / len(assets)

            spy_price_today = self.price[self.exclude].iloc[idx]
            spy_ma_today = spy_ma.iloc[idx]
            risk_on = pd.notna(spy_ma_today) and spy_price_today > spy_ma_today

            cash_buffer = 0.05 if risk_on else 0.1
            defensive_share = 0.1 if risk_on else 0.35
            momentum_share = max(0.0, 1.0 - (cash_buffer + defensive_share))

            # If there is no strong signal, lean entirely on defensives
            if momentum_weights.sum() == 0:
                defensive_share += momentum_share
                momentum_share = 0.0

            final_sector_weights = (
                momentum_share * momentum_weights + defensive_share * defensive_weights
            )

            self.portfolio_weights.loc[date, assets] = final_sector_weights.values
            self.portfolio_weights.loc[date, self.exclude] = 0.0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
