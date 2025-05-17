"""
Financial metrics calculation utilities for FINCON.
Implements metrics for evaluating trading performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def calculate_daily_returns(prices, positions):
    """
    Calculate daily returns based on prices and positions.
    
    Args:
        prices (pd.Series): Daily closing prices
        positions (pd.Series): Daily positions (-1 to 1)
        
    Returns:
        pd.Series: Daily returns
    """
    # Calculate price returns
    price_returns = prices.pct_change().fillna(0)
    
    # Shift positions to align with next day's returns
    aligned_positions = positions.shift(1).fillna(0)
    
    # Calculate strategy returns
    strategy_returns = price_returns * aligned_positions
    
    return strategy_returns


def calculate_cumulative_return(daily_returns):
    """
    Calculate cumulative return from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        float: Cumulative return as percentage
    """
    return ((1 + daily_returns).cumprod() - 1) * 100


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.01, trading_days=252):
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        daily_returns (pd.Series): Daily returns
        risk_free_rate (float): Annualized risk-free rate
        trading_days (int): Number of trading days in a year
        
    Returns:
        float: Annualized Sharpe ratio
    """
    # Convert annual risk-free rate to daily
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days) - 1
    
    # Calculate excess returns
    excess_returns = daily_returns - daily_risk_free
    
    # Calculate annualized Sharpe ratio
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)
    
    return sharpe_ratio


def calculate_max_drawdown(daily_returns):
    """
    Calculate maximum drawdown.
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        float: Maximum drawdown as percentage
    """
    # Calculate cumulative returns
    cum_returns = (1 + daily_returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns / running_max - 1) * 100
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown


def calculate_cvar(daily_returns, confidence_level=0.01):
    """
    Calculate Conditional Value at Risk (CVaR).
    
    Args:
        daily_returns (pd.Series): Daily returns
        confidence_level (float): Confidence level (default: 0.01 for 1%)
        
    Returns:
        float: CVaR value
    """
    # Sort returns in ascending order
    sorted_returns = sorted(daily_returns)
    
    # Calculate the index for VaR
    var_index = int(len(sorted_returns) * confidence_level)
    
    # Ensure at least one value
    var_index = max(1, var_index)
    
    # Calculate CVaR as the average of the worst var_index values
    cvar = np.mean(sorted_returns[:var_index])
    
    return cvar


def calculate_performance_metrics(daily_returns, risk_free_rate=0.01):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        daily_returns (pd.Series): Daily returns
        risk_free_rate (float): Annualized risk-free rate
        
    Returns:
        dict: Performance metrics
    """
    if daily_returns.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "cvar_1pct": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0
        }
    
    # Calculate basic metrics
    cumulative_return = calculate_cumulative_return(daily_returns).iloc[-1]
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(daily_returns)
    
    # Calculate annualized metrics
    annualized_return = (1 + daily_returns.mean()) ** 252 - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    
    # Calculate CVaR
    cvar_1pct = calculate_cvar(daily_returns, 0.01)
    
    # Calculate win rate
    win_rate = len(daily_returns[daily_returns > 0]) / len(daily_returns)
    
    # Calculate profit/loss ratio
    avg_profit = daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0
    avg_loss = abs(daily_returns[daily_returns < 0].mean()) if len(daily_returns[daily_returns < 0]) > 0 else 1
    profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else 0
    
    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "annualized_return": annualized_return * 100,  # Convert to percentage
        "annualized_volatility": annualized_volatility * 100,  # Convert to percentage
        "cvar_1pct": cvar_1pct * 100,  # Convert to percentage
        "win_rate": win_rate * 100,  # Convert to percentage
        "profit_loss_ratio": profit_loss_ratio
    }


def calculate_portfolio_metrics(weights, returns, cov_matrix):
    """
    Calculate portfolio metrics based on weights, returns, and covariance matrix.
    
    Args:
        weights (np.array): Portfolio weights
        returns (np.array): Expected returns for each asset
        cov_matrix (np.array): Covariance matrix of asset returns
        
    Returns:
        dict: Portfolio metrics
    """
    # Calculate portfolio expected return
    portfolio_return = np.sum(weights * returns)
    
    # Calculate portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
    
    return {
        "expected_return": portfolio_return * 100,  # Convert to percentage
        "volatility": portfolio_volatility * 100,  # Convert to percentage
        "sharpe_ratio": sharpe_ratio
    }


def evaluate_trading_performance(prices, positions):
    """
    Evaluate trading performance based on prices and positions.
    
    Args:
        prices (dict or pd.DataFrame): Historical prices for each symbol
        positions (dict or pd.DataFrame): Trading positions for each symbol
        
    Returns:
        dict: Performance evaluation results
    """
    # Convert inputs to DataFrames if they're dictionaries
    if isinstance(prices, dict):
        prices = pd.DataFrame(prices)
    if isinstance(positions, dict):
        positions = pd.DataFrame(positions)
    
    # Calculate daily returns for each symbol
    symbol_returns = {}
    for column in prices.columns:
        if column in positions.columns:
            symbol_returns[column] = calculate_daily_returns(prices[column], positions[column])
    
    # Combine returns into a DataFrame
    returns_df = pd.DataFrame(symbol_returns)
    
    # Calculate portfolio returns (assuming equal weighting if multiple symbols)
    portfolio_returns = returns_df.mean(axis=1)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns)
    
    # Calculate daily PnL
    daily_pnl = portfolio_returns * 100  # Convert to percentage
    
    # Calculate cumulative PnL
    cumulative_pnl = calculate_cumulative_return(portfolio_returns)
    
    return {
        "metrics": metrics,
        "daily_returns": portfolio_returns,
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "symbol_metrics": {
            symbol: calculate_performance_metrics(returns)
            for symbol, returns in symbol_returns.items()
        }
    }