"""
Evaluation metrics for FINCON system.
Implements metrics for evaluating trading performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_cumulative_return(daily_returns):
    """
    Calculate cumulative return from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        pd.Series: Cumulative returns
    """
    return (1 + daily_returns).cumprod() - 1

def calculate_annualized_return(daily_returns):
    """
    Calculate annualized return from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        float: Annualized return
    """
    # Calculate total return
    total_return = (1 + daily_returns).prod() - 1
    
    # Annualize based on number of trading days
    num_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / num_days) - 1
    
    return annualized_return

def calculate_volatility(daily_returns, annualized=True):
    """
    Calculate volatility from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        annualized (bool): Whether to annualize volatility
        
    Returns:
        float: Volatility
    """
    volatility = daily_returns.std()
    
    if annualized:
        volatility *= np.sqrt(252)
        
    return volatility

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.01):
    """
    Calculate Sharpe ratio from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        risk_free_rate (float): Annualized risk-free rate
        
    Returns:
        float: Sharpe ratio
    """
    # Convert annual risk-free rate to daily
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    
    # Calculate excess returns
    excess_returns = daily_returns - daily_risk_free
    
    # Calculate Sharpe ratio
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return 0.0
        
    sharpe_ratio = mean_excess_return / std_excess_return * np.sqrt(252)
    
    return sharpe_ratio

def calculate_max_drawdown(daily_returns):
    """
    Calculate maximum drawdown from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        float: Maximum drawdown as percentage
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns / running_max - 1) * 100
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def calculate_cvar(daily_returns, confidence_level=0.01):
    """
    Calculate Conditional Value at Risk (CVaR) from daily returns.
    
    Args:
        daily_returns (pd.Series): Daily returns
        confidence_level (float): Confidence level
        
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

def calculate_win_rate(daily_returns):
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        float: Win rate as percentage
    """
    positive_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    
    if total_days == 0:
        return 0.0
        
    win_rate = positive_days / total_days * 100
    
    return win_rate

def calculate_profit_loss_ratio(daily_returns):
    """
    Calculate profit/loss ratio.
    
    Args:
        daily_returns (pd.Series): Daily returns
        
    Returns:
        float: Profit/loss ratio
    """
    # Separate positive and negative returns
    positive_returns = daily_returns[daily_returns > 0]
    negative_returns = daily_returns[daily_returns < 0]
    
    # Calculate average profit and loss
    avg_profit = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 1
    
    # Calculate profit/loss ratio
    if avg_loss == 0:
        return float('inf')
        
    profit_loss_ratio = avg_profit / avg_loss
    
    return profit_loss_ratio

def calculate_performance_metrics(daily_returns, risk_free_rate=0.01):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        daily_returns (pd.Series): Daily returns
        risk_free_rate (float): Annualized risk-free rate
        
    Returns:
        dict: Performance metrics
    """
    if len(daily_returns) == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "cvar_1pct": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0
        }
    
    metrics = {
        "cumulative_return": calculate_cumulative_return(daily_returns).iloc[-1] * 100,
        "annualized_return": calculate_annualized_return(daily_returns) * 100,
        "volatility": calculate_volatility(daily_returns) * 100,
        "sharpe_ratio": calculate_sharpe_ratio(daily_returns, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(daily_returns),
        "cvar_1pct": calculate_cvar(daily_returns) * 100,
        "win_rate": calculate_win_rate(daily_returns),
        "profit_loss_ratio": calculate_profit_loss_ratio(daily_returns)
    }
    
    return metrics

def evaluate_trading_strategy(prices, positions):
    """
    Evaluate trading strategy performance.
    
    Args:
        prices (pd.DataFrame): Historical prices
        positions (pd.DataFrame): Trading positions
        
    Returns:
        dict: Performance evaluation
    """
    # Calculate daily returns
    daily_returns = calculate_strategy_returns(prices, positions)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(daily_returns)
    
    # Calculate cumulative returns
    cumulative_returns = calculate_cumulative_return(daily_returns)
    
    return {
        "metrics": metrics,
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns
    }

def calculate_strategy_returns(prices, positions):
    """
    Calculate strategy returns based on prices and positions.
    
    Args:
        prices (pd.DataFrame): Historical prices
        positions (pd.DataFrame): Trading positions
        
    Returns:
        pd.Series: Strategy returns
    """
    # Calculate price returns
    price_returns = prices.pct_change()
    
    # Align positions with next day's returns
    aligned_positions = positions.shift(1)
    
    # Calculate strategy returns (element-wise multiplication)
    strategy_returns = price_returns * aligned_positions
    
    # For multi-asset strategies, take the mean across assets
    if isinstance(strategy_returns, pd.DataFrame) and strategy_returns.shape[1] > 1:
        strategy_returns = strategy_returns.mean(axis=1)
    
    # Drop NaNs
    strategy_returns = strategy_returns.dropna()
    
    return strategy_returns

def compare_strategies(strategy_returns_dict, benchmark_returns=None, risk_free_rate=0.01):
    """
    Compare multiple trading strategies.
    
    Args:
        strategy_returns_dict (dict): Dictionary of strategy returns
        benchmark_returns (pd.Series, optional): Benchmark returns
        risk_free_rate (float): Annualized risk-free rate
        
    Returns:
        pd.DataFrame: Comparison of strategies
    """
    # Initialize results dictionary
    results = {}
    
    # Calculate metrics for each strategy
    for name, returns in strategy_returns_dict.items():
        metrics = calculate_performance_metrics(returns, risk_free_rate)
        results[name] = metrics
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        benchmark_metrics = calculate_performance_metrics(benchmark_returns, risk_free_rate)
        results["Benchmark"] = benchmark_metrics
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results).T
    
    return comparison_df