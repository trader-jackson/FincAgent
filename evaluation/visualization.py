"""
Visualization utilities for FINCON system.
Functions for visualizing trading performance and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_cumulative_returns(returns_dict, title=None, figsize=(12, 6), save_path=None):
    """
    Plot cumulative returns for multiple strategies.
    
    Args:
        returns_dict (dict): Dictionary of strategy returns series
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Calculate and plot cumulative returns for each strategy
    for name, returns in returns_dict.items():
        cumulative_returns = (1 + returns).cumprod() - 1
        plt.plot(cumulative_returns.index, cumulative_returns * 100, label=name)
    
    # Add title and labels
    plt.title(title or "Cumulative Returns Comparison", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_drawdowns(returns, title=None, figsize=(12, 6), save_path=None):
    """
    Plot drawdowns for a trading strategy.
    
    Args:
        returns (pd.Series): Strategy returns
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdowns
    drawdowns = (cumulative_returns / running_max - 1) * 100
    
    # Plot drawdowns
    plt.plot(drawdowns.index, drawdowns, color='red')
    
    # Add title and labels
    plt.title(title or "Drawdowns Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Drawdown (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_rolling_sharpe(returns, window=30, title=None, figsize=(12, 6), save_path=None):
    """
    Plot rolling Sharpe ratio.
    
    Args:
        returns (pd.Series): Strategy returns
        window (int): Rolling window size
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Calculate rolling mean and standard deviation
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Calculate rolling Sharpe ratio (annualized)
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    
    # Plot rolling Sharpe ratio
    plt.plot(rolling_sharpe.index, rolling_sharpe, color='blue')
    
    # Add title and labels
    plt.title(title or f"{window}-Day Rolling Sharpe Ratio", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_monthly_returns(returns, title=None, figsize=(12, 8), save_path=None):
    """
    Plot monthly returns heatmap.
    
    Args:
        returns (pd.Series): Strategy returns
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Convert daily returns to monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create DataFrame with year and month
    monthly_returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    # Pivot to create heatmap data
    heatmap_data = monthly_returns_df.pivot(index='year', columns='month', values='return')
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]
    
    # Create heatmap
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap=cmap, center=0, linewidths=.5, cbar_kws={"label": "Return (%)"})
    
    # Add title
    plt.title(title or "Monthly Returns (%)", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_trading_positions(positions, prices=None, title=None, figsize=(12, 8), save_path=None):
    """
    Plot trading positions over time.
    
    Args:
        positions (pd.DataFrame): Trading positions
        prices (pd.DataFrame, optional): Price data
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create subplot layout
    n_assets = positions.shape[1] if isinstance(positions, pd.DataFrame) else 1
    fig, axes = plt.subplots(n_assets, 1, figsize=figsize, sharex=True)
    
    # Ensure axes is always a list
    if n_assets == 1:
        axes = [axes]
    
    # Plot positions for each asset
    for i, asset in enumerate(positions.columns if isinstance(positions, pd.DataFrame) else [positions.name]):
        ax = axes[i]
        
        # Get position data for current asset
        if isinstance(positions, pd.DataFrame):
            asset_positions = positions[asset]
        else:
            asset_positions = positions
        
        # Plot positions
        ax.fill_between(asset_positions.index, 0, asset_positions, where=asset_positions > 0, color='green', alpha=0.3, label='Long')
        ax.fill_between(asset_positions.index, 0, asset_positions, where=asset_positions < 0, color='red', alpha=0.3, label='Short')
        
        # Plot price if available
        if prices is not None:
            # Get price data for current asset
            if isinstance(prices, pd.DataFrame) and asset in prices.columns:
                asset_prices = prices[asset]
            elif isinstance(prices, pd.Series) and prices.name == asset:
                asset_prices = prices
            else:
                asset_prices = None
                
            if asset_prices is not None:
                # Normalize prices to fit on the same plot
                normalized_prices = asset_prices / asset_prices.iloc[0]
                
                # Create secondary y-axis for prices
                ax2 = ax.twinx()
                ax2.plot(normalized_prices.index, normalized_prices, color='blue', alpha=0.5, label='Price')
                ax2.set_ylabel('Normalized Price', color='blue')
                
                # Add price legend
                lines2, labels2 = ax2.get_legend_handles_labels()
                
        # Add asset name to title
        ax.set_title(f"{asset} - {'Price and ' if prices is not None else ''}Positions")
        ax.set_ylabel('Position Size')
        ax.grid(True, alpha=0.3)
        
        # Add position legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if prices is not None and asset_prices is not None:
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')
    
    # Set common title and x-axis label
    fig.suptitle(title or "Trading Positions Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation_matrix(correlation_matrix, title=None, figsize=(10, 8), save_path=None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=.5, cbar_kws={"label": "Correlation"})
    
    # Add title
    plt.title(title or "Correlation Matrix", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_performance_comparison(metrics_df, figsize=(14, 8), save_path=None):
    """
    Plot performance metrics comparison.
    
    Args:
        metrics_df (pd.DataFrame): Performance metrics for multiple strategies
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot cumulative return
    metrics_df['cumulative_return'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Cumulative Return (%)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot Sharpe ratio
    metrics_df['sharpe_ratio'].plot(kind='bar', ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Sharpe Ratio', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot maximum drawdown
    metrics_df['max_drawdown'].plot(kind='bar', ax=axes[1, 0], color='red')
    axes[1, 0].set_title('Maximum Drawdown (%)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot volatility
    metrics_df['volatility'].plot(kind='bar', ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Annualized Volatility (%)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Set common title
    fig.suptitle('Performance Metrics Comparison', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_performance_report(returns, positions, prices=None, output_dir=None):
    """
    Create comprehensive performance report with multiple visualizations.
    
    Args:
        returns (pd.Series): Strategy returns
        positions (pd.DataFrame): Trading positions
        prices (pd.DataFrame, optional): Price data
        output_dir (str, optional): Directory to save report figures
        
    Returns:
        dict: Paths to generated figures
    """
    # Create output directory if not exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Dictionary to store figure paths
    figure_paths = {}
    
    # Plot cumulative returns
    fig_returns = plot_cumulative_returns(
        {'Strategy': returns}, 
        title="Cumulative Returns", 
        save_path=os.path.join(output_dir, f"cumulative_returns_{timestamp}.png") if output_dir else None
    )
    if output_dir:
        figure_paths['cumulative_returns'] = os.path.join(output_dir, f"cumulative_returns_{timestamp}.png")
    
    # Plot drawdowns
    fig_drawdowns = plot_drawdowns(
        returns, 
        title="Drawdowns", 
        save_path=os.path.join(output_dir, f"drawdowns_{timestamp}.png") if output_dir else None
    )
    if output_dir:
        figure_paths['drawdowns'] = os.path.join(output_dir, f"drawdowns_{timestamp}.png")
    
    # Plot rolling Sharpe ratio
    fig_sharpe = plot_rolling_sharpe(
        returns, 
        window=30, 
        title="30-Day Rolling Sharpe Ratio", 
        save_path=os.path.join(output_dir, f"rolling_sharpe_{timestamp}.png") if output_dir else None
    )
    if output_dir:
        figure_paths['rolling_sharpe'] = os.path.join(output_dir, f"rolling_sharpe_{timestamp}.png")
    
    # Plot monthly returns
    try:
        fig_monthly = plot_monthly_returns(
            returns, 
            title="Monthly Returns (%)", 
            save_path=os.path.join(output_dir, f"monthly_returns_{timestamp}.png") if output_dir else None
        )
        if output_dir:
            figure_paths['monthly_returns'] = os.path.join(output_dir, f"monthly_returns_{timestamp}.png")
    except:
        # Skip if not enough data for monthly resampling
        pass
    
    # Plot trading positions
    fig_positions = plot_trading_positions(
        positions, 
        prices=prices, 
        title="Trading Positions", 
        save_path=os.path.join(output_dir, f"positions_{timestamp}.png") if output_dir else None
    )
    if output_dir:
        figure_paths['positions'] = os.path.join(output_dir, f"positions_{timestamp}.png")
    
    # Close all figures
    plt.close('all')
    
    return figure_paths