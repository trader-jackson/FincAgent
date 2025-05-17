"""
Main script for FINCON system.
Implements a synthesized LLM multi-agent system with conceptual verbal reinforcement
for enhanced financial decision making.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import FINCON components
from agents.manager_agent import ManagerAgent
from agents.analyst_agents.news_analyst import NewsAnalyst
from agents.analyst_agents.filing_analyst import FilingAnalyst
from agents.analyst_agents.ecc_analyst import ECCAnalyst
from agents.analyst_agents.data_analyst import DataAnalyst
from agents.analyst_agents.stock_selection_agent import StockSelectionAgent
from agents.risk_control_agent import RiskControlAgent
from risk_control.within_episode import WithinEpisodeRiskControl
from risk_control.over_episode import OverEpisodeRiskControl
from utils.data_utils import load_stock_data, load_news_data, load_filings_data, load_ecc_data
from utils.financial_metrics import evaluate_trading_performance
from utils.llm_utils import LLMClient
from evaluation.metrics import calculate_performance_metrics
from evaluation.visualization import plot_cumulative_returns, plot_trading_positions, create_performance_report

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fincon.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("fincon_main")

class FINCON:
    """
    FINCON system main class.
    
    Orchestrates the interaction between manager agent, analyst agents,
    and risk control components for financial decision making.
    """
    
    def __init__(self, 
                 target_symbols, 
                 task_type="single_stock_trading",
                 training_start=None,
                 training_end=None,
                 testing_start=None,
                 testing_end=None,
                 llm_model=None,
                 llm_api_key=None,
                 max_episodes=None,
                 results_dir=None):
        """
        Initialize FINCON system.
        
        Args:
            target_symbols (list): List of stock symbols to trade
            task_type (str): Type of task ("single_stock_trading" or "portfolio_management")
            training_start (str): Training start date (YYYY-MM-DD)
            training_end (str): Training end date (YYYY-MM-DD)
            testing_start (str): Testing start date (YYYY-MM-DD)
            testing_end (str): Testing end date (YYYY-MM-DD)
            llm_model (str): LLM model name
            llm_api_key (str): LLM API key
            max_episodes (int): Maximum number of training episodes
            results_dir (str): Directory to save results
        """
        self.target_symbols = target_symbols
        self.task_type = task_type
        
        # Set dates
        self.training_start = training_start or config.TRAINING_START_DATE
        self.training_end = training_end or config.TRAINING_END_DATE
        self.testing_start = testing_start or config.TESTING_START_DATE
        self.testing_end = testing_end or config.TESTING_END_DATE
        
        # Training settings
        self.max_episodes = max_episodes or config.MAX_EPISODES
        
        # Results directory
        self.results_dir = results_dir or config.RESULTS_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model_name=llm_model or config.LLM_MODEL,
            api_key=llm_api_key or config.LLM_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        # Initialize risk control components
        self.within_episode_risk_control = WithinEpisodeRiskControl(
            confidence_level=config.CVAR_CONFIDENCE_LEVEL,
            threshold_decline=config.CVAR_THRESHOLD_DECLINE
        )
        
        # Initialize agents
        self._init_agents()
        
        # Data storage
        self.market_data = {}
        self.news_data = []
        self.filings_data = []
        self.ecc_data = []
        
        # Training and testing results
        self.training_results = []
        self.testing_results = {}
        self.current_portfolio = {symbol: 0.0 for symbol in target_symbols}
        
        logger.info(f"Initialized FINCON system with task type: {task_type}")
        logger.info(f"Target symbols: {', '.join(target_symbols)}")
        
    def _init_agents(self):
        """Initialize all agents in the system."""
        # Initialize within-episode risk control
        risk_control = self.within_episode_risk_control
        
        # Initialize analysts
        self.news_analyst = NewsAnalyst("news_agent", self.target_symbols)
        self.filing_analyst = FilingAnalyst("filing_agent", self.target_symbols)
        self.ecc_analyst = ECCAnalyst("ecc_agent", self.target_symbols)
        self.data_analyst = DataAnalyst("data_agent", self.target_symbols)
        self.stock_selection_agent = StockSelectionAgent("stock_selection_agent", self.target_symbols)
        
        # Initialize manager
        self.manager = ManagerAgent("manager_agent", self.target_symbols, risk_control)
        
        # Initialize risk control agent
        self.risk_control_agent = RiskControlAgent("risk_control_agent", self.target_symbols)
        
        # Set task type for manager
        self.manager.task_type = self.task_type
        
    def load_data(self, start_date, end_date, use_cache=True):
        """
        Load data for the system.
        
        Args:
            start_date (str): Start date for data loading
            end_date (str): End date for data loading
            use_cache (bool): Whether to use cached data
            
        Returns:
            dict: Loaded data
        """
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        # Load stock price data
        self.market_data = load_stock_data(
            self.target_symbols, start_date, end_date, 
            data_dir=os.path.join(config.DATA_DIR, "stock_data"),
            use_cache=use_cache
        )
        
        # Load news data
        self.news_data = load_news_data(
            self.target_symbols, start_date, end_date,
            data_dir=os.path.join(config.DATA_DIR, "news_data"),
            use_cache=use_cache
        )
        
        # Load filings data
        self.filings_data = load_filings_data(
            self.target_symbols, start_date, end_date,
            data_dir=os.path.join(config.DATA_DIR, "filings"),
            use_cache=use_cache
        )
        
        # Load ECC data
        self.ecc_data = load_ecc_data(
            self.target_symbols, start_date, end_date,
            data_dir=os.path.join(config.DATA_DIR, "ecc_audio"),
            use_cache=use_cache
        )
        
        logger.info(f"Loaded {len(self.market_data)} stock prices, {len(self.news_data)} news articles, "
                   f"{len(self.filings_data)} SEC filings, and {len(self.ecc_data)} earnings calls")
                   
        return {
            "market_data": self.market_data,
            "news_data": self.news_data,
            "filings_data": self.filings_data,
            "ecc_data": self.ecc_data
        }
        
    def run_episode(self, start_date, end_date, is_training=True):
        """
        Run a single episode.
        
        Args:
            start_date (str): Start date for the episode
            end_date (str): End date for the episode
            is_training (bool): Whether this is a training episode
            
        Returns:
            dict: Episode results
        """
        logger.info(f"Running {'training' if is_training else 'testing'} episode from {start_date} to {end_date}")
        
        # Reset agents for new episode
        self.manager.reset()
        self.within_episode_risk_control.reset()
        
        # Storage for episode data
        episode_id = f"episode_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        actions = []
        daily_pnl = []
        portfolio_values = []
        positions = {}
        risk_alerts = []
        
        # Create date range for the episode
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = [start_date_obj + timedelta(days=x) for x in range((end_date_obj - start_date_obj).days + 1)]
        trading_dates = [d.strftime("%Y-%m-%d") for d in date_range if d.weekday() < 5]  # Exclude weekends
        
        # Filter data for the date range
        market_data_for_episode = {}
        for symbol, data in self.market_data.items():
            mask = (data.index >= start_date) & (data.index <= end_date)
            market_data_for_episode[symbol] = data[mask]
        
        news_data_for_episode = [news for news in self.news_data if start_date <= news["date"] <= end_date]
        filings_data_for_episode = [filing for filing in self.filings_data if start_date <= filing["date_filed"] <= end_date]
        ecc_data_for_episode = [call for call in self.ecc_data if start_date <= call["date"] <= end_date]
        
        # Initialize positions storage
        for symbol in self.target_symbols:
            positions[symbol] = pd.Series(0.0, index=market_data_for_episode[symbol].index)
        
        # Initialize portfolio selection for portfolio management
        portfolio_stocks = self.target_symbols
        if self.task_type == "portfolio_management" and len(self.target_symbols) > 3:
            # Select a subset of stocks for portfolio
            portfolio_selection = self.stock_selection_agent.process(
                market_data_for_episode, 
                num_stocks=min(3, len(self.target_symbols))
            )
            portfolio_stocks = portfolio_selection["selected_stocks"]
            
            logger.info(f"Selected portfolio stocks: {', '.join(portfolio_stocks)}")
        
        # Run the episode day by day
        for date in trading_dates:
            # Get data for the current date
            current_date_data = self._get_data_for_date(date, market_data_for_episode, news_data_for_episode, 
                                                      filings_data_for_episode, ecc_data_for_episode)
            
            # Skip days with missing data
            if not current_date_data["market_data"]:
                continue
                
            # Process data with analyst agents
            analyst_insights = self._run_analyst_agents(current_date_data)
            
            # Check for risk alerts
            risk_alert = None
            if len(daily_pnl) > 0:
                last_pnl = daily_pnl[-1]
                cvar = self.within_episode_risk_control.update_pnl(last_pnl)
                risk_alert = self.within_episode_risk_control.check_risk_alert()
                
                if risk_alert:
                    risk_alerts.append(risk_alert)
                    logger.warning(f"Risk alert triggered on {date}: {risk_alert['level']}")
            
            # Make trading decisions with manager agent
            observations = {**analyst_insights}
            if risk_alert:
                observations["risk_alert"] = risk_alert
                
            trading_decision = self.manager.process(observations)
            
            # Update positions based on trading decision
            if self.task_type == "single_stock_trading":
                symbol = self.target_symbols[0]
                position_size = trading_decision["position_size"]
                self.current_portfolio[symbol] = position_size
                
                # Record position for this date
                date_idx = current_date_data["market_data"][symbol].index
                if not date_idx.empty:
                    positions[symbol].loc[date_idx[0]] = position_size
                
            else:  # Portfolio management
                for symbol, weight in trading_decision["weights"].items():
                    self.current_portfolio[symbol] = weight
                    
                    # Record position for this date
                    if symbol in current_date_data["market_data"]:
                        date_idx = current_date_data["market_data"][symbol].index
                        if not date_idx.empty:
                            positions[symbol].loc[date_idx[0]] = weight
            
            # Store trading action
            action = {
                "timestamp": date,
                "decision": trading_decision,
                "portfolio": self.current_portfolio.copy(),
                "risk_status": self.manager.risk_status
            }
            actions.append(action)
            
            # Calculate PnL for the day
            day_pnl = self._calculate_daily_pnl(date, positions, market_data_for_episode)
            daily_pnl.append(day_pnl)
            
            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(date, positions, market_data_for_episode)
            portfolio_values.append(portfolio_value)
            
            # Perform self-reflection if PnL is negative or risk alert is triggered
            if day_pnl < 0 or risk_alert:
                reflection = self.manager.self_reflect(day_pnl)
                logger.info(f"Self-reflection on {date}: {reflection[:100]}...")
                
            # Send feedback to analysts based on performance
            feedback = self.manager.send_feedback_to_analysts(day_pnl)
            for analyst_id, analyst_feedback in feedback.items():
                if analyst_id == "news_analyst":
                    self.news_analyst.receive_feedback(analyst_feedback)
                elif analyst_id == "filing_analyst":
                    self.filing_analyst.receive_feedback(analyst_feedback)
                elif analyst_id == "ecc_analyst":
                    self.ecc_analyst.receive_feedback(analyst_feedback)
                elif analyst_id == "data_analyst":
                    self.data_analyst.receive_feedback(analyst_feedback)
        
        # Calculate overall performance metrics
        metrics = self._calculate_episode_metrics(daily_pnl, positions, market_data_for_episode)
        
        # Create episode result
        episode_result = {
            "id": episode_id,
            "type": "training" if is_training else "testing",
            "start_date": start_date,
            "end_date": end_date,
            "actions": actions,
            "daily_pnl": daily_pnl,
            "portfolio_values": portfolio_values,
            "risk_alerts": risk_alerts,
            "metrics": metrics,
            "positions": {symbol: positions[symbol].to_dict() for symbol in positions}
        }
        
        # If it's a training episode, store in episodic memory
        if is_training:
            self.manager.episodic_memory.add_episode(
                episode_id, start_date, end_date, actions, daily_pnl, metrics
            )
            self.training_results.append(episode_result)
        else:
            self.testing_results = episode_result
        
        logger.info(f"Episode completed with cumulative return: {metrics['cumulative_return']:.2f}% "
                   f"and Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                   
        return episode_result
        
    def train(self):
        """
        Train the FINCON system for multiple episodes.
        
        Returns:
            dict: Training results
        """
        logger.info(f"Starting training with {self.max_episodes} episodes")
        
        # Load training data
        self.load_data(self.training_start, self.training_end)
        
        # Run episodes
        for i in range(self.max_episodes):
            logger.info(f"Starting training episode {i+1}/{self.max_episodes}")
            
            # Run episode
            episode = self.run_episode(self.training_start, self.training_end, is_training=True)
            
            # Skip belief update for first episode
            if i == 0:
                continue
                
            # Compare with previous episode and update beliefs
            prev_episode = self.training_results[i-1]
            
            # Perform episode comparison with risk control agent
            comparison = self.risk_control_agent.process_episode_comparison(
                episode, prev_episode
            )
            
            # Update manager's investment beliefs
            self.manager.update_investment_beliefs(comparison)
            
            # Propagate belief updates to analyst agents
            propagation_map = comparison.get("propagation_map", {})
            for agent_type, beliefs in propagation_map.items():
                if agent_type == "news_analyst":
                    self.news_analyst.update_beliefs(beliefs)
                elif agent_type == "filing_analyst":
                    self.filing_analyst.update_beliefs(beliefs)
                elif agent_type == "ecc_analyst":
                    self.ecc_analyst.update_beliefs(beliefs)
                elif agent_type == "data_analyst":
                    self.data_analyst.update_beliefs(beliefs)
                elif agent_type == "stock_selection_agent":
                    self.stock_selection_agent.update_beliefs(beliefs)
            
            # Log results
            logger.info(f"Episode {i+1} completed with return: {episode['metrics']['cumulative_return']:.2f}% " +
                       f"and Sharpe ratio: {episode['metrics']['sharpe_ratio']:.2f}")
            
            # Save intermediate results
            self._save_training_results()
        
        logger.info("Training completed")
        
        # Get best episode by Sharpe ratio
        best_episode = max(self.training_results, key=lambda x: x["metrics"]["sharpe_ratio"])
        logger.info(f"Best training episode: {best_episode['id']} with " +
                   f"Sharpe ratio: {best_episode['metrics']['sharpe_ratio']:.2f}")
        
        return {
            "episodes": self.training_results,
            "best_episode": best_episode
        }
        
    def test(self):
        """
        Test the FINCON system after training.
        
        Returns:
            dict: Testing results
        """
        logger.info("Starting testing")
        
        # Load testing data
        self.load_data(self.testing_start, self.testing_end)
        
        # Run testing episode
        testing_results = self.run_episode(self.testing_start, self.testing_end, is_training=False)
        
        # Log results
        logger.info(f"Testing completed with return: {testing_results['metrics']['cumulative_return']:.2f}% " +
                   f"and Sharpe ratio: {testing_results['metrics']['sharpe_ratio']:.2f}")
        
        # Save testing results
        self._save_testing_results()
        
        return testing_results
        
    def _run_analyst_agents(self, data):
        """
        Run all analyst agents on the given data.
        
        Args:
            data (dict): Data for the current date
            
        Returns:
            dict: Insights from analyst agents
        """
        insights = {}
        
        # Process market data with data analyst
        if data["market_data"]:
            data_analysis = self.data_analyst.process(data["market_data"])
            insights["data_analyst"] = data_analysis
        
        # Process news with news analyst
        if data["news_data"]:
            news_analysis = self.news_analyst.process(data["news_data"])
            insights["news_analyst"] = news_analysis
        
        # Process filings with filing analyst
        if data["filings_data"]:
            filing_analysis = self.filing_analyst.process(data["filings_data"])
            insights["filing_analyst"] = filing_analysis
        
        # Process earnings calls with ECC analyst
        if data["ecc_data"]:
            ecc_analysis = self.ecc_analyst.process(data["ecc_data"])
            insights["ecc_analyst"] = ecc_analysis
        
        return insights
        
    def _get_data_for_date(self, date, market_data, news_data, filings_data, ecc_data):
        """
        Get data for a specific date.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            market_data (dict): Market data dictionary
            news_data (list): News data list
            filings_data (list): Filings data list
            ecc_data (list): ECC data list
            
        Returns:
            dict: Data for the date
        """
        # Filter market data for the date
        market_data_for_date = {}
        for symbol, data in market_data.items():
            date_data = data[data.index.strftime("%Y-%m-%d") == date]
            if not date_data.empty:
                market_data_for_date[symbol] = date_data
        
        # Filter news data for the date
        news_data_for_date = [news for news in news_data if news["date"] == date]
        
        # Filter filings data for the date
        filings_data_for_date = [filing for filing in filings_data if filing["date_filed"] == date]
        
        # Filter ECC data for the date
        ecc_data_for_date = [call for call in ecc_data if call["date"] == date]
        
        return {
            "date": date,
            "market_data": market_data_for_date,
            "news_data": news_data_for_date,
            "filings_data": filings_data_for_date,
            "ecc_data": ecc_data_for_date
        }
        
    def _calculate_daily_pnl(self, date, positions, market_data):
        """
        Calculate daily PnL.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            positions (dict): Dictionary of positions for each symbol
            market_data (dict): Market data dictionary
            
        Returns:
            float: Daily PnL percentage
        """
        pnl = 0.0
        position_count = 0
        
        for symbol in self.target_symbols:
            if symbol in positions and symbol in market_data:
                # Get position and return for the date
                position_series = positions[symbol]
                price_series = market_data[symbol]["Close"]
                
                # Calculate returns
                price_returns = price_series.pct_change()
                
                # Get date's return and previous position
                date_mask = price_returns.index.strftime("%Y-%m-%d") == date
                if not any(date_mask):
                    continue
                
                date_return = price_returns[date_mask].iloc[0]
                
                # Get previous day's position
                prev_day_idx = list(position_series.index).index(price_returns.index[date_mask][0]) - 1
                if prev_day_idx >= 0:
                    prev_position = position_series.iloc[prev_day_idx]
                else:
                    prev_position = 0.0
                
                # Calculate PnL
                symbol_pnl = date_return * prev_position
                pnl += symbol_pnl
                position_count += 1
        
        # Average PnL across positions
        if position_count > 0:
            pnl /= position_count
            
        return pnl * 100  # Convert to percentage
        
    def _calculate_portfolio_value(self, date, positions, market_data):
        """
        Calculate portfolio value.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            positions (dict): Dictionary of positions for each symbol
            market_data (dict): Market data dictionary
            
        Returns:
            float: Portfolio value
        """
        # Initial investment amount
        initial_investment = config.INITIAL_INVESTMENT
        
        # Calculate daily returns
        daily_returns = []
        
        for symbol in self.target_symbols:
            if symbol in positions and symbol in market_data:
                # Get position and return for the date
                position_series = positions[symbol]
                price_series = market_data[symbol]["Close"]
                
                # Calculate returns
                price_returns = price_series.pct_change()
                
                # Get date's return and previous position
                date_mask = price_returns.index.strftime("%Y-%m-%d") == date
                if not any(date_mask):
                    continue
                
                date_return = price_returns[date_mask].iloc[0]
                
                # Get previous day's position
                prev_day_idx = list(position_series.index).index(price_returns.index[date_mask][0]) - 1
                if prev_day_idx >= 0:
                    prev_position = position_series.iloc[prev_day_idx]
                else:
                    prev_position = 0.0
                
                # Add to daily returns weighted by position
                daily_returns.append(date_return * prev_position)
        
        # Calculate portfolio return
        if daily_returns:
            portfolio_return = sum(daily_returns) / len(daily_returns)
        else:
            portfolio_return = 0.0
            
        # Calculate cumulative return up to this day
        cumulative_returns = np.array(self._get_cumulative_returns_to_date(date, positions, market_data))
        if len(cumulative_returns) > 0:
            portfolio_value = initial_investment * (1 + cumulative_returns[-1] / 100)
        else:
            portfolio_value = initial_investment
            
        return portfolio_value
        
    def _get_cumulative_returns_to_date(self, end_date, positions, market_data):
        """
        Get cumulative returns up to a specific date.
        
        Args:
            end_date (str): End date in YYYY-MM-DD format
            positions (dict): Dictionary of positions for each symbol
            market_data (dict): Market data dictionary
            
        Returns:
            list: Daily cumulative returns up to the date
        """
        daily_pnl = []
        
        # Get date range
        all_dates = []
        for symbol, data in market_data.items():
            dates = data.index.strftime("%Y-%m-%d").tolist()
            all_dates.extend(dates)
        
        all_dates = sorted(list(set(all_dates)))
        date_range = [d for d in all_dates if d <= end_date]
        
        # Calculate daily PnL for each date
        for date in date_range:
            daily_pnl.append(self._calculate_daily_pnl(date, positions, market_data))
            
        # Calculate cumulative returns
        cumulative_returns = []
        cumulative_return = 0.0
        
        for pnl in daily_pnl:
            cumulative_return = (1 + cumulative_return / 100) * (1 + pnl / 100) * 100 - 100
            cumulative_returns.append(cumulative_return)
            
        return cumulative_returns
        
    def _calculate_episode_metrics(self, daily_pnl, positions, market_data):
        """
        Calculate performance metrics for an episode using evaluation metrics module.
        
        Args:
            daily_pnl (list): List of daily PnL percentages
            positions (dict): Dictionary of positions for each symbol
            market_data (dict): Market data dictionary
            
        Returns:
            dict: Performance metrics
        """
        # Convert positions to DataFrame
        positions_df = {}
        for symbol, pos in positions.items():
            positions_df[symbol] = pos
        positions_df = pd.DataFrame(positions_df)
            
        # Convert market data to DataFrame for prices
        prices_df = {}
        for symbol, data in market_data.items():
            prices_df[symbol] = data["Close"]
        prices_df = pd.DataFrame(prices_df)
            
        # Use evaluation metrics to calculate performance
        from evaluation.metrics import calculate_strategy_returns, calculate_performance_metrics
        
        # Calculate returns using the evaluation module
        strategy_returns = calculate_strategy_returns(prices_df, positions_df)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(strategy_returns, risk_free_rate=config.RISK_FREE_RATE)
        
        return metrics
        
    def _save_training_results(self):
        """Save training results to disk."""
        results_path = os.path.join(self.run_dir, "training_results.json")
        
        # Prepare results for saving
        serializable_results = []
        
        for episode in self.training_results:
            # Convert positions to serializable format
            positions = {}
            for symbol, pos in episode["positions"].items():
                if isinstance(pos, pd.Series):
                    positions[symbol] = pos.to_dict()
                else:
                    positions[symbol] = pos
                    
            serializable_episode = {
                "id": episode["id"],
                "type": episode["type"],
                "start_date": episode["start_date"],
                "end_date": episode["end_date"],
                "metrics": episode["metrics"],
                "daily_pnl": episode["daily_pnl"],
                "portfolio_values": episode["portfolio_values"],
                "risk_alerts": episode["risk_alerts"],
                "positions": positions
            }
            serializable_results.append(serializable_episode)
        
        # Save to disk
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved training results to {results_path}")
    
    def _save_testing_results(self):
        """Save testing results to disk."""
        results_path = os.path.join(self.run_dir, "testing_results.json")
        
        # Prepare results for saving
        # Convert positions to serializable format
        positions = {}
        for symbol, pos in self.testing_results["positions"].items():
            if isinstance(pos, pd.Series):
                positions[symbol] = pos.to_dict()
            else:
                positions[symbol] = pos
                
        serializable_results = {
            "id": self.testing_results["id"],
            "type": self.testing_results["type"],
            "start_date": self.testing_results["start_date"],
            "end_date": self.testing_results["end_date"],
            "metrics": self.testing_results["metrics"],
            "daily_pnl": self.testing_results["daily_pnl"],
            "portfolio_values": self.testing_results["portfolio_values"],
            "risk_alerts": self.testing_results["risk_alerts"],
            "positions": positions
        }
        
        # Save to disk
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Saved testing results to {results_path}")
        
    def visualize_results(self):
        """
        Create visualizations of the results using the evaluation visualization module.
        
        Returns:
            dict: Paths to generated visualizations
        """
        logger.info("Generating visualizations")
        
        # Create directory for visualizations
        viz_dir = os.path.join(self.run_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        visualization_paths = {}
        
        # Import visualization functions
        from evaluation.visualization import (
            plot_cumulative_returns, 
            plot_drawdowns, 
            plot_trading_positions, 
            plot_rolling_sharpe,
            plot_monthly_returns,
            create_performance_report
        )
        
        # Visualize training results if available
        if self.training_results:
            # Extract best episode based on Sharpe ratio
            best_episode = max(self.training_results, key=lambda x: x["metrics"]["sharpe_ratio"])
            
            # Create returns dictionary for each episode
            episode_returns = {}
            for i, episode in enumerate(self.training_results):
                # Convert positions to DataFrame
                positions_df = {}
                for symbol, pos in episode["positions"].items():
                    if isinstance(pos, dict):
                        positions_df[symbol] = pd.Series(pos)
                    else:
                        positions_df[symbol] = pos
                positions_df = pd.DataFrame(positions_df)
                
                # Convert daily PnL to returns Series
                daily_returns = pd.Series(episode["daily_pnl"], name=f"Episode {i+1}") / 100
                episode_returns[f"Episode {i+1}"] = daily_returns
            
            # Plot cumulative returns
            cr_path = os.path.join(viz_dir, "training_cumulative_returns.png")
            fig = plot_cumulative_returns(
                episode_returns, 
                title="Training: Cumulative Returns Across Episodes",
                save_path=cr_path
            )
            visualization_paths["training_cumulative_returns"] = cr_path
            
            # Plot positions for best episode
            pos_path = os.path.join(viz_dir, "training_best_positions.png")
            positions_df = {}
            for symbol, pos in best_episode["positions"].items():
                if isinstance(pos, dict):
                    positions_df[symbol] = pd.Series(pos)
                else:
                    positions_df[symbol] = pos
            positions_df = pd.DataFrame(positions_df)
            
            fig = plot_trading_positions(
                positions_df,
                title=f"Training: Positions for Best Episode ({best_episode['id']})",
                save_path=pos_path
            )
            visualization_paths["training_best_positions"] = pos_path
            
            # Extract market data for the best episode
            market_data_best = {}
            for symbol in self.target_symbols:
                start_date = best_episode["start_date"]
                end_date = best_episode["end_date"]
                if symbol in self.market_data:
                    mask = (self.market_data[symbol].index >= start_date) & (self.market_data[symbol].index <= end_date)
                    market_data_best[symbol] = self.market_data[symbol][mask]["Close"]
            market_data_df = pd.DataFrame(market_data_best)
            
            # Create drawdowns plot
            dd_path = os.path.join(viz_dir, "training_best_drawdowns.png")
            daily_returns = pd.Series(best_episode["daily_pnl"]) / 100
            fig = plot_drawdowns(
                daily_returns,
                title=f"Training: Drawdowns for Best Episode ({best_episode['id']})",
                save_path=dd_path
            )
            visualization_paths["training_best_drawdowns"] = dd_path
            
            # Create rolling Sharpe ratio plot
            rs_path = os.path.join(viz_dir, "training_best_rolling_sharpe.png")
            fig = plot_rolling_sharpe(
                daily_returns,
                window=30,
                title=f"Training: 30-Day Rolling Sharpe Ratio for Best Episode ({best_episode['id']})",
                save_path=rs_path
            )
            visualization_paths["training_best_rolling_sharpe"] = rs_path
            
            # Create comprehensive performance report for best episode
            report_paths = create_performance_report(
                daily_returns,
                positions_df,
                market_data_df,
                output_dir=os.path.join(viz_dir, "training_best_report")
            )
            visualization_paths.update({f"training_best_{k}": v for k, v in report_paths.items()})
        
        # Visualize testing results if available
        if self.testing_results:
            # Convert positions to DataFrame
            positions_df = {}
            for symbol, pos in self.testing_results["positions"].items():
                if isinstance(pos, dict):
                    positions_df[symbol] = pd.Series(pos)
                else:
                    positions_df[symbol] = pos
            positions_df = pd.DataFrame(positions_df)
            
            # Convert daily PnL to returns Series
            daily_returns = pd.Series(self.testing_results["daily_pnl"], name="FINCON Strategy") / 100
            
            # Plot cumulative returns
            cr_path = os.path.join(viz_dir, "testing_cumulative_returns.png")
            fig = plot_cumulative_returns(
                {"FINCON Strategy": daily_returns},
                title="Testing: Cumulative Returns",
                save_path=cr_path
            )
            visualization_paths["testing_cumulative_returns"] = cr_path
            
            # Plot positions
            pos_path = os.path.join(viz_dir, "testing_positions.png")
            fig = plot_trading_positions(
                positions_df,
                title="Testing: Trading Positions",
                save_path=pos_path
            )
            visualization_paths["testing_positions"] = pos_path
            
            # Extract market data for testing period
            market_data_test = {}
            for symbol in self.target_symbols:
                start_date = self.testing_results["start_date"]
                end_date = self.testing_results["end_date"]
                if symbol in self.market_data:
                    mask = (self.market_data[symbol].index >= start_date) & (self.market_data[symbol].index <= end_date)
                    market_data_test[symbol] = self.market_data[symbol][mask]["Close"]
            market_data_df = pd.DataFrame(market_data_test)
            
            # Create drawdowns plot
            dd_path = os.path.join(viz_dir, "testing_drawdowns.png")
            fig = plot_drawdowns(
                daily_returns,
                title="Testing: Drawdowns",
                save_path=dd_path
            )
            visualization_paths["testing_drawdowns"] = dd_path
            
            # Create rolling Sharpe ratio plot
            rs_path = os.path.join(viz_dir, "testing_rolling_sharpe.png")
            fig = plot_rolling_sharpe(
                daily_returns,
                window=30,
                title="Testing: 30-Day Rolling Sharpe Ratio",
                save_path=rs_path
            )
            visualization_paths["testing_rolling_sharpe"] = rs_path
            
            # Create monthly returns plot if sufficient data
            if len(daily_returns) >= 60:  # Need at least 2 months of data
                mr_path = os.path.join(viz_dir, "testing_monthly_returns.png")
                try:
                    fig = plot_monthly_returns(
                        daily_returns,
                        title="Testing: Monthly Returns",
                        save_path=mr_path
                    )
                    visualization_paths["testing_monthly_returns"] = mr_path
                except Exception as e:
                    logger.warning(f"Could not create monthly returns plot: {str(e)}")
            
            # Create comprehensive performance report
            report_paths = create_performance_report(
                daily_returns,
                positions_df,
                market_data_df,
                output_dir=os.path.join(viz_dir, "testing_report")
            )
            visualization_paths.update({f"testing_{k}": v for k, v in report_paths.items()})
        
        logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths
    
    def run_full_pipeline(self):
        """
        Run the complete pipeline: training and testing.
        
        Returns:
            dict: Results from the pipeline
        """
        # Train the system
        training_results = self.train()
        
        # Test the system
        testing_results = self.test()
        
        # Generate visualizations
        visualization_paths = self.visualize_results()
        
        # Generate summary report
        summary = {
            "timestamp": self.timestamp,
            "task_type": self.task_type,
            "target_symbols": self.target_symbols,
            "training_period": f"{self.training_start} to {self.training_end}",
            "testing_period": f"{self.testing_start} to {self.testing_end}",
            "training_episodes": len(self.training_results),
            "best_training_episode": {
                "id": max(self.training_results, key=lambda x: x["metrics"]["sharpe_ratio"])["id"],
                "metrics": max(self.training_results, key=lambda x: x["metrics"]["sharpe_ratio"])["metrics"]
            } if self.training_results else None,
            "testing_metrics": self.testing_results["metrics"] if self.testing_results else None,
            "visualization_paths": visualization_paths
        }
        
        # Save summary report
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved summary report to {summary_path}")
        
        return {
            "training_results": training_results,
            "testing_results": testing_results,
            "visualizations": visualization_paths,
            "summary": summary
        }


def main():
    """Main function to run FINCON system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FINCON: LLM Multi-Agent System for Financial Decision Making")
    
    # Task configuration
    parser.add_argument("--task_type", type=str, default="single_stock_trading",
                       choices=["single_stock_trading", "portfolio_management"],
                       help="Type of financial decision-making task")
    parser.add_argument("--symbols", type=str, nargs="+", required=True,
                       help="Stock symbols to trade")
    
    # Date ranges
    parser.add_argument("--training_start", type=str, default=config.TRAINING_START_DATE,
                       help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--training_end", type=str, default=config.TRAINING_END_DATE,
                       help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--testing_start", type=str, default=config.TESTING_START_DATE,
                       help="Testing start date (YYYY-MM-DD)")
    parser.add_argument("--testing_end", type=str, default=config.TESTING_END_DATE,
                       help="Testing end date (YYYY-MM-DD)")
    
    # Training settings
    parser.add_argument("--max_episodes", type=int, default=config.MAX_EPISODES,
                       help="Maximum number of training episodes")
    
    # LLM settings
    parser.add_argument("--llm_model", type=str, default=config.LLM_MODEL,
                       help="LLM model name")
    parser.add_argument("--llm_api_key", type=str, default=None,
                       help="LLM API key (defaults to environment variable)")
    
    # Output settings
    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR,
                       help="Directory to save results")
    parser.add_argument("--skip_visualizations", action="store_true",
                       help="Skip generating visualizations")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize FINCON system
    fincon = FINCON(
        target_symbols=args.symbols,
        task_type=args.task_type,
        training_start=args.training_start,
        training_end=args.training_end,
        testing_start=args.testing_start,
        testing_end=args.testing_end,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        max_episodes=args.max_episodes,
        results_dir=args.results_dir
    )
    
    # Run pipeline
    results = fincon.run_full_pipeline()
    
    # Log final results
    logger.info("=" * 50)
    logger.info("FINCON pipeline completed successfully")
    logger.info(f"Results saved to: {fincon.run_dir}")
    logger.info("=" * 50)
    logger.info("Best Training Episode Metrics:")
    for metric, value in results["summary"]["best_training_episode"]["metrics"].items():
        logger.info(f"  {metric}: {value}")
    logger.info("=" * 50)
    logger.info("Testing Metrics:")
    for metric, value in results["summary"]["testing_metrics"].items():
        logger.info(f"  {metric}: {value}")
    logger.info("=" * 50)
    
    return results


if __name__ == "__main__":
    main()