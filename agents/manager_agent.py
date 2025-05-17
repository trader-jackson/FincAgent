"""
Manager agent for FINCON.
Responsible for consolidating analyst insights and making trading decisions.
"""

import logging
import json
from datetime import datetime
import numpy as np
import cvxpy as cp

from agents.base_agent import BaseAgent
from risk_control.within_episode import WithinEpisodeRiskControl

class ManagerAgent(BaseAgent):
    """
    Manager agent for FINCON.
    
    Responsible for consolidating insights from analyst agents and making trading decisions.
    The manager agent is the sole decision-maker in the FINCON system.
    """
    
    def __init__(self, agent_id, target_symbols, risk_control=None):
        """
        Initialize manager agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list): List of stock symbols to trade
            risk_control (WithinEpisodeRiskControl, optional): Risk control component
        """
        super().__init__(agent_id, "manager", target_symbols)
        
        # Initialize risk control if not provided
        self.risk_control = risk_control or WithinEpisodeRiskControl()
        
        # Track current positions and trading history
        self.current_positions = {symbol: 0.0 for symbol in target_symbols}
        self.trading_history = []
        self.daily_pnl = []
        
        # Risk status
        self.risk_status = "NORMAL"  # Can be NORMAL or ELEVATED
        self.risk_alert = None
        
        # Current task type (single stock or portfolio)
        self.task_type = "single_stock_trading" if len(target_symbols) == 1 else "portfolio_management"
        
        # Store the latest analyst insights
        self.latest_insights = {}
        
        self.logger.info(f"Initialized manager agent with task type: {self.task_type}")
        
    def process(self, observations):
        """
        Process observations and make trading decisions.
        
        Args:
            observations (dict): Dictionary of observations from various sources
            
        Returns:
            dict: Trading decisions and reasoning
        """
        # Store observations in working memory
        processed_observations = []
        for source, observation in observations.items():
            processed_observation = self.working_memory.process_observation(observation, source)
            processed_observations.append(processed_observation)
            
            # Update latest insights for analyst sources
            if source.endswith("_analyst"):
                self.latest_insights[source] = observation
        
        # Check if there's a risk alert
        if "risk_alert" in observations:
            self.risk_alert = observations["risk_alert"]
            self.risk_status = "ELEVATED" if self.risk_alert["level"] in ["MEDIUM", "HIGH"] else "NORMAL"
            
        # Generate trading decisions based on task type
        if self.task_type == "single_stock_trading":
            return self._make_single_stock_decision(processed_observations)
        else:
            return self._make_portfolio_decision(processed_observations)
            
    def _make_single_stock_decision(self, observations):
        """
        Make trading decision for a single stock.
        
        Args:
            observations (list): Processed observations
            
        Returns:
            dict: Trading decision and reasoning
        """
        symbol = self.target_symbols[0]
        current_position = self.current_positions.get(symbol, 0.0)
        
        # Build trading prompt
        prompt = self._build_trading_prompt(observations, symbol, current_position)
        
        # Get decision from LLM
        llm_response = self._get_llm_response(prompt)
        
        # Parse decision
        try:
            decision = self._parse_trading_decision(llm_response, symbol)
        except Exception as e:
            self.logger.error(f"Error parsing trading decision: {str(e)}")
            decision = {
                "symbol": symbol,
                "decision": "HOLD",
                "position_size": current_position,
                "reasoning": f"Error parsing decision: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            
        # Update current position
        self.current_positions[symbol] = decision["position_size"]
        
        # Add to trading history
        self.trading_history.append(decision)
        
        return decision
        
    def _make_portfolio_decision(self, observations):
        """
        Make trading decisions for a portfolio of stocks.
        
        Args:
            observations (list): Processed observations
            
        Returns:
            dict: Portfolio decisions and reasoning
        """
        # Build trading prompt for each symbol
        decisions = {}
        reasonings = {}
        
        for symbol in self.target_symbols:
            current_position = self.current_positions.get(symbol, 0.0)
            
            # Build trading prompt
            prompt = self._build_trading_prompt(observations, symbol, current_position)
            
            # Get decision from LLM
            llm_response = self._get_llm_response(prompt)
            
            # Parse decision
            try:
                decision = self._parse_trading_decision(llm_response, symbol)
                decisions[symbol] = decision["decision"]
                reasonings[symbol] = decision["reasoning"]
            except Exception as e:
                self.logger.error(f"Error parsing trading decision for {symbol}: {str(e)}")
                decisions[symbol] = "HOLD"
                reasonings[symbol] = f"Error parsing decision: {str(e)}"
                
        # Calculate portfolio weights using mean-variance optimization
        weights = self._optimize_portfolio_weights(decisions)
        
        # Create portfolio decision
        portfolio_decision = {
            "timestamp": datetime.now().isoformat(),
            "decisions": decisions,
            "weights": weights,
            "reasonings": reasonings,
            "risk_status": self.risk_status
        }
        
        # Update current positions
        for symbol, weight in weights.items():
            self.current_positions[symbol] = weight
            
        # Add to trading history
        self.trading_history.append(portfolio_decision)
        
        return portfolio_decision
        
    def _build_trading_prompt(self, observations, symbol, current_position):
        """
        Build trading decision prompt.
        
        Args:
            observations (list): Processed observations
            symbol (str): Stock symbol
            current_position (float): Current position size
            
        Returns:
            str: Trading decision prompt
        """
        # Get latest insights from memory
        memory_query = f"trading insights for {symbol}"
        memory_events = self.retrieve_from_procedural_memory(memory_query)
        
        # Get current risk status
        risk_text = "NORMAL"
        if self.risk_status == "ELEVATED":
            risk_text = f"ELEVATED (Alert Level: {self.risk_alert['level']})"
            
        # Build prompt
        prompt = f"""You are the trading manager for {symbol} with responsibility for making trading decisions.

Current Information:
- Symbol: {symbol}
- Current Position: {current_position} (Positive = Long, Negative = Short, 0 = No Position)
- Current Risk Status: {risk_text}
- Date: {datetime.now().strftime('%Y-%m-%d')}

Latest Analyst Insights:
"""

        # Add analyst insights
        for source, insight in self.latest_insights.items():
            if source.endswith("_analyst"):
                analyst_type = source.replace("_analyst", "").upper()
                prompt += f"\n{analyst_type} INSIGHTS:\n"
                
                # Format insights based on type
                if isinstance(insight, dict):
                    for key, value in insight.items():
                        if key not in ["timestamp", "source", "type"]:
                            prompt += f"- {key}: {value}\n"
                elif isinstance(insight, str):
                    prompt += insight + "\n"
                else:
                    prompt += str(insight) + "\n"
        
        # Add memory events
        prompt += "\nRelevant Historical Insights:\n"
        for event in memory_events:
            prompt += f"- {event['content']}\n"
            
        # Add self-reflection from episodic memory if available
        if self.episodic_memory:
            latest_beliefs = self.episodic_memory.get_latest_beliefs()
            if latest_beliefs:
                prompt += "\nInvestment Beliefs:\n"
                for aspect, belief in latest_beliefs.get("beliefs", {}).items():
                    prompt += f"- {aspect.replace('_', ' ').title()}: {belief}\n"
        
        # Add risk guidance
        if self.risk_status == "ELEVATED":
            prompt += f"\nRisk Alert Information:\n{self.risk_alert['description']}\n"
            
            if "recommendations" in self.risk_alert:
                prompt += "\nRisk Recommendations:\n"
                for rec in self.risk_alert["recommendations"]:
                    prompt += f"- {rec}\n"
        
        # Add decision instructions
        prompt += """
Based on the information above, make a trading decision for this symbol. You must choose one of the following actions:
1. BUY (establish or increase a long position)
2. SELL (establish or increase a short position)
3. HOLD (maintain current position)

For your decision, also specify a position size between -1.0 (maximum short) and 1.0 (maximum long), where 0 represents no position.

Your response must be in the following format:

Decision: [BUY/SELL/HOLD]
Position Size: [number between -1.0 and 1.0]
Reasoning:
[Your detailed reasoning explaining the decision, referencing the analyst insights that influenced your decision]

Please ensure your decision is well-reasoned and considers all available information, including risk status.
"""

        return prompt
        
    def _parse_trading_decision(self, llm_response, symbol):
        """
        Parse trading decision from LLM response.
        
        Args:
            llm_response (str): LLM response text
            symbol (str): Stock symbol
            
        Returns:
            dict: Parsed trading decision
        """
        # Parse decision
        decision_match = None
        if "Decision:" in llm_response:
            decision_parts = llm_response.split("Decision:")
            if len(decision_parts) > 1:
                decision_line = decision_parts[1].strip().split("\n")[0].strip()
                if "BUY" in decision_line.upper():
                    decision_match = "BUY"
                elif "SELL" in decision_line.upper():
                    decision_match = "SELL"
                elif "HOLD" in decision_line.upper():
                    decision_match = "HOLD"
        
        # If no match, default to HOLD
        if not decision_match:
            self.logger.warning(f"Could not parse decision from response for {symbol}, defaulting to HOLD")
            decision_match = "HOLD"
            
        # Parse position size
        position_size = None
        if "Position Size:" in llm_response:
            size_parts = llm_response.split("Position Size:")
            if len(size_parts) > 1:
                size_line = size_parts[1].strip().split("\n")[0].strip()
                try:
                    position_size = float(size_line)
                    # Ensure position size is between -1 and 1
                    position_size = max(-1.0, min(1.0, position_size))
                except ValueError:
                    self.logger.warning(f"Could not parse position size from response for {symbol}")
        
        # If no position size parsed, use default based on decision
        if position_size is None:
            if decision_match == "BUY":
                position_size = 1.0
            elif decision_match == "SELL":
                position_size = -1.0
            else:  # HOLD
                position_size = self.current_positions.get(symbol, 0.0)
                
        # Parse reasoning
        reasoning = "No reasoning provided."
        if "Reasoning:" in llm_response:
            reasoning_parts = llm_response.split("Reasoning:")
            if len(reasoning_parts) > 1:
                reasoning = reasoning_parts[1].strip()
                
        # Create decision object
        decision = {
            "symbol": symbol,
            "decision": decision_match,
            "position_size": position_size,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "risk_status": self.risk_status
        }
        
        return decision
        
    def _optimize_portfolio_weights(self, decisions):
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            decisions (dict): Trading decisions for each symbol
            
        Returns:
            dict: Optimized portfolio weights
        """
        # Convert trading decisions to constraints
        constraints = []
        symbols = list(decisions.keys())
        n = len(symbols)
        
        # Create portfolio weights variable
        w = cp.Variable(n)
        
        # Add constraints based on trading decisions
        for i, symbol in enumerate(symbols):
            decision = decisions[symbol]
            if decision == "BUY":
                # For BUY, weight should be positive
                constraints.append(w[i] >= 0)
                constraints.append(w[i] <= 1)
            elif decision == "SELL":
                # For SELL, weight should be negative
                constraints.append(w[i] <= 0)
                constraints.append(w[i] >= -1)
            else:  # HOLD
                # For HOLD, weight should be 0
                constraints.append(w[i] == 0)
                
        # Add constraint for sum of absolute weights (leverage constraint)
        constraints.append(cp.sum(cp.abs(w)) <= 1)
        
        # If risk status is elevated, add additional constraints
        if self.risk_status == "ELEVATED":
            # Reduce overall exposure
            constraints.append(cp.sum(cp.abs(w)) <= 0.7)
            
        # Get expected returns and covariance matrix
        # These would typically come from historical data or analyst forecasts
        # For simplicity, we'll use dummy values here
        returns = np.random.normal(0.001, 0.0005, n)  # Expected daily returns
        cov_matrix = np.eye(n) * 0.0001  # Simplified covariance matrix
        
        # Define optimization objective
        objective = cp.Maximize(returns @ w - cp.quad_form(w, cov_matrix))
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            # Check if solution was found
            if problem.status == "optimal" or problem.status == "optimal_inaccurate":
                # Convert solution to dictionary
                weights = {symbol: float(w.value[i]) for i, symbol in enumerate(symbols)}
                return weights
            else:
                self.logger.warning(f"Portfolio optimization failed: {problem.status}")
                # Return zero weights
                return {symbol: 0.0 for symbol in symbols}
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            # Return zero weights
            return {symbol: 0.0 for symbol in symbols}
            
    def update_pnl(self, daily_pnl, symbol=None):
        """
        Update daily PnL and recalculate risk.
        
        Args:
            daily_pnl (float): Daily Profit and Loss value
            symbol (str, optional): Symbol for single stock trading
            
        Returns:
            float: Updated CVaR value
        """
        self.daily_pnl.append(daily_pnl)
        
        # Update CVaR in risk control
        cvar = self.risk_control.update_pnl(daily_pnl)
        
        # Check for risk alert
        risk_alert = self.risk_control.check_risk_alert()
        if risk_alert:
            self.risk_alert = risk_alert
            self.risk_status = "ELEVATED"
            
            # Add recommendations
            positions = self.current_positions
            if symbol:
                positions = {symbol: self.current_positions.get(symbol, 0.0)}
                
            recommendations = self.risk_control.generate_risk_recommendations(risk_alert, positions)
            self.risk_alert["recommendations"] = recommendations
            
            self.logger.warning(f"Risk alert triggered: {risk_alert['level']} - CVaR: {cvar}")
        else:
            # Reset risk status if no alert
            self.risk_status = "NORMAL"
            
        return cvar
        
    def self_reflect(self, latest_pnl):
        """
        Conduct self-reflection on trading decisions.
        
        Args:
            latest_pnl (float): Latest daily PnL
            
        Returns:
            str: Self-reflection text
        """
        # Build reflection prompt
        prompt = f"""As the trading manager, reflect on your recent trading decisions and performance.

Latest performance:
- Daily PnL: {latest_pnl}
- Current risk status: {self.risk_status}

Recent trading decisions:
"""

        # Add recent trading decisions
        recent_decisions = self.trading_history[-5:] if len(self.trading_history) > 5 else self.trading_history
        for decision in recent_decisions:
            if "symbol" in decision:
                # Single stock decision
                prompt += f"- {decision['timestamp']}: {decision['symbol']} - {decision['decision']} (Position: {decision['position_size']})\n"
            else:
                # Portfolio decision
                prompt += f"- {decision['timestamp']}: Portfolio decision with {len(decision['decisions'])} symbols\n"
                
        prompt += """
Please reflect on these decisions and provide:
1. Assessment of what went well and what could be improved
2. Factors that most influenced the recent performance
3. Adjustments that should be made to the trading strategy
4. How effectively risk was managed

Your reflection should be concise and focused on actionable insights for future trading decisions.
"""

        # Get reflection from LLM
        reflection = self._get_llm_response(prompt)
        
        # Store reflection in procedural memory
        reflection_id = self.add_to_procedural_memory(
            reflection,
            "self_reflection",
            self._generate_embedding(reflection)
        )
        
        # If episodic memory exists, add reflection there too
        if self.episodic_memory:
            # Get current episode or create new one
            episodes = self.episodic_memory.get_episodes(limit=1)
            if episodes:
                current_episode = episodes[0]
                
                # Add reflection to episode
                if "reflections" not in current_episode:
                    current_episode["reflections"] = []
                    
                current_episode["reflections"].append({
                    "timestamp": datetime.now().isoformat(),
                    "content": reflection,
                    "pnl": latest_pnl
                })
                
        self.logger.info(f"Generated self-reflection: {reflection_id}")
        return reflection
        
    def send_feedback_to_analysts(self, daily_pnl):
        """
        Send feedback to analyst agents based on performance.
        
        Args:
            daily_pnl (float): Daily Profit and Loss value
            
        Returns:
            dict: Feedback for each analyst
        """
        feedback = {}
        
        # Get latest trading decision
        latest_decision = self.trading_history[-1] if self.trading_history else None
        if not latest_decision:
            return feedback
            
        # Determine which analysts contributed to the decision
        contributing_analysts = []
        
        # Extract reasoning to identify contributing analysts
        if "reasoning" in latest_decision:
            reasoning = latest_decision["reasoning"]
            
            # Check for mentions of different analyst types
            if "news" in reasoning.lower() or "article" in reasoning.lower() or "media" in reasoning.lower():
                contributing_analysts.append("news_analyst")
                
            if "filing" in reasoning.lower() or "10-k" in reasoning.lower() or "10-q" in reasoning.lower():
                contributing_analysts.append("filing_analyst")
                
            if "earnings call" in reasoning.lower() or "conference call" in reasoning.lower():
                contributing_analysts.append("ecc_analyst")
                
            if "technical" in reasoning.lower() or "indicator" in reasoning.lower() or "momentum" in reasoning.lower():
                contributing_analysts.append("data_analyst")
                
        # Generate feedback for each contributing analyst
        for analyst in contributing_analysts:
            # Increase importance of relevant memory events
            for source, insight in self.latest_insights.items():
                if source == analyst:
                    # Create feedback based on performance
                    if daily_pnl > 0:
                        feedback[analyst] = {
                            "performance_impact": "POSITIVE",
                            "importance_change": 0.5,
                            "message": f"Your insights contributed to a profitable trading decision with PnL of {daily_pnl}."
                        }
                    else:
                        feedback[analyst] = {
                            "performance_impact": "NEGATIVE",
                            "importance_change": -0.2,
                            "message": f"Your insights were used in a trading decision that resulted in a loss of {daily_pnl}."
                        }
                    
                    # Add insight to procedural memory with updated importance
                    insight_text = json.dumps(insight) if isinstance(insight, dict) else str(insight)
                    insight_embedding = self._generate_embedding(insight_text)
                    
                    # Add to memory with importance based on performance
                    importance = 1.0 + (0.5 if daily_pnl > 0 else -0.2)
                    memory_id = self.add_to_procedural_memory(
                        insight_text,
                        f"{analyst}_insight",
                        insight_embedding,
                        importance
                    )
                    
                    self.logger.info(f"Added {analyst} insight to memory with ID {memory_id} and importance {importance}")
                        
        return feedback
        
    def update_investment_beliefs(self, beliefs_update):
        """
        Update investment beliefs with new conceptualized beliefs.
        
        Args:
            beliefs_update (dict): Updated investment beliefs
            
        Returns:
            bool: True if beliefs were updated successfully
        """
        if not self.episodic_memory:
            self.logger.warning("Cannot update investment beliefs: no episodic memory")
            return False
            
        # Update beliefs in episodic memory
        self.episodic_memory.update_beliefs(
            beliefs_update["id"],
            beliefs_update["updated_beliefs"],
            beliefs_update.get("source_episodes"),
            beliefs_update.get("reasoning")
        )
        
        self.logger.info(f"Updated investment beliefs: {beliefs_update['id']}")
        return True
        
    def reset(self):
        """Reset agent for a new episode."""
        # Reset risk control
        self.risk_control.reset()
        
        # Reset trading state
        self.trading_history = []
        self.daily_pnl = []
        self.current_positions = {symbol: 0.0 for symbol in self.target_symbols}
        self.risk_status = "NORMAL"
        self.risk_alert = None
        
        # Clear working memory
        self.working_memory.clear()
        
        self.logger.info("Reset manager agent for new episode")