"""
Over-episode risk control for FINCON system.
Implements Conceptual Verbal Reinforcement (CVRF) for investment belief updates.
"""

import logging
import json
import numpy as np
from datetime import datetime

class OverEpisodeRiskControl:
    """
    Over-episode risk control component for FINCON.
    
    Updates investment beliefs based on reasoning trajectories and profit-and-loss trends,
    distilled into conceptual perspectives using CVRF mechanism.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize over-episode risk control.
        
        Args:
            llm_client: LLM client for generating conceptual insights
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger("over_episode_risk_control")
        self.belief_aspects = [
            "historical_momentum",
            "news_insights",
            "filing_information",
            "earnings_call_insights",
            "market_data_analysis",
            "other_aspects"
        ]
        
    def compare_episodes(self, episode1, episode2):
        """
        Compare two episodes to determine which performed better.
        
        Args:
            episode1 (dict): First episode data
            episode2 (dict): Second episode data
            
        Returns:
            dict: Comparison results
        """
        metrics1 = episode1.get("metrics", {})
        metrics2 = episode2.get("metrics", {})
        
        # Compare using Sharpe Ratio as primary metric
        sharpe1 = metrics1.get("sharpe_ratio", -float('inf'))
        sharpe2 = metrics2.get("sharpe_ratio", -float('inf'))
        
        # Determine which episode performed better
        if sharpe1 > sharpe2:
            better_episode = episode1
            worse_episode = episode2
            comparison_text = f"Episode {episode1['id']} outperformed Episode {episode2['id']} with a Sharpe Ratio of {sharpe1:.4f} vs {sharpe2:.4f}"
        else:
            better_episode = episode2
            worse_episode = episode1
            comparison_text = f"Episode {episode2['id']} outperformed Episode {episode1['id']} with a Sharpe Ratio of {sharpe2:.4f} vs {sharpe1:.4f}"
            
        # Calculate action overlap
        action_overlap = self._calculate_action_overlap(episode1, episode2)
        
        comparison = {
            "better_episode": better_episode["id"],
            "worse_episode": worse_episode["id"],
            "better_sharpe": max(sharpe1, sharpe2),
            "worse_sharpe": min(sharpe1, sharpe2),
            "sharpe_difference": abs(sharpe1 - sharpe2),
            "action_overlap": action_overlap,
            "comparison_text": comparison_text
        }
        
        self.logger.info(comparison_text)
        return comparison
        
    def _calculate_action_overlap(self, episode1, episode2):
        """
        Calculate the overlap in trading actions between two episodes.
        
        Args:
            episode1 (dict): First episode
            episode2 (dict): Second episode
            
        Returns:
            float: Percentage of overlapping actions (0-1)
        """
        actions1 = episode1.get("actions", [])
        actions2 = episode2.get("actions", [])
        
        # Ensure actions are of the same length
        min_length = min(len(actions1), len(actions2))
        
        if min_length == 0:
            return 0.0
            
        # Count matching actions
        matches = 0
        for i in range(min_length):
            action1 = actions1[i]
            action2 = actions2[i]
            
            # Check if the trading decisions match
            if "decision" in action1 and "decision" in action2:
                if action1["decision"] == action2["decision"]:
                    matches += 1
                    
        return matches / min_length
        
    def extract_conceptualized_insights(self, episode, max_insights=3):
        """
        Extract conceptualized investment insights from an episode.
        
        Args:
            episode (dict): Episode data
            max_insights (int): Maximum number of insights per aspect
            
        Returns:
            dict: Conceptualized insights for each aspect
        """
        # Extract profitable and unprofitable trading patterns
        profitable_trades = []
        unprofitable_trades = []
        
        for action in episode.get("actions", []):
            if "pnl" in action and "reasoning" in action:
                if action["pnl"] > 0:
                    profitable_trades.append(action)
                else:
                    unprofitable_trades.append(action)
                    
        # Sort by PnL (most profitable/unprofitable first)
        profitable_trades = sorted(profitable_trades, key=lambda x: x["pnl"], reverse=True)
        unprofitable_trades = sorted(unprofitable_trades, key=lambda x: x["pnl"])
        
        # Limit to max_insights per category
        profitable_trades = profitable_trades[:max_insights]
        unprofitable_trades = unprofitable_trades[:max_insights]
        
        # Extract patterns in reasoning from profitable and unprofitable trades
        patterns = {}
        
        for aspect in self.belief_aspects:
            patterns[aspect] = {"profitable": [], "unprofitable": []}
            
            # Extract patterns from profitable trades
            for trade in profitable_trades:
                reasoning = trade.get("reasoning", "")
                if self._reasoning_contains_aspect(reasoning, aspect):
                    pattern = self._extract_aspect_pattern(reasoning, aspect)
                    if pattern and pattern not in patterns[aspect]["profitable"]:
                        patterns[aspect]["profitable"].append(pattern)
                        
            # Extract patterns from unprofitable trades
            for trade in unprofitable_trades:
                reasoning = trade.get("reasoning", "")
                if self._reasoning_contains_aspect(reasoning, aspect):
                    pattern = self._extract_aspect_pattern(reasoning, aspect)
                    if pattern and pattern not in patterns[aspect]["unprofitable"]:
                        patterns[aspect]["unprofitable"].append(pattern)
        
        # Use LLM to conceptualize insights
        conceptualized_insights = {}
        
        if self.llm_client:
            for aspect in self.belief_aspects:
                if patterns[aspect]["profitable"] or patterns[aspect]["unprofitable"]:
                    # Prepare prompt for LLM
                    prompt = self._prepare_conceptualization_prompt(
                        aspect, 
                        patterns[aspect]["profitable"],
                        patterns[aspect]["unprofitable"]
                    )
                    
                    # Get conceptualized insight from LLM
                    response = self._get_llm_response(prompt)
                    
                    # Store conceptualized insight
                    conceptualized_insights[aspect] = response
                    
        return conceptualized_insights
        
    def _reasoning_contains_aspect(self, reasoning, aspect):
        """
        Check if reasoning contains references to a specific aspect.
        
        Args:
            reasoning (str): Reasoning text
            aspect (str): Aspect to check for
            
        Returns:
            bool: True if aspect is found in reasoning
        """
        aspect_keywords = {
            "historical_momentum": ["momentum", "trend", "price movement", "technical", "moving average"],
            "news_insights": ["news", "article", "press", "announcement", "media"],
            "filing_information": ["filing", "10-K", "10-Q", "SEC", "report", "financial statement"],
            "earnings_call_insights": ["earnings call", "conference call", "earnings report", "guidance"],
            "market_data_analysis": ["market data", "volume", "volatility", "indicators"],
            "other_aspects": ["sector", "industry", "competitors", "macroeconomic", "sentiment"]
        }
        
        keywords = aspect_keywords.get(aspect, [])
        return any(keyword.lower() in reasoning.lower() for keyword in keywords)
        
    def _extract_aspect_pattern(self, reasoning, aspect):
        """
        Extract pattern related to specific aspect from reasoning.
        
        Args:
            reasoning (str): Reasoning text
            aspect (str): Aspect to extract pattern for
            
        Returns:
            str: Extracted pattern or None if not found
        """
        # Split reasoning into sentences
        sentences = reasoning.split('.')
        relevant_sentences = []
        
        aspect_keywords = {
            "historical_momentum": ["momentum", "trend", "price movement", "technical", "moving average"],
            "news_insights": ["news", "article", "press", "announcement", "media"],
            "filing_information": ["filing", "10-K", "10-Q", "SEC", "report", "financial statement"],
            "earnings_call_insights": ["earnings call", "conference call", "earnings report", "guidance"],
            "market_data_analysis": ["market data", "volume", "volatility", "indicators"],
            "other_aspects": ["sector", "industry", "competitors", "macroeconomic", "sentiment"]
        }
        
        keywords = aspect_keywords.get(aspect, [])
        
        # Find sentences relevant to the aspect
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                relevant_sentences.append(sentence.strip())
                
        if not relevant_sentences:
            return None
            
        # Return concatenated relevant sentences
        return " ".join(relevant_sentences)
        
    def _prepare_conceptualization_prompt(self, aspect, profitable_patterns, unprofitable_patterns):
        """
        Prepare prompt for LLM to conceptualize insights.
        
        Args:
            aspect (str): Aspect to conceptualize
            profitable_patterns (list): Patterns from profitable trades
            unprofitable_patterns (list): Patterns from unprofitable trades
            
        Returns:
            str: Prompt for LLM
        """
        prompt = f"""You are a financial investment advisor specializing in conceptualizing trading patterns. 
        
Based on the trading history of an investment agent, I need you to conceptualize investment insights related to '{aspect}'.

Profitable trading patterns:
{self._format_patterns(profitable_patterns)}

Unprofitable trading patterns:
{self._format_patterns(unprofitable_patterns)}

Please provide a conceptualized investment belief update about '{aspect}' based on these patterns. Focus on:
1. What strategies related to '{aspect}' led to profitable trades?
2. What strategies related to '{aspect}' led to unprofitable trades?
3. How should the investment approach to '{aspect}' be adjusted to improve performance?

Your response should be concise (2-3 paragraphs) and provide actionable guidance. Be specific about what to do and what to avoid regarding '{aspect}'.
"""
        return prompt
        
    def _format_patterns(self, patterns):
        """
        Format patterns list for prompt.
        
        Args:
            patterns (list): List of patterns
            
        Returns:
            str: Formatted patterns text
        """
        if not patterns:
            return "No clear patterns identified."
            
        return "\n".join([f"- {pattern}" for pattern in patterns])
        
    def _get_llm_response(self, prompt):
        """
        Get response from LLM.
        
        Args:
            prompt (str): Prompt for LLM
            
        Returns:
            str: LLM response
        """
        if not self.llm_client:
            return "LLM client not available for conceptualization."
            
        try:
            # Call LLM API (implementation depends on specific LLM client)
            response = self.llm_client.generate(prompt, temperature=0.0)
            return response
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return f"Error conceptualizing insights: {str(e)}"
            
    def update_investment_beliefs(self, better_episode, worse_episode):
        """
        Update investment beliefs using CVRF mechanism.
        
        Args:
            better_episode (dict): Better performing episode
            worse_episode (dict): Worse performing episode
            
        Returns:
            dict: Updated investment beliefs
        """
        # Extract conceptualized insights from both episodes
        better_insights = self.extract_conceptualized_insights(better_episode)
        worse_insights = self.extract_conceptualized_insights(worse_episode)
        
        # Compare conceptualized insights to derive updated beliefs
        updated_beliefs = {}
        reasoning = {}
        
        for aspect in self.belief_aspects:
            better_insight = better_insights.get(aspect)
            worse_insight = worse_insights.get(aspect)
            
            if better_insight:
                # If we have insights from the better episode, use them
                updated_beliefs[aspect] = better_insight
                reasoning[aspect] = f"Derived from better-performing episode {better_episode['id']}"
            elif worse_insight:
                # If we only have insights from the worse episode, use negative learning
                updated_beliefs[aspect] = f"Avoid: {worse_insight}"
                reasoning[aspect] = f"Derived by avoiding patterns from worse-performing episode {worse_episode['id']}"
                
        # Calculate overlap percentage for learning rate
        action_overlap = self._calculate_action_overlap(better_episode, worse_episode)
        learning_rate = 1.0 - action_overlap
        
        # Create belief update object
        belief_update = {
            "id": f"belief_update_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "source_episodes": [better_episode["id"], worse_episode["id"]],
            "better_episode": better_episode["id"],
            "updated_beliefs": updated_beliefs,
            "reasoning": reasoning,
            "learning_rate": learning_rate,
            "action_overlap": action_overlap
        }
        
        self.logger.info(f"Generated investment belief update with learning rate {learning_rate:.2f}")
        return belief_update
        
    def generate_belief_propagation_map(self, belief_update):
        """
        Generate map for propagating beliefs to specific agents.
        
        Args:
            belief_update (dict): Updated investment beliefs
            
        Returns:
            dict: Mapping of beliefs to agent types
        """
        propagation_map = {}
        
        # Define which beliefs should be propagated to which agents
        agent_aspect_mapping = {
            "manager": self.belief_aspects,  # Manager gets all aspects
            "news_analyst": ["news_insights"],
            "filing_analyst": ["filing_information"],
            "ecc_analyst": ["earnings_call_insights"],
            "data_analyst": ["market_data_analysis", "historical_momentum"],
            "stock_selection_agent": ["historical_momentum", "market_data_analysis", "other_aspects"]
        }
        
        # Create propagation map
        for agent_type, aspects in agent_aspect_mapping.items():
            relevant_beliefs = {}
            
            for aspect in aspects:
                if aspect in belief_update["updated_beliefs"]:
                    relevant_beliefs[aspect] = belief_update["updated_beliefs"][aspect]
                    
            if relevant_beliefs:
                propagation_map[agent_type] = relevant_beliefs
                
        return propagation_map
        
    def format_belief_update_for_agent(self, agent_type, beliefs):
        """
        Format belief update for a specific agent type.
        
        Args:
            agent_type (str): Type of agent
            beliefs (dict): Beliefs to format
            
        Returns:
            str: Formatted belief update text
        """
        formatted_text = f"INVESTMENT BELIEF UPDATE FOR {agent_type.upper()}\n"
        formatted_text += "=============================================\n\n"
        
        for aspect, belief in beliefs.items():
            formatted_text += f"{aspect.replace('_', ' ').title()}:\n"
            formatted_text += f"{belief}\n\n"
            
        formatted_text += "Please incorporate these updated investment beliefs into your analysis and decision-making process."
        
        return formatted_text