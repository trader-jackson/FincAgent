"""
Risk control agent for FINCON.
Implements within-episode and over-episode risk control mechanisms.
"""

import logging
from datetime import datetime
import json
import numpy as np

from agents.base_agent import BaseAgent
from risk_control.within_episode import WithinEpisodeRiskControl
from risk_control.over_episode import OverEpisodeRiskControl

class RiskControlAgent(BaseAgent):
    """
    Risk control agent for FINCON.
    
    Responsible for monitoring risk and updating investment beliefs through 
    within-episode and over-episode risk control mechanisms.
    """
    
    def __init__(self, agent_id, target_symbols=None):
        """
        Initialize risk control agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list, optional): List of stock symbols to monitor
        """
        super().__init__(agent_id, "risk_control", target_symbols)
        
        # Initialize risk control components
        self.within_episode = WithinEpisodeRiskControl()
        self.over_episode = OverEpisodeRiskControl(self._llm_client_adapter())
        
        # Track episode history
        self.episode_history = {}
        
    def process_daily_risk(self, daily_pnl, current_positions):
        """
        Process daily risk and check for alerts.
        
        Args:
            daily_pnl (float): Daily Profit and Loss value
            current_positions (dict): Current trading positions
            
        Returns:
            dict or None: Risk alert if triggered, None otherwise
        """
        # Update PnL history and recalculate CVaR
        cvar = self.within_episode.update_pnl(daily_pnl)
        
        # Check for risk alert
        risk_alert = self.within_episode.check_risk_alert()
        
        if risk_alert:
            # Generate recommendations
            recommendations = self.within_episode.generate_risk_recommendations(risk_alert, current_positions)
            risk_alert["recommendations"] = recommendations
            
            # Store alert in procedural memory
            alert_text = json.dumps(risk_alert)
            alert_embedding = self._generate_embedding(alert_text)
            memory_id = self.add_to_procedural_memory(
                alert_text,
                "risk_alert",
                alert_embedding,
                importance=2.0  # High importance for risk alerts
            )
            
            self.logger.warning(f"Risk alert triggered: {risk_alert['level']} - CVaR: {cvar}")
            
        return risk_alert
        
    def process_episode_comparison(self, episode1, episode2):
        """
        Compare two episodes and update investment beliefs.
        
        Args:
            episode1 (dict): First episode data
            episode2 (dict): Second episode data
            
        Returns:
            dict: Updated investment beliefs
        """
        # Store episodes in history
        self.episode_history[episode1["id"]] = episode1
        self.episode_history[episode2["id"]] = episode2
        
        # Compare episodes to determine which performed better
        comparison = self.over_episode.compare_episodes(episode1, episode2)
        
        # Extract conceptualized insights from both episodes
        better_insights = self.over_episode.extract_conceptualized_insights(
            self.episode_history[comparison["better_episode"]]
        )
        
        worse_insights = self.over_episode.extract_conceptualized_insights(
            self.episode_history[comparison["worse_episode"]]
        )
        
        # Update investment beliefs
        belief_update = self.over_episode.update_investment_beliefs(
            self.episode_history[comparison["better_episode"]], 
            self.episode_history[comparison["worse_episode"]]
        )
        
        # Store in procedural memory
        update_text = json.dumps(belief_update)
        update_embedding = self._generate_embedding(update_text)
        memory_id = self.add_to_procedural_memory(
            update_text,
            "belief_update",
            update_embedding,
            importance=2.0  # High importance for belief updates
        )
        
        # Generate belief propagation map
        propagation_map = self.over_episode.generate_belief_propagation_map(belief_update)
        belief_update["propagation_map"] = propagation_map
        
        self.logger.info(f"Generated investment belief update with ID {memory_id}")
        return belief_update
        
    def _llm_client_adapter(self):
        """
        Create adapter for LLM client compatible with OverEpisodeRiskControl.
        
        Returns:
            object: LLM client adapter
        """
        # Create adapter class to bridge BaseAgent _get_llm_response with OverEpisodeRiskControl
        class LLMClientAdapter:
            def __init__(self, agent):
                self.agent = agent
                
            def generate(self, prompt, temperature=None):
                return self.agent._get_llm_response(prompt, temperature)
                
        return LLMClientAdapter(self)
        
    def get_risk_status(self):
        """
        Get current risk status.
        
        Returns:
            str: Current risk status
        """
        # Check if CVaR history is available
        if not self.within_episode.cvar_history:
            return "NORMAL"
            
        # Check if there was a recent alert
        recent_alerts = self.procedural_memory.get_events_by_type("risk_alert")
        if recent_alerts:
            # Sort by timestamp (newest first)
            recent_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Get most recent alert
            most_recent_alert = recent_alerts[0]
            
            # Parse alert JSON
            try:
                alert_data = json.loads(most_recent_alert["content"])
                return f"ELEVATED ({alert_data.get('level', 'UNKNOWN')})"
            except:
                pass
                
        return "NORMAL"
        
    def get_latest_beliefs(self):
        """
        Get latest investment beliefs.
        
        Returns:
            dict: Latest investment beliefs
        """
        # Get recent belief updates
        recent_beliefs = self.procedural_memory.get_events_by_type("belief_update")
        if not recent_beliefs:
            return {}
            
        # Sort by timestamp (newest first)
        recent_beliefs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Get most recent belief update
        most_recent_belief = recent_beliefs[0]
        
        # Parse belief JSON
        try:
            belief_data = json.loads(most_recent_belief["content"])
            return belief_data
        except:
            return {}
            
    def reset_within_episode(self):
        """Reset within-episode risk control."""
        self.within_episode.reset()
        self.logger.info("Reset within-episode risk control")
        
    def format_belief_update_for_agent(self, agent_type, beliefs):
        """
        Format belief update for a specific agent type.
        
        Args:
            agent_type (str): Type of agent
            beliefs (dict): Beliefs to format
            
        Returns:
            str: Formatted belief update text
        """
        return self.over_episode.format_belief_update_for_agent(agent_type, beliefs)