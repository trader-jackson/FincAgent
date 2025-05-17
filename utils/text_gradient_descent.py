"""
Text gradient descent for FINCON system.
Implements verbal reinforcement mechanisms for belief updates.
"""

import logging
import re
import difflib
from datetime import datetime

# Configure logging
logger = logging.getLogger("text_gradient_descent")

class TextGradientDescent:
    """
    Implements text-based gradient descent for verbal reinforcement.
    
    Updates text-based beliefs by analyzing reasoning trajectories and performance
    differences between learning episodes.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize text gradient descent.
        
        Args:
            llm_client: LLM client for generating conceptual insights
        """
        self.llm_client = llm_client
        
    def update_text(self, original_text, update_direction, learning_rate=0.5):
        """
        Update text based on update direction and learning rate.
        
        Args:
            original_text (str): Original text to update
            update_direction (str): Direction to update the text
            learning_rate (float): Learning rate for update (0-1)
            
        Returns:
            str: Updated text
        """
        # No LLM client, return original text
        if not self.llm_client:
            logger.warning("No LLM client provided for text gradient descent")
            return original_text
            
        # Prepare prompt for generating updated text
        prompt = f"""You are responsible for updating text-based investment beliefs based on new insights.

Original Text:
```
{original_text}
```

Update Direction:
```
{update_direction}
```

Learning Rate: {learning_rate}

Please update the original text according to the update direction, with the magnitude of changes proportional to the learning rate (0-1). 
A learning rate of 0 would keep the original text, while a learning rate of 1 would fully incorporate the update direction.
This learning rate of {learning_rate} means you should {learning_rate*100:.0f}% integrate the new insights.

The updated text should maintain the same style, format, and structure as the original, but incorporate the key insights from the update direction proportionally to the learning rate.

Return only the updated text without any additional comments or explanations.
"""
        # Generate updated text
        updated_text = self.llm_client.generate(prompt, temperature=0.0)
        
        return updated_text
        
    def compute_text_similarity(self, text1, text2):
        """
        Compute similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Use difflib's SequenceMatcher to compute similarity
        matcher = difflib.SequenceMatcher(None, text1, text2)
        similarity = matcher.ratio()
        
        return similarity
        
    def compute_learning_rate(self, action_sequences):
        """
        Compute learning rate based on action sequences overlap.
        
        Args:
            action_sequences (list): List of action sequence pairs
            
        Returns:
            float: Learning rate based on action overlap
        """
        # If no action sequences, use default rate
        if not action_sequences:
            return 0.5
            
        # Calculate overlaps for each pair
        overlaps = []
        for seq1, seq2 in action_sequences:
            overlap = self._calculate_action_overlap(seq1, seq2)
            overlaps.append(overlap)
            
        # Compute learning rate as inverse of average overlap
        if overlaps:
            avg_overlap = sum(overlaps) / len(overlaps)
            # Learning rate is higher when overlap is lower
            learning_rate = 1.0 - avg_overlap
            # Ensure learning rate is in [0, 1]
            learning_rate = max(0.0, min(1.0, learning_rate))
        else:
            learning_rate = 0.5
            
        return learning_rate
        
    def _calculate_action_overlap(self, actions1, actions2):
        """
        Calculate the overlap in actions between two sequences.
        
        Args:
            actions1 (list): First sequence of actions
            actions2 (list): Second sequence of actions
            
        Returns:
            float: Percentage of overlapping actions (0-1)
        """
        # Ensure actions are of the same length
        min_length = min(len(actions1), len(actions2))
        
        if min_length == 0:
            return 0.0
            
        # Count matching actions
        matches = 0
        for i in range(min_length):
            if actions1[i] == actions2[i]:
                matches += 1
                
        return matches / min_length
        
    def generate_meta_prompt(self, better_episode, worse_episode):
        """
        Generate meta-prompt for belief update.
        
        Args:
            better_episode (dict): Better performing episode
            worse_episode (dict): Worse performing episode
            
        Returns:
            str: Meta-prompt for belief update
        """
        # No LLM client, return empty prompt
        if not self.llm_client:
            logger.warning("No LLM client provided for generating meta-prompt")
            return ""
            
        # Prepare comparison data
        comparison_data = self._prepare_episode_comparison(better_episode, worse_episode)
        
        # Build prompt
        prompt = f"""You are a financial investment advisor responsible for identifying why one trading strategy outperformed another.

Better Performing Episode (ID: {better_episode['id']}):
- Sharpe Ratio: {comparison_data['better_sharpe']:.4f}
- Cumulative Return: {comparison_data['better_return']:.2%}
- Max Drawdown: {comparison_data['better_drawdown']:.2%}

Worse Performing Episode (ID: {worse_episode['id']}):
- Sharpe Ratio: {comparison_data['worse_sharpe']:.4f}
- Cumulative Return: {comparison_data['worse_return']:.2%}
- Max Drawdown: {comparison_data['worse_drawdown']:.2%}

Differences in Trading Decisions:
{comparison_data['decision_differences']}

Based on this information, please identify the key investment insights that explain why the better episode outperformed the worse episode.
Focus on specific trading strategies, information usage patterns, and risk management approaches that led to better performance.

Your analysis should be organized by different aspects of investment decision-making:
1. Historical Momentum: How technical indicators and price momentum influenced decisions
2. News Insights: How financial news information was interpreted and applied
3. Filing Information: How SEC filing data was utilized in decision-making
4. Earnings Call Insights: How earnings call information impacted trading choices
5. Market Data Analysis: How market data and technical analysis affected performance
6. Other Aspects: Any other relevant factors that contributed to performance differences

Format your response as a concise, actionable meta-prompt that can guide investment beliefs going forward.
"""
        # Generate meta-prompt
        meta_prompt = self.llm_client.generate(prompt, temperature=0.1)
        
        return meta_prompt
        
    def _prepare_episode_comparison(self, better_episode, worse_episode):
        """
        Prepare episode comparison data for meta-prompt.
        
        Args:
            better_episode (dict): Better performing episode
            worse_episode (dict): Worse performing episode
            
        Returns:
            dict: Comparison data
        """
        comparison = {
            "better_sharpe": better_episode.get("metrics", {}).get("sharpe_ratio", 0.0),
            "worse_sharpe": worse_episode.get("metrics", {}).get("sharpe_ratio", 0.0),
            "better_return": better_episode.get("metrics", {}).get("cumulative_return", 0.0) / 100.0,
            "worse_return": worse_episode.get("metrics", {}).get("cumulative_return", 0.0) / 100.0,
            "better_drawdown": better_episode.get("metrics", {}).get("max_drawdown", 0.0) / 100.0,
            "worse_drawdown": worse_episode.get("metrics", {}).get("max_drawdown", 0.0) / 100.0,
            "decision_differences": ""
        }
        
        # Extract decision differences
        better_actions = better_episode.get("actions", [])
        worse_actions = worse_episode.get("actions", [])
        
        # Find significant differences
        differences = []
        min_length = min(len(better_actions), len(worse_actions))
        
        for i in range(min_length):
            better_action = better_actions[i]
            worse_action = worse_actions[i]
            
            # Check if decisions differ
            if better_action.get("decision") != worse_action.get("decision"):
                date = better_action.get("timestamp", "").split("T")[0]
                symbol = better_action.get("symbol", "Unknown")
                
                better_decision = better_action.get("decision", "UNKNOWN")
                worse_decision = worse_action.get("decision", "UNKNOWN")
                
                differences.append(f"Date: {date}, Symbol: {symbol}")
                differences.append(f"- Better episode: {better_decision}")
                differences.append(f"- Worse episode: {worse_decision}")
                differences.append(f"- Better reasoning: {better_action.get('reasoning', '')[:100]}...")
                differences.append(f"- Worse reasoning: {worse_action.get('reasoning', '')[:100]}...")
                differences.append("")
                
        # Limit to 5 differences
        if len(differences) > 25:
            differences = differences[:25]
            differences.append("(Additional differences truncated)")
            
        comparison["decision_differences"] = "\n".join(differences)
        
        return comparison