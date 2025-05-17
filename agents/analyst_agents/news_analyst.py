"""
News analyst agent for FINCON.
Extracts investment insights from financial news.
"""

import logging
from datetime import datetime
import json
import re

from agents.base_agent import BaseAgent

class NewsAnalyst(BaseAgent):
    """
    News analyst agent for FINCON.
    
    Specializes in extracting investment insights and sentiments from daily financial news.
    """
    
    def __init__(self, agent_id, target_symbols):
        """
        Initialize news analyst agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list): List of stock symbols to analyze
        """
        super().__init__(agent_id, "news_analyst", target_symbols)
        
        # Initialize sentiment dictionary
        self.sentiment_history = {symbol: [] for symbol in target_symbols}
        
        # Track processed news IDs to avoid duplicates
        self.processed_news_ids = set()
        
    def process(self, news_data):
        """
        Process financial news data and extract insights.
        
        Args:
            news_data (list): List of news articles
            
        Returns:
            dict: Extracted insights and sentiment analysis
        """
        if not news_data:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No news data provided.",
                "key_insights": [],
                "sentiment": {}
            }
        
        # Filter news related to target symbols
        relevant_news = []
        for article in news_data:
            # Check if article is already processed
            article_id = article.get("id", article.get("url", str(hash(json.dumps(article)))))
            if article_id in self.processed_news_ids:
                continue
                
            # Check if article is relevant to any target symbol
            is_relevant = False
            article_symbols = article.get("symbols", [])
            
            # If symbols are not provided, check content
            if not article_symbols and "content" in article:
                for symbol in self.target_symbols:
                    # Check if symbol appears in content
                    if re.search(r'\b' + re.escape(symbol) + r'\b', article["content"]):
                        is_relevant = True
                        if "symbols" not in article:
                            article["symbols"] = []
                        article["symbols"].append(symbol)
                        
            # Check explicit symbols
            else:
                for symbol in self.target_symbols:
                    if symbol in article_symbols:
                        is_relevant = True
                        break
                        
            if is_relevant:
                relevant_news.append(article)
                self.processed_news_ids.add(article_id)
                
        # If no relevant news, return empty insights
        if not relevant_news:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No relevant news found for target symbols.",
                "key_insights": [],
                "sentiment": {}
            }
            
        # Store relevant news in working memory
        for article in relevant_news:
            self.working_memory.add_item(article, "news_article")
            
        # Build prompt for LLM
        prompt = self._build_analysis_prompt(relevant_news)
        
        # Get analysis from LLM
        llm_response = self._get_llm_response(prompt)
        
        # Parse analysis
        try:
            analysis = self._parse_news_analysis(llm_response, relevant_news)
        except Exception as e:
            self.logger.error(f"Error parsing news analysis: {str(e)}")
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": f"Error parsing news analysis: {str(e)}",
                "key_insights": [],
                "sentiment": {}
            }
            
        # Store analysis in procedural memory
        analysis_text = json.dumps(analysis)
        analysis_embedding = self._generate_embedding(analysis_text)
        memory_id = self.add_to_procedural_memory(
            analysis_text,
            "news_analysis",
            analysis_embedding
        )
        
        # Update sentiment history
        for symbol, sentiment in analysis.get("sentiment", {}).items():
            if symbol in self.sentiment_history:
                self.sentiment_history[symbol].append({
                    "timestamp": analysis["timestamp"],
                    "sentiment": sentiment
                })
                
        self.logger.info(f"Generated news analysis with ID {memory_id}")
        return analysis
        
    def _build_analysis_prompt(self, news_articles):
        """
        Build prompt for news analysis.
        
        Args:
            news_articles (list): List of news articles
            
        Returns:
            str: Analysis prompt
        """
        prompt = f"""You are a financial news analyst specializing in extracting investment insights from daily financial news.

Your task is to analyze the following news articles related to these stocks: {', '.join(self.target_symbols)}

Please extract key insights, sentiments, and investment implications from these articles.

News Articles:
"""

        # Add news articles
        for i, article in enumerate(news_articles, 1):
            prompt += f"\nARTICLE {i}:\n"
            prompt += f"Date: {article.get('date', 'Unknown')}\n"
            prompt += f"Title: {article.get('title', 'Unknown')}\n"
            prompt += f"Source: {article.get('source', 'Unknown')}\n"
            prompt += f"Symbols: {', '.join(article.get('symbols', []))}\n"
            prompt += "Content:\n"
            prompt += f"{article.get('content', 'No content available')}\n"
            
        # Add instructions
        prompt += """
Based on these articles, please provide:

1. Key Insights: Extract the most important investment-relevant insights from these articles (bullet points)
2. Sentiment Analysis: Determine the sentiment (POSITIVE, NEGATIVE, or NEUTRAL) for each relevant stock symbol
3. Sentiment Score: Provide a sentiment score from -1.0 (extremely negative) to 1.0 (extremely positive)
4. Key Events: Identify any significant events or developments mentioned
5. Market Impact: Assess potential market impact of the news
6. Trading Implications: Suggest potential trading implications based solely on this news

Format your response using the following structure:

KEY INSIGHTS:
- [First key insight]
- [Second key insight]
...

SENTIMENT ANALYSIS:
[SYMBOL]: [SENTIMENT] ([SCORE])
[Brief explanation of sentiment assessment]
...

KEY EVENTS:
- [First key event]
- [Second key event]
...

MARKET IMPACT:
[Assessment of potential market impact]

TRADING IMPLICATIONS:
[Potential trading implications based on this news]

Please ensure your analysis is objective, balanced, and directly based on the content of the articles.
"""

        return prompt
        
    def _parse_news_analysis(self, llm_response, news_articles):
        """
        Parse news analysis from LLM response.
        
        Args:
            llm_response (str): LLM response text
            news_articles (list): Original news articles
            
        Returns:
            dict: Parsed news analysis
        """
        # Initialize analysis structure
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.target_symbols,
            "source_articles": len(news_articles),
            "key_insights": [],
            "sentiment": {},
            "key_events": [],
            "market_impact": "",
            "trading_implications": ""
        }
        
        # Parse key insights
        if "KEY INSIGHTS:" in llm_response:
            insights_section = llm_response.split("KEY INSIGHTS:")[1].split("SENTIMENT ANALYSIS:")[0]
            insights_bullets = [line.strip()[2:].strip() for line in insights_section.strip().split("\n") if line.strip().startswith("-")]
            analysis["key_insights"] = insights_bullets
            
        # Parse sentiment analysis
        if "SENTIMENT ANALYSIS:" in llm_response:
            sentiment_section = llm_response.split("SENTIMENT ANALYSIS:")[1]
            if "KEY EVENTS:" in sentiment_section:
                sentiment_section = sentiment_section.split("KEY EVENTS:")[0]
                
            # Process each line in sentiment section
            for line in sentiment_section.strip().split("\n"):
                # Check if line contains symbol and sentiment
                if ":" in line and any(symbol in line for symbol in self.target_symbols):
                    try:
                        # Extract symbol
                        symbol_part = line.split(":")[0].strip()
                        
                        # Find which target symbol is in the line
                        symbol = next((s for s in self.target_symbols if s in symbol_part), None)
                        if not symbol:
                            continue
                            
                        # Extract sentiment and score
                        sentiment_part = line.split(":")[1].strip()
                        
                        sentiment_value = "NEUTRAL"  # Default
                        if "POSITIVE" in sentiment_part:
                            sentiment_value = "POSITIVE"
                        elif "NEGATIVE" in sentiment_part:
                            sentiment_value = "NEGATIVE"
                            
                        # Extract score if present
                        score = 0.0
                        score_match = re.search(r'\(([+-]?[0-9]*\.?[0-9]+)\)', sentiment_part)
                        if score_match:
                            score = float(score_match.group(1))
                            
                        # Store sentiment
                        analysis["sentiment"][symbol] = {
                            "value": sentiment_value,
                            "score": score
                        }
                    except Exception as e:
                        self.logger.warning(f"Error parsing sentiment line: {line}, Error: {str(e)}")
            
        # Parse key events
        if "KEY EVENTS:" in llm_response:
            events_section = llm_response.split("KEY EVENTS:")[1]
            if "MARKET IMPACT:" in events_section:
                events_section = events_section.split("MARKET IMPACT:")[0]
                
            events_bullets = [line.strip()[2:].strip() for line in events_section.strip().split("\n") if line.strip().startswith("-")]
            analysis["key_events"] = events_bullets
            
        # Parse market impact
        if "MARKET IMPACT:" in llm_response:
            impact_section = llm_response.split("MARKET IMPACT:")[1]
            if "TRADING IMPLICATIONS:" in impact_section:
                impact_section = impact_section.split("TRADING IMPLICATIONS:")[0]
                
            analysis["market_impact"] = impact_section.strip()
            
        # Parse trading implications
        if "TRADING IMPLICATIONS:" in llm_response:
            implications_section = llm_response.split("TRADING IMPLICATIONS:")[1]
            analysis["trading_implications"] = implications_section.strip()
            
        return analysis
        
    def receive_feedback(self, feedback):
        """
        Receive and process feedback from manager agent.
        
        Args:
            feedback (dict): Feedback from manager agent
            
        Returns:
            bool: True if feedback was processed successfully
        """
        if not feedback:
            return False
            
        # Update memory based on feedback
        importance_change = feedback.get("importance_change", 0.0)
        
        # Get recent memory events
        recent_events = self.procedural_memory.get_events_by_type("news_analysis")
        if not recent_events:
            return False
            
        # Update importance of most recent event
        recent_events.sort(key=lambda x: x["timestamp"], reverse=True)
        if recent_events:
            most_recent_event = recent_events[0]
            self.procedural_memory.update_importance(most_recent_event["id"], importance_change)
            
            self.logger.info(f"Updated importance of event {most_recent_event['id']} by {importance_change}")
            
        return True
        
    def update_beliefs(self, beliefs_update):
        """
        Update agent's beliefs based on feedback from risk control.
        
        Args:
            beliefs_update (dict): Updated beliefs
            
        Returns:
            bool: True if beliefs were updated successfully
        """
        if not beliefs_update or "news_insights" not in beliefs_update:
            return False
            
        # Create belief update prompt
        new_belief = beliefs_update["news_insights"]
        
        prompt = f"""You are a financial news analyst. You have received updated investment beliefs regarding news analysis:

{new_belief}

Based on this feedback, update your approach to analyzing financial news. Consider:
1. What specific aspects of news should you pay more attention to?
2. How should you adjust your sentiment analysis?
3. What types of news are most valuable for investment decisions?

Provide a concise summary of how you will adjust your analysis approach.
"""
        # Get response from LLM
        response = self._get_llm_response(prompt)
        
        # Store updated belief in procedural memory
        belief_text = f"Updated Belief - News Analysis: {new_belief}\n\nImplementation Plan: {response}"
        belief_embedding = self._generate_embedding(belief_text)
        
        memory_id = self.add_to_procedural_memory(
            belief_text,
            "belief_update",
            belief_embedding,
            importance=2.0  # High importance for belief updates
        )
        
        self.logger.info(f"Updated news analysis beliefs with ID {memory_id}")
        return True
        
    def get_sentiment_trend(self, symbol, days=5):
        """
        Get sentiment trend for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of most recent days to consider
            
        Returns:
            dict: Sentiment trend information
        """
        if symbol not in self.sentiment_history or not self.sentiment_history[symbol]:
            return {
                "symbol": symbol,
                "sentiment_trend": "NEUTRAL",
                "average_score": 0.0,
                "data_points": 0
            }
            
        # Get most recent sentiment entries
        recent_entries = sorted(self.sentiment_history[symbol], key=lambda x: x["timestamp"], reverse=True)
        recent_entries = recent_entries[:days]
        
        if not recent_entries:
            return {
                "symbol": symbol,
                "sentiment_trend": "NEUTRAL",
                "average_score": 0.0,
                "data_points": 0
            }
            
        # Calculate average sentiment score
        sentiment_scores = [entry["sentiment"].get("score", 0.0) for entry in recent_entries if "sentiment" in entry]
        if not sentiment_scores:
            average_score = 0.0
        else:
            average_score = sum(sentiment_scores) / len(sentiment_scores)
            
        # Determine trend
        if average_score > 0.3:
            trend = "POSITIVE"
        elif average_score < -0.3:
            trend = "NEGATIVE"
        else:
            trend = "NEUTRAL"
            
        return {
            "symbol": symbol,
            "sentiment_trend": trend,
            "average_score": average_score,
            "data_points": len(recent_entries)
        }