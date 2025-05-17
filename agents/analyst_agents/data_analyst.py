"""
Data analyst agent for FINCON.
Analyzes market data and computes technical indicators.
"""

import logging
from datetime import datetime
import json
import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

class DataAnalyst(BaseAgent):
    """
    Data analyst agent for FINCON.
    
    Specializes in quantitative analysis of market data, including technical indicators,
    momentum analysis, and risk metrics.
    """
    
    def __init__(self, agent_id, target_symbols):
        """
        Initialize data analyst agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list): List of stock symbols to analyze
        """
        super().__init__(agent_id, "data_analyst", target_symbols)
        
        # Track technical analysis history
        self.technical_history = {symbol: {} for symbol in target_symbols}
        
    def process(self, market_data):
        """
        Process market data and compute technical indicators.
        
        Args:
            market_data (dict): Dictionary of market data for each symbol
            
        Returns:
            dict: Technical analysis and insights
        """
        if not market_data:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No market data provided.",
                "technical_indicators": {},
                "momentum_analysis": {},
                "risk_metrics": {}
            }
        
        # Filter data for target symbols
        relevant_data = {}
        for symbol, data in market_data.items():
            if symbol in self.target_symbols:
                relevant_data[symbol] = data
                
        # If no relevant data, return empty analysis
        if not relevant_data:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No relevant market data found for target symbols.",
                "technical_indicators": {},
                "momentum_analysis": {},
                "risk_metrics": {}
            }
            
        # Process market data for each symbol
        technical_indicators = {}
        momentum_analysis = {}
        risk_metrics = {}
        
        for symbol, data in relevant_data.items():
            # Store last 30 days of data in working memory
            recent_data = data.tail(30)
            self.working_memory.add_item(
                {"symbol": symbol, "data": recent_data.to_dict()},
                "market_data"
            )
            
            # Compute technical indicators
            technical_indicators[symbol] = self._compute_technical_indicators(data, symbol)
            
            # Compute momentum analysis
            momentum_analysis[symbol] = self._compute_momentum_analysis(data, symbol)
            
            # Compute risk metrics
            risk_metrics[symbol] = self._compute_risk_metrics(data, symbol)
        
        # Create analysis object
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "symbols": list(relevant_data.keys()),
            "technical_indicators": technical_indicators,
            "momentum_analysis": momentum_analysis,
            "risk_metrics": risk_metrics
        }
        
        # Get summary insights from LLM
        analysis["insights"] = self._generate_insights(analysis)
        
        # Store analysis in procedural memory
        analysis_text = json.dumps({
            "timestamp": analysis["timestamp"],
            "symbols": analysis["symbols"],
            "insights": analysis["insights"]
        })
        analysis_embedding = self._generate_embedding(analysis_text)
        memory_id = self.add_to_procedural_memory(
            analysis_text,
            "market_data_analysis",
            analysis_embedding
        )
        
        # Update technical history
        for symbol, indicators in technical_indicators.items():
            if symbol in self.technical_history:
                self.technical_history[symbol][analysis["timestamp"]] = indicators
                
        self.logger.info(f"Generated market data analysis with ID {memory_id}")
        return analysis
        
    def _compute_technical_indicators(self, data, symbol):
        """
        Compute technical indicators for a stock.
        
        Args:
            data (pd.DataFrame): Price data for the stock
            symbol (str): Stock symbol
            
        Returns:
            dict: Technical indicators
        """
        # Ensure we have at least 200 days of data
        if len(data) < 200:
            self.logger.warning(f"Insufficient data for {symbol} to compute all technical indicators")
            
        # Get most recent values (last row)
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        # Extract basic indicators
        indicators = {}
        for col in ["Close", "Volume", "SMA20", "SMA50", "SMA200", "RSI", "MACD", "MACD_Signal", "Momentum"]:
            if col in latest:
                indicators[col] = float(latest[col])
        
        # Calculate price changes
        if "Close" in latest and "Close" in prev:
            indicators["price_change_1d"] = float(latest["Close"] - prev["Close"])
            indicators["price_change_pct_1d"] = float((latest["Close"] / prev["Close"] - 1) * 100)
            
        # Calculate change from moving averages
        if "Close" in latest and "SMA20" in latest:
            indicators["change_from_sma20"] = float((latest["Close"] / latest["SMA20"] - 1) * 100)
        if "Close" in latest and "SMA50" in latest:
            indicators["change_from_sma50"] = float((latest["Close"] / latest["SMA50"] - 1) * 100)
        if "Close" in latest and "SMA200" in latest:
            indicators["change_from_sma200"] = float((latest["Close"] / latest["SMA200"] - 1) * 100)
            
        # Add moving average crossovers
        if "SMA20" in latest and "SMA50" in latest:
            indicators["sma20_above_sma50"] = latest["SMA20"] > latest["SMA50"]
            
            # Check if crossover occurred recently (last 5 days)
            if len(data) >= 5:
                prev_above = data.iloc[-5]["SMA20"] > data.iloc[-5]["SMA50"]
                indicators["sma20_50_crossover"] = (indicators["sma20_above_sma50"] != prev_above)
                indicators["sma20_50_golden_cross"] = (indicators["sma20_above_sma50"] and indicators["sma20_50_crossover"])
                indicators["sma20_50_death_cross"] = (not indicators["sma20_above_sma50"] and indicators["sma20_50_crossover"])
        
        # Add MACD signals
        if "MACD" in latest and "MACD_Signal" in latest:
            indicators["macd_above_signal"] = latest["MACD"] > latest["MACD_Signal"]
            
            # Check if crossover occurred recently
            if len(data) >= 5:
                prev_macd_above = data.iloc[-5]["MACD"] > data.iloc[-5]["MACD_Signal"]
                indicators["macd_signal_crossover"] = (indicators["macd_above_signal"] != prev_macd_above)
                indicators["macd_bullish_crossover"] = (indicators["macd_above_signal"] and indicators["macd_signal_crossover"])
                indicators["macd_bearish_crossover"] = (not indicators["macd_above_signal"] and indicators["macd_signal_crossover"])
        
        # Add RSI signals
        if "RSI" in latest:
            indicators["rsi_overbought"] = latest["RSI"] > 70
            indicators["rsi_oversold"] = latest["RSI"] < 30
        
        return indicators
        
    def _compute_momentum_analysis(self, data, symbol):
        """
        Compute momentum analysis for a stock.
        
        Args:
            data (pd.DataFrame): Price data for the stock
            symbol (str): Stock symbol
            
        Returns:
            dict: Momentum analysis
        """
        momentum = {}
        
        # Get recent closing prices
        if len(data) < 30:
            self.logger.warning(f"Insufficient data for {symbol} to compute momentum analysis")
            return momentum
            
        # Calculate short-term momentum (5 days)
        momentum["short_term"] = float(data["Close"].pct_change(5).iloc[-1] * 100) if "Close" in data else 0.0
        
        # Calculate medium-term momentum (20 days)
        momentum["medium_term"] = float(data["Close"].pct_change(20).iloc[-1] * 100) if "Close" in data else 0.0
        
        # Calculate long-term momentum (60 days)
        if len(data) >= 60:
            momentum["long_term"] = float(data["Close"].pct_change(60).iloc[-1] * 100) if "Close" in data else 0.0
        
        # Determine momentum signals
        momentum["short_term_signal"] = "POSITIVE" if momentum.get("short_term", 0) > 2 else "NEGATIVE" if momentum.get("short_term", 0) < -2 else "NEUTRAL"
        momentum["medium_term_signal"] = "POSITIVE" if momentum.get("medium_term", 0) > 5 else "NEGATIVE" if momentum.get("medium_term", 0) < -5 else "NEUTRAL"
        momentum["long_term_signal"] = "POSITIVE" if momentum.get("long_term", 0) > 10 else "NEGATIVE" if momentum.get("long_term", 0) < -10 else "NEUTRAL"
        
        # Calculate overall momentum signal
        positive_signals = sum(1 for signal in [momentum.get("short_term_signal"), momentum.get("medium_term_signal"), momentum.get("long_term_signal")] if signal == "POSITIVE")
        negative_signals = sum(1 for signal in [momentum.get("short_term_signal"), momentum.get("medium_term_signal"), momentum.get("long_term_signal")] if signal == "NEGATIVE")
        
        if positive_signals > negative_signals:
            momentum["overall_signal"] = "POSITIVE"
        elif negative_signals > positive_signals:
            momentum["overall_signal"] = "NEGATIVE"
        else:
            momentum["overall_signal"] = "NEUTRAL"
            
        return momentum
        
    def _compute_risk_metrics(self, data, symbol):
        """
        Compute risk metrics for a stock.
        
        Args:
            data (pd.DataFrame): Price data for the stock
            symbol (str): Stock symbol
            
        Returns:
            dict: Risk metrics
        """
        risk = {}
        
        # Calculate daily returns
        if "Close" not in data or len(data) < 30:
            self.logger.warning(f"Insufficient data for {symbol} to compute risk metrics")
            return risk
            
        daily_returns = data["Close"].pct_change().dropna()
        
        # Calculate volatility (annualized)
        risk["volatility_daily"] = float(daily_returns.std())
        risk["volatility_annualized"] = float(risk["volatility_daily"] * np.sqrt(252))
        
        # Calculate Value at Risk (VaR) at 95% confidence level
        var_95 = float(np.percentile(daily_returns, 5))
        risk["var_95"] = var_95
        
        # Calculate Conditional Value at Risk (CVaR) at 95% confidence level
        risk["cvar_95"] = float(daily_returns[daily_returns <= var_95].mean())
        
        # Calculate downside risk
        negative_returns = daily_returns[daily_returns < 0]
        risk["downside_risk"] = float(negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0.0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        risk["max_drawdown"] = float(drawdown.min() * 100)
        
        # Calculate risk assessment
        if risk["volatility_annualized"] > 0.3:
            risk["volatility_assessment"] = "HIGH"
        elif risk["volatility_annualized"] > 0.15:
            risk["volatility_assessment"] = "MEDIUM"
        else:
            risk["volatility_assessment"] = "LOW"
            
        # Overall risk assessment based on volatility and drawdown
        if risk["volatility_assessment"] == "HIGH" or risk["max_drawdown"] < -20:
            risk["overall_assessment"] = "HIGH"
        elif risk["volatility_assessment"] == "MEDIUM" or risk["max_drawdown"] < -10:
            risk["overall_assessment"] = "MEDIUM"
        else:
            risk["overall_assessment"] = "LOW"
            
        return risk
        
    def _generate_insights(self, analysis):
        """
        Generate market data insights using LLM.
        
        Args:
            analysis (dict): Technical analysis data
            
        Returns:
            dict: Market insights for each symbol
        """
        insights = {}
        
        for symbol in analysis["symbols"]:
            # Build prompt for LLM
            prompt = self._build_insights_prompt(
                symbol,
                analysis["technical_indicators"].get(symbol, {}),
                analysis["momentum_analysis"].get(symbol, {}),
                analysis["risk_metrics"].get(symbol, {})
            )
            
            # Get insights from LLM
            llm_response = self._get_llm_response(prompt)
            
            # Parse insights
            try:
                symbol_insights = self._parse_insights(llm_response)
                insights[symbol] = symbol_insights
            except Exception as e:
                self.logger.error(f"Error parsing insights for {symbol}: {str(e)}")
                insights[symbol] = {
                    "key_points": [f"Error parsing insights: {str(e)}"],
                    "recommendation": "NEUTRAL",
                    "confidence": "LOW"
                }
                
        return insights
        
    def _build_insights_prompt(self, symbol, technical_indicators, momentum_analysis, risk_metrics):
        """
        Build prompt for generating market data insights.
        
        Args:
            symbol (str): Stock symbol
            technical_indicators (dict): Technical indicators
            momentum_analysis (dict): Momentum analysis
            risk_metrics (dict): Risk metrics
            
        Returns:
            str: Insights prompt
        """
        prompt = f"""You are a quantitative data analyst specializing in market data analysis.

Your task is to analyze the following technical indicators, momentum analysis, and risk metrics for {symbol} and provide investment insights.

Technical Indicators:
"""

        # Add technical indicators
        for indicator, value in technical_indicators.items():
            prompt += f"- {indicator}: {value}\n"
            
        # Add momentum analysis
        prompt += "\nMomentum Analysis:\n"
        for metric, value in momentum_analysis.items():
            prompt += f"- {metric}: {value}\n"
            
        # Add risk metrics
        prompt += "\nRisk Metrics:\n"
        for metric, value in risk_metrics.items():
            prompt += f"- {metric}: {value}\n"
            
        # Add instructions
        prompt += """
Based on this data, please provide:

1. Key Points: Extract the most important insights from this market data (bullet points)
2. Technical Outlook: Summarize the technical outlook based on indicators
3. Momentum Assessment: Provide an assessment of the momentum signals
4. Risk Assessment: Analyze the risk profile
5. Recommendation: Suggest a trading recommendation (BUY, SELL, or HOLD) with confidence level (HIGH, MEDIUM, LOW)

Format your response using the following structure:

KEY POINTS:
- [First key point]
- [Second key point]
...

TECHNICAL OUTLOOK:
[Summary of technical outlook]

MOMENTUM ASSESSMENT:
[Assessment of momentum signals]

RISK ASSESSMENT:
[Analysis of risk profile]

RECOMMENDATION:
[BUY/SELL/HOLD] (Confidence: [HIGH/MEDIUM/LOW])
[Brief explanation of recommendation]

Please ensure your analysis is objective, balanced, and based directly on the data provided.
"""

        return prompt
        
    def _parse_insights(self, llm_response):
        """
        Parse market data insights from LLM response.
        
        Args:
            llm_response (str): LLM response text
            
        Returns:
            dict: Parsed market insights
        """
        # Initialize insights structure
        insights = {
            "key_points": [],
            "technical_outlook": "",
            "momentum_assessment": "",
            "risk_assessment": "",
            "recommendation": "NEUTRAL",
            "confidence": "LOW",
            "explanation": ""
        }
        
        # Parse key points
        if "KEY POINTS:" in llm_response:
            points_section = llm_response.split("KEY POINTS:")[1].split("TECHNICAL OUTLOOK:")[0]
            points_bullets = [line.strip()[2:].strip() for line in points_section.strip().split("\n") if line.strip().startswith("-")]
            insights["key_points"] = points_bullets
            
        # Parse technical outlook
        if "TECHNICAL OUTLOOK:" in llm_response:
            outlook_section = llm_response.split("TECHNICAL OUTLOOK:")[1].split("MOMENTUM ASSESSMENT:")[0]
            insights["technical_outlook"] = outlook_section.strip()
            
        # Parse momentum assessment
        if "MOMENTUM ASSESSMENT:" in llm_response:
            momentum_section = llm_response.split("MOMENTUM ASSESSMENT:")[1].split("RISK ASSESSMENT:")[0]
            insights["momentum_assessment"] = momentum_section.strip()
            
        # Parse risk assessment
        if "RISK ASSESSMENT:" in llm_response:
            risk_section = llm_response.split("RISK ASSESSMENT:")[1].split("RECOMMENDATION:")[0]
            insights["risk_assessment"] = risk_section.strip()
            
        # Parse recommendation
        if "RECOMMENDATION:" in llm_response:
            rec_section = llm_response.split("RECOMMENDATION:")[1]
            rec_lines = rec_section.strip().split("\n")
            
            if rec_lines:
                first_line = rec_lines[0]
                
                # Extract recommendation
                if "BUY" in first_line:
                    insights["recommendation"] = "BUY"
                elif "SELL" in first_line:
                    insights["recommendation"] = "SELL"
                else:
                    insights["recommendation"] = "HOLD"
                    
                # Extract confidence
                if "Confidence: HIGH" in first_line or "Confidence: High" in first_line:
                    insights["confidence"] = "HIGH"
                elif "Confidence: MEDIUM" in first_line or "Confidence: Medium" in first_line:
                    insights["confidence"] = "MEDIUM"
                else:
                    insights["confidence"] = "LOW"
                    
                # Extract explanation
                if len(rec_lines) > 1:
                    insights["explanation"] = "\n".join(rec_lines[1:]).strip()
            
        return insights
        
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
        recent_events = self.procedural_memory.get_events_by_type("market_data_analysis")
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
        aspects = ["market_data_analysis", "historical_momentum"]
        relevant_aspects = [aspect for aspect in aspects if aspect in beliefs_update]
        
        if not beliefs_update or not relevant_aspects:
            return False
            
        # Create belief update prompt
        belief_text = ""
        for aspect in relevant_aspects:
            belief_text += f"{aspect.replace('_', ' ').title()}: {beliefs_update[aspect]}\n\n"
        
        prompt = f"""You are a quantitative market data analyst. You have received updated investment beliefs:

{belief_text}

Based on this feedback, update your approach to analyzing market data. Consider:
1. What specific technical indicators should you pay more attention to?
2. How should you interpret momentum signals differently?
3. What risk metrics are most valuable for trading decisions?

Provide a concise summary of how you will adjust your analysis approach.
"""
        # Get response from LLM
        response = self._get_llm_response(prompt)
        
        # Store updated belief in procedural memory
        belief_text = f"Updated Belief - Market Data Analysis: {belief_text}\n\nImplementation Plan: {response}"
        belief_embedding = self._generate_embedding(belief_text)
        
        memory_id = self.add_to_procedural_memory(
            belief_text,
            "belief_update",
            belief_embedding,
            importance=2.0  # High importance for belief updates
        )
        
        self.logger.info(f"Updated market data analysis beliefs with ID {memory_id}")
        return True