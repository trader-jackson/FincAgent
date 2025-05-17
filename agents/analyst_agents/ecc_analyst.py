"""
ECC analyst agent for FINCON.
Extracts investment insights from earnings call conferences.
"""

import logging
from datetime import datetime
import json
import re

from agents.base_agent import BaseAgent

class ECCAnalyst(BaseAgent):
    """
    Earnings Call Conference (ECC) analyst agent for FINCON.
    
    Specializes in extracting investment insights from earnings call transcripts and audio analysis.
    Focuses on executive communications, tone analysis, and forward guidance.
    """
    
    def __init__(self, agent_id, target_symbols):
        """
        Initialize ECC analyst agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list): List of stock symbols to analyze
        """
        super().__init__(agent_id, "ecc_analyst", target_symbols)
        
        # Track processed call IDs to avoid duplicates
        self.processed_call_ids = set()
        
        # Track sentiment history
        self.sentiment_history = {symbol: [] for symbol in target_symbols}
        
    def process(self, ecc_data):
        """
        Process earnings call data and extract insights.
        
        Args:
            ecc_data (list): List of earnings call data
            
        Returns:
            dict: Extracted insights and analysis
        """
        if not ecc_data:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No earnings call data provided.",
                "key_insights": [],
                "recommendations": {}
            }
        
        # Filter calls related to target symbols
        relevant_calls = []
        for call in ecc_data:
            # Check if call is already processed
            call_id = call.get("id", call.get("symbol", "") + call.get("date", ""))
            if call_id in self.processed_call_ids:
                continue
                
            # Check if call is relevant to any target symbol
            if call.get("symbol") in self.target_symbols:
                relevant_calls.append(call)
                self.processed_call_ids.add(call_id)
                
        # If no relevant calls, return empty insights
        if not relevant_calls:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No relevant earnings calls found for target symbols.",
                "key_insights": [],
                "recommendations": {}
            }
            
        # Store relevant calls in working memory
        for call in relevant_calls:
            self.working_memory.add_item(call, "earnings_call")
            
        # Build prompt for LLM
        prompt = self._build_analysis_prompt(relevant_calls)
        
        # Get analysis from LLM
        llm_response = self._get_llm_response(prompt)
        
        # Parse analysis
        try:
            analysis = self._parse_ecc_analysis(llm_response, relevant_calls)
        except Exception as e:
            self.logger.error(f"Error parsing earnings call analysis: {str(e)}")
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": f"Error parsing earnings call analysis: {str(e)}",
                "key_insights": [],
                "recommendations": {}
            }
            
        # Store analysis in procedural memory
        analysis_text = json.dumps(analysis)
        analysis_embedding = self._generate_embedding(analysis_text)
        memory_id = self.add_to_procedural_memory(
            analysis_text,
            "ecc_analysis",
            analysis_embedding
        )
        
        # Update sentiment history
        for symbol, sentiment in analysis.get("sentiment", {}).items():
            if symbol in self.sentiment_history:
                self.sentiment_history[symbol].append({
                    "timestamp": analysis["timestamp"],
                    "sentiment": sentiment
                })
                
        self.logger.info(f"Generated earnings call analysis with ID {memory_id}")
        return analysis
        
    def _build_analysis_prompt(self, calls):
        """
        Build prompt for earnings call analysis.
        
        Args:
            calls (list): List of earnings calls
            
        Returns:
            str: Analysis prompt
        """
        prompt = f"""You are an earnings call analyst specializing in extracting investment insights from earnings conference calls.

Your task is to analyze the following earnings calls related to these stocks: {', '.join(self.target_symbols)}

Please extract key insights, forward guidance, and investment implications from these calls, considering both the content and tone.

Earnings Calls:
"""

        # Add calls
        for i, call in enumerate(calls, 1):
            prompt += f"\nCALL {i}:\n"
            prompt += f"Company: {call.get('company', 'Unknown')}\n"
            prompt += f"Symbol: {call.get('symbol', 'Unknown')}\n"
            prompt += f"Date: {call.get('date', 'Unknown')}\n"
            prompt += f"Quarter: {call.get('quarter', 'Unknown')}\n"
            
            # Add audio analysis if available
            if "audio_analysis" in call:
                prompt += "Audio Analysis:\n"
                for key, value in call["audio_analysis"].items():
                    prompt += f"- {key}: {value}\n"
            
            prompt += "Transcript:\n"
            prompt += f"{call.get('transcript', 'No transcript available')}\n"
            
        # Add instructions
        prompt += """
Based on these earnings calls, please provide:

1. Key Highlights: Extract the most important points from the earnings calls (bullet points)
2. Financial Results: Summarize the financial results discussed in the calls
3. Guidance and Outlook: Extract forward guidance and management outlook
4. Audio Analysis Insights: Provide insights based on the tone, confidence, and speaking patterns
5. Q&A Insights: Extract notable questions and answers from the Q&A sections
6. Sentiment Analysis: Determine the sentiment (POSITIVE, NEGATIVE, or NEUTRAL) for each company
7. Recommendations: Suggest potential trading implications based on these calls

Format your response using the following structure:

KEY HIGHLIGHTS:
- [First highlight]
- [Second highlight]
...

FINANCIAL RESULTS:
[Summary of financial results]

GUIDANCE AND OUTLOOK:
[Summary of guidance and outlook]

AUDIO ANALYSIS INSIGHTS:
[Insights from tone, confidence, and speaking patterns]

Q&A INSIGHTS:
[Notable questions and answers]

SENTIMENT ANALYSIS:
[SYMBOL]: [SENTIMENT]
[Brief explanation of sentiment assessment]
...

RECOMMENDATIONS:
[SYMBOL]: [POSITIVE/NEGATIVE/NEUTRAL]
[Brief explanation of recommendation]
...

Please ensure your analysis is objective, balanced, and considers both the content and tone of the earnings calls.
"""

        return prompt
        
    def _parse_ecc_analysis(self, llm_response, calls):
        """
        Parse earnings call analysis from LLM response.
        
        Args:
            llm_response (str): LLM response text
            calls (list): Original earnings calls
            
        Returns:
            dict: Parsed earnings call analysis
        """
        # Initialize analysis structure
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.target_symbols,
            "source_calls": len(calls),
            "key_highlights": [],
            "financial_results": "",
            "guidance_outlook": "",
            "audio_insights": "",
            "qa_insights": "",
            "sentiment": {},
            "recommendations": {}
        }
        
        # Parse key highlights
        if "KEY HIGHLIGHTS:" in llm_response:
            highlights_section = llm_response.split("KEY HIGHLIGHTS:")[1].split("FINANCIAL RESULTS:")[0]
            highlights_bullets = [line.strip()[2:].strip() for line in highlights_section.strip().split("\n") if line.strip().startswith("-")]
            analysis["key_highlights"] = highlights_bullets
            
        # Parse financial results
        if "FINANCIAL RESULTS:" in llm_response:
            results_section = llm_response.split("FINANCIAL RESULTS:")[1].split("GUIDANCE AND OUTLOOK:")[0]
            analysis["financial_results"] = results_section.strip()
            
        # Parse guidance and outlook
        if "GUIDANCE AND OUTLOOK:" in llm_response:
            guidance_section = llm_response.split("GUIDANCE AND OUTLOOK:")[1].split("AUDIO ANALYSIS INSIGHTS:")[0]
            analysis["guidance_outlook"] = guidance_section.strip()
            
        # Parse audio analysis insights
        if "AUDIO ANALYSIS INSIGHTS:" in llm_response:
            audio_section = llm_response.split("AUDIO ANALYSIS INSIGHTS:")[1].split("Q&A INSIGHTS:")[0]
            analysis["audio_insights"] = audio_section.strip()
            
        # Parse Q&A insights
        if "Q&A INSIGHTS:" in llm_response:
            qa_section = llm_response.split("Q&A INSIGHTS:")[1].split("SENTIMENT ANALYSIS:")[0]
            analysis["qa_insights"] = qa_section.strip()
            
        # Parse sentiment analysis
        if "SENTIMENT ANALYSIS:" in llm_response:
            sentiment_section = llm_response.split("SENTIMENT ANALYSIS:")[1].split("RECOMMENDATIONS:")[0]
            
            # Process each line in sentiment section
            for line in sentiment_section.strip().split("\n"):
                if ":" in line and any(symbol in line for symbol in self.target_symbols):
                    try:
                        # Extract symbol
                        symbol_part = line.split(":")[0].strip()
                        
                        # Find which target symbol is in the line
                        symbol = next((s for s in self.target_symbols if s in symbol_part), None)
                        if not symbol:
                            continue
                            
                        # Extract sentiment
                        sentiment_part = line.split(":")[1].strip()
                        
                        sentiment_value = "NEUTRAL"  # Default
                        if "POSITIVE" in sentiment_part:
                            sentiment_value = "POSITIVE"
                        elif "NEGATIVE" in sentiment_part:
                            sentiment_value = "NEGATIVE"
                            
                        # Store sentiment
                        analysis["sentiment"][symbol] = {
                            "value": sentiment_value,
                            "explanation": sentiment_part
                        }
                    except Exception as e:
                        self.logger.warning(f"Error parsing sentiment line: {line}, Error: {str(e)}")
            
        # Parse recommendations
        if "RECOMMENDATIONS:" in llm_response:
            rec_section = llm_response.split("RECOMMENDATIONS:")[1]
            
            # Process each line in recommendation section
            for line in rec_section.strip().split("\n"):
                if ":" in line and any(symbol in line for symbol in self.target_symbols):
                    try:
                        # Extract symbol
                        symbol_part = line.split(":")[0].strip()
                        
                        # Find which target symbol is in the line
                        symbol = next((s for s in self.target_symbols if s in symbol_part), None)
                        if not symbol:
                            continue
                            
                        # Extract recommendation
                        rec_part = line.split(":")[1].strip()
                        
                        sentiment_value = "NEUTRAL"  # Default
                        if "POSITIVE" in rec_part:
                            sentiment_value = "POSITIVE"
                        elif "NEGATIVE" in rec_part:
                            sentiment_value = "NEGATIVE"
                            
                        # Store recommendation
                        analysis["recommendations"][symbol] = {
                            "value": sentiment_value,
                            "explanation": rec_part
                        }
                    except Exception as e:
                        self.logger.warning(f"Error parsing recommendation line: {line}, Error: {str(e)}")
            
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
        recent_events = self.procedural_memory.get_events_by_type("ecc_analysis")
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
        if not beliefs_update or "earnings_call_insights" not in beliefs_update:
            return False
            
        # Create belief update prompt
        new_belief = beliefs_update["earnings_call_insights"]
        
        prompt = f"""You are an earnings call analyst. You have received updated investment beliefs regarding earnings call analysis:

{new_belief}

Based on this feedback, update your approach to analyzing earnings calls. Consider:
1. What specific aspects of earnings calls should you pay more attention to?
2. How should you analyze the tone and confidence levels differently?
3. What parts of earnings calls are most valuable for investment decisions?

Provide a concise summary of how you will adjust your analysis approach.
"""
        # Get response from LLM
        response = self._get_llm_response(prompt)
        
        # Store updated belief in procedural memory
        belief_text = f"Updated Belief - Earnings Call Analysis: {new_belief}\n\nImplementation Plan: {response}"
        belief_embedding = self._generate_embedding(belief_text)
        
        memory_id = self.add_to_procedural_memory(
            belief_text,
            "belief_update",
            belief_embedding,
            importance=2.0  # High importance for belief updates
        )
        
        self.logger.info(f"Updated earnings call analysis beliefs with ID {memory_id}")
        return True
        
    def get_sentiment_trend(self, symbol, days=5):
        """
        Get sentiment trend for a specific symbol based on earnings calls.
        
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
                "sentiment_data": []
            }
            
        # Get most recent sentiment entries
        recent_entries = sorted(self.sentiment_history[symbol], key=lambda x: x["timestamp"], reverse=True)
        recent_entries = recent_entries[:days]
        
        if not recent_entries:
            return {
                "symbol": symbol,
                "sentiment_trend": "NEUTRAL",
                "sentiment_data": []
            }
            
        # Count sentiment values
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for entry in recent_entries:
            if "sentiment" in entry and "value" in entry["sentiment"]:
                sentiment_value = entry["sentiment"]["value"]
                sentiment_counts[sentiment_value] += 1
                
        # Determine overall trend
        if sentiment_counts["POSITIVE"] > sentiment_counts["NEGATIVE"] and sentiment_counts["POSITIVE"] > sentiment_counts["NEUTRAL"]:
            trend = "POSITIVE"
        elif sentiment_counts["NEGATIVE"] > sentiment_counts["POSITIVE"] and sentiment_counts["NEGATIVE"] > sentiment_counts["NEUTRAL"]:
            trend = "NEGATIVE"
        else:
            trend = "NEUTRAL"
            
        return {
            "symbol": symbol,
            "sentiment_trend": trend,
            "sentiment_data": [
                {
                    "timestamp": entry["timestamp"],
                    "value": entry["sentiment"]["value"] if "sentiment" in entry and "value" in entry["sentiment"] else "NEUTRAL"
                } 
                for entry in recent_entries
            ]
        }