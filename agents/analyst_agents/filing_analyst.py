"""
Filing analyst agent for FINCON.
Extracts investment insights from SEC filings (10-K, 10-Q).
"""

import logging
from datetime import datetime
import json
import re

from agents.base_agent import BaseAgent

class FilingAnalyst(BaseAgent):
    """
    Filing analyst agent for FINCON.
    
    Specializes in extracting investment insights from SEC filings (10-K and 10-Q reports).
    Focuses on financial statement analysis and management discussions.
    """
    
    def __init__(self, agent_id, target_symbols):
        """
        Initialize filing analyst agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list): List of stock symbols to analyze
        """
        super().__init__(agent_id, "filing_analyst", target_symbols)
        
        # Track processed filing IDs to avoid duplicates
        self.processed_filing_ids = set()
        
    def process(self, filings_data):
        """
        Process SEC filings data and extract insights.
        
        Args:
            filings_data (list): List of SEC filings
            
        Returns:
            dict: Extracted insights and analysis
        """
        if not filings_data:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No filing data provided.",
                "key_insights": [],
                "recommendations": {}
            }
        
        # Filter filings related to target symbols
        relevant_filings = []
        for filing in filings_data:
            # Check if filing is already processed
            filing_id = filing.get("id", filing.get("type", "") + filing.get("date_filed", ""))
            if filing_id in self.processed_filing_ids:
                continue
                
            # Check if filing is relevant to any target symbol
            if filing.get("symbol") in self.target_symbols:
                relevant_filings.append(filing)
                self.processed_filing_ids.add(filing_id)
                
        # If no relevant filings, return empty insights
        if not relevant_filings:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": "No relevant filings found for target symbols.",
                "key_insights": [],
                "recommendations": {}
            }
            
        # Store relevant filings in working memory
        for filing in relevant_filings:
            self.working_memory.add_item(filing, "sec_filing")
            
        # Build prompt for LLM
        prompt = self._build_analysis_prompt(relevant_filings)
        
        # Get analysis from LLM
        llm_response = self._get_llm_response(prompt)
        
        # Parse analysis
        try:
            analysis = self._parse_filing_analysis(llm_response, relevant_filings)
        except Exception as e:
            self.logger.error(f"Error parsing filing analysis: {str(e)}")
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "symbols": self.target_symbols,
                "message": f"Error parsing filing analysis: {str(e)}",
                "key_insights": [],
                "recommendations": {}
            }
            
        # Store analysis in procedural memory
        analysis_text = json.dumps(analysis)
        analysis_embedding = self._generate_embedding(analysis_text)
        memory_id = self.add_to_procedural_memory(
            analysis_text,
            "filing_analysis",
            analysis_embedding
        )
                
        self.logger.info(f"Generated filing analysis with ID {memory_id}")
        return analysis
        
    def _build_analysis_prompt(self, filings):
        """
        Build prompt for filing analysis.
        
        Args:
            filings (list): List of SEC filings
            
        Returns:
            str: Analysis prompt
        """
        prompt = f"""You are a financial filing analyst specializing in analyzing SEC filings such as 10-K and 10-Q reports.

Your task is to analyze the following SEC filings related to these stocks: {', '.join(self.target_symbols)}

Please extract key insights, financial information, and investment implications from these filings.

SEC Filings:
"""

        # Add filings
        for i, filing in enumerate(filings, 1):
            prompt += f"\nFILING {i}:\n"
            prompt += f"Type: {filing.get('type', 'Unknown')}\n"
            prompt += f"Company: {filing.get('company', 'Unknown')}\n"
            prompt += f"Symbol: {filing.get('symbol', 'Unknown')}\n"
            prompt += f"Date Filed: {filing.get('date_filed', 'Unknown')}\n"
            prompt += f"Period Ending: {filing.get('period_ending', 'Unknown')}\n"
            prompt += "Management's Discussion & Analysis:\n"
            prompt += f"{filing.get('mda', 'No MD&A content available')}\n"
            
        # Add instructions
        prompt += """
Based on these filings, please provide:

1. Key Insights: Extract the most important insights from these filings (bullet points)
2. Financial Performance: Summarize the company's financial performance
3. Management Analysis: Extract significant points from management's discussion
4. Risk Factors: Identify key risks mentioned in the filings
5. Forward-Looking Statements: Highlight important forward-looking statements
6. Recommendation: Suggest potential trading implications based on these filings

Format your response using the following structure:

KEY INSIGHTS:
- [First key insight]
- [Second key insight]
...

FINANCIAL PERFORMANCE:
[Summary of financial performance]

MANAGEMENT ANALYSIS:
[Summary of management's discussion]

RISK FACTORS:
- [First risk factor]
- [Second risk factor]
...

FORWARD-LOOKING STATEMENTS:
- [First forward-looking statement]
- [Second forward-looking statement]
...

RECOMMENDATIONS:
[Symbol]: [POSITIVE/NEGATIVE/NEUTRAL]
[Brief explanation of recommendation]
...

Please ensure your analysis is objective, balanced, and directly based on the content of the filings.
"""

        return prompt
        
    def _parse_filing_analysis(self, llm_response, filings):
        """
        Parse filing analysis from LLM response.
        
        Args:
            llm_response (str): LLM response text
            filings (list): Original filings
            
        Returns:
            dict: Parsed filing analysis
        """
        # Initialize analysis structure
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.target_symbols,
            "source_filings": len(filings),
            "key_insights": [],
            "financial_performance": "",
            "management_analysis": "",
            "risk_factors": [],
            "forward_looking_statements": [],
            "recommendations": {}
        }
        
        # Parse key insights
        if "KEY INSIGHTS:" in llm_response:
            insights_section = llm_response.split("KEY INSIGHTS:")[1].split("FINANCIAL PERFORMANCE:")[0]
            insights_bullets = [line.strip()[2:].strip() for line in insights_section.strip().split("\n") if line.strip().startswith("-")]
            analysis["key_insights"] = insights_bullets
            
        # Parse financial performance
        if "FINANCIAL PERFORMANCE:" in llm_response:
            perf_section = llm_response.split("FINANCIAL PERFORMANCE:")[1].split("MANAGEMENT ANALYSIS:")[0]
            analysis["financial_performance"] = perf_section.strip()
            
        # Parse management analysis
        if "MANAGEMENT ANALYSIS:" in llm_response:
            mgmt_section = llm_response.split("MANAGEMENT ANALYSIS:")[1].split("RISK FACTORS:")[0]
            analysis["management_analysis"] = mgmt_section.strip()
            
        # Parse risk factors
        if "RISK FACTORS:" in llm_response:
            risk_section = llm_response.split("RISK FACTORS:")[1].split("FORWARD-LOOKING STATEMENTS:")[0]
            risk_bullets = [line.strip()[2:].strip() for line in risk_section.strip().split("\n") if line.strip().startswith("-")]
            analysis["risk_factors"] = risk_bullets
            
        # Parse forward-looking statements
        if "FORWARD-LOOKING STATEMENTS:" in llm_response:
            fls_section = llm_response.split("FORWARD-LOOKING STATEMENTS:")[1].split("RECOMMENDATIONS:")[0]
            fls_bullets = [line.strip()[2:].strip() for line in fls_section.strip().split("\n") if line.strip().startswith("-")]
            analysis["forward_looking_statements"] = fls_bullets
            
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
        recent_events = self.procedural_memory.get_events_by_type("filing_analysis")
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
        if not beliefs_update or "filing_information" not in beliefs_update:
            return False
            
        # Create belief update prompt
        new_belief = beliefs_update["filing_information"]
        
        prompt = f"""You are a financial filing analyst. You have received updated investment beliefs regarding SEC filing analysis:

{new_belief}

Based on this feedback, update your approach to analyzing SEC filings. Consider:
1. What specific aspects of filings should you pay more attention to?
2. How should you interpret financial metrics differently?
3. What parts of filings are most valuable for investment decisions?

Provide a concise summary of how you will adjust your analysis approach.
"""
        # Get response from LLM
        response = self._get_llm_response(prompt)
        
        # Store updated belief in procedural memory
        belief_text = f"Updated Belief - Filing Analysis: {new_belief}\n\nImplementation Plan: {response}"
        belief_embedding = self._generate_embedding(belief_text)
        
        memory_id = self.add_to_procedural_memory(
            belief_text,
            "belief_update",
            belief_embedding,
            importance=2.0  # High importance for belief updates
        )
        
        self.logger.info(f"Updated filing analysis beliefs with ID {memory_id}")
        return True