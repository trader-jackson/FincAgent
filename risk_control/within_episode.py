"""
Within-episode risk control for FINCON system.
Implements CVaR-based risk monitoring and alerts.
"""

import logging
import numpy as np
from datetime import datetime

class WithinEpisodeRiskControl:
    """
    Within-episode risk control component for FINCON.
    
    Monitors daily investment risk through Conditional Value at Risk (CVaR)
    and issues alerts when significant risk increases are detected.
    """
    
    def __init__(self, confidence_level=0.01, threshold_decline=0.05):
        """
        Initialize within-episode risk control.
        
        Args:
            confidence_level (float): Confidence level for CVaR calculation (default: 0.01 for 1%)
            threshold_decline (float): Threshold for CVaR decline to trigger alert (default: 0.05 for 5%)
        """
        self.confidence_level = confidence_level
        self.threshold_decline = threshold_decline
        self.pnl_history = []
        self.cvar_history = []
        self.logger = logging.getLogger("within_episode_risk_control")
        
    def calculate_cvar(self, pnl_series):
        """
        Calculate Conditional Value at Risk (CVaR) from PnL series.
        
        Args:
            pnl_series (list): List of daily Profit and Loss values
            
        Returns:
            float: CVaR value
        """
        if not pnl_series or len(pnl_series) < 2:
            return 0.0
            
        # Convert to numpy array
        pnl_array = np.array(pnl_series)
        
        # Sort in ascending order (worst to best)
        sorted_pnl = np.sort(pnl_array)
        
        # Calculate the index for VaR (Value at Risk)
        var_index = int(len(sorted_pnl) * self.confidence_level)
        
        # Ensure we have at least one value
        var_index = max(1, var_index)
        
        # Calculate CVaR as the average of the worst var_index values
        cvar = np.mean(sorted_pnl[:var_index])
        
        return cvar
        
    def update_pnl(self, daily_pnl):
        """
        Update PnL history and recalculate CVaR.
        
        Args:
            daily_pnl (float): Daily Profit and Loss value
            
        Returns:
            float: Updated CVaR value
        """
        # Add daily PnL to history
        self.pnl_history.append(daily_pnl)
        
        # Calculate CVaR
        cvar = self.calculate_cvar(self.pnl_history)
        
        # Add CVaR to history
        self.cvar_history.append(cvar)
        
        self.logger.debug(f"Updated PnL: {daily_pnl}, CVaR: {cvar}")
        return cvar
        
    def check_risk_alert(self):
        """
        Check if a risk alert should be triggered based on CVaR decline.
        
        Returns:
            dict: Risk alert data if triggered, None otherwise
        """
        if len(self.cvar_history) < 2:
            return None
            
        # Get current and previous CVaR
        current_cvar = self.cvar_history[-1]
        previous_cvar = self.cvar_history[-2]
        
        # Calculate percentage change
        # If previous CVaR is close to zero or positive, use absolute difference
        if abs(previous_cvar) < 1e-6 or previous_cvar > 0:
            change = previous_cvar - current_cvar
            percentage_change = change
        else:
            # Calculate percentage decline
            percentage_change = (previous_cvar - current_cvar) / abs(previous_cvar)
            
        # Check if decline exceeds threshold and CVaR is negative
        if percentage_change > self.threshold_decline and current_cvar < 0:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "CVaR_DECLINE",
                "current_cvar": current_cvar,
                "previous_cvar": previous_cvar,
                "percentage_change": percentage_change,
                "level": self._determine_alert_level(percentage_change),
                "description": (
                    f"Significant decline in Conditional Value at Risk (CVaR) detected. "
                    f"CVaR declined from {previous_cvar:.4f} to {current_cvar:.4f}, "
                    f"representing a {percentage_change*100:.2f}% change."
                )
            }
            
            self.logger.warning(f"Risk alert triggered: CVaR declined by {percentage_change*100:.2f}%")
            return alert
            
        return None
        
    def _determine_alert_level(self, percentage_change):
        """
        Determine alert level based on percentage change.
        
        Args:
            percentage_change (float): Percentage change in CVaR
            
        Returns:
            str: Alert level (LOW, MEDIUM, HIGH)
        """
        if percentage_change > 0.20:
            return "HIGH"
        elif percentage_change > 0.10:
            return "MEDIUM"
        else:
            return "LOW"
            
    def generate_risk_recommendations(self, alert, current_positions):
        """
        Generate risk mitigation recommendations based on alert.
        
        Args:
            alert (dict): Risk alert data
            current_positions (dict): Current trading positions
            
        Returns:
            list: Risk mitigation recommendations
        """
        recommendations = []
        
        # Default recommendations
        recommendations.append("Consider more conservative position sizing for new trades")
        recommendations.append("Review current positions to identify largest risk contributors")
        
        # Add specific recommendations based on alert level
        if alert["level"] == "HIGH":
            recommendations.append("Consider reducing positions in high-volatility assets")
            recommendations.append("Temporarily pause opening new positions until risk stabilizes")
            
        elif alert["level"] == "MEDIUM":
            recommendations.append("Consider implementing tighter stop-loss levels")
            recommendations.append("Reduce position sizes for new trades by 25-50%")
            
        # Add position-specific recommendations
        if current_positions:
            for symbol, position in current_positions.items():
                # For large negative positions
                if position < -0.5:
                    recommendations.append(f"Consider reducing short position in {symbol}")
                # For large positive positions
                elif position > 0.5:
                    recommendations.append(f"Consider reducing long position in {symbol}")
                    
        return recommendations
        
    def reset(self):
        """Reset risk control for a new episode."""
        self.pnl_history = []
        self.cvar_history = []
        self.logger.info("Reset within-episode risk control")
        
    def to_dict(self):
        """
        Convert risk control state to dictionary.
        
        Returns:
            dict: Risk control state
        """
        return {
            "confidence_level": self.confidence_level,
            "threshold_decline": self.threshold_decline,
            "pnl_history": self.pnl_history,
            "cvar_history": self.cvar_history
        }
        
    def from_dict(self, data):
        """
        Load risk control state from dictionary.
        
        Args:
            data (dict): Risk control state
        """
        self.confidence_level = data.get("confidence_level", self.confidence_level)
        self.threshold_decline = data.get("threshold_decline", self.threshold_decline)
        self.pnl_history = data.get("pnl_history", [])
        self.cvar_history = data.get("cvar_history", [])