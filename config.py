"""
Configuration file for FINCON system.
Contains parameters for training, testing, LLM settings, and agent configuration.
"""

import os
from datetime import datetime

# General settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Training and testing periods
TRAINING_START_DATE = "2022-01-03"
TRAINING_END_DATE = "2022-10-04"
TESTING_START_DATE = "2022-10-05"
TESTING_END_DATE = "2023-06-10"

# LLM settings
LLM_MODEL = "gpt-4-turbo"
LLM_TEMPERATURE = 0.3
LLM_BELIEF_TEMPERATURE = 0.0  # For consistent belief generation
LLM_MAX_TOKENS = 1024
LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# Memory settings
MAX_MEMORY_EVENTS = 5  # Number of top memory events retrieved for each agent
MEMORY_DECAY_RATES = {
    "news_analyst": 0.9,     # Daily decay rate for news (fast decay)
    "filing_analyst": 0.99,  # Decay rate for SEC filings (slow decay)
    "ecc_analyst": 0.97,     # Decay rate for earnings calls (medium decay)
    "data_analyst": 0.95,    # Decay rate for market data (medium-fast decay)
}

# Risk control settings
CVAR_CONFIDENCE_LEVEL = 0.01  # Bottom 1% of daily PnLs
CVAR_THRESHOLD_DECLINE = 0.05  # 5% decline in CVaR triggers risk alert

# Portfolio management settings
MAX_PORTFOLIO_SIZE = 10
PORTFOLIO_REBALANCE_FREQUENCY = "daily"

# Stocks for testing
SINGLE_STOCKS = [
    "TSLA", "AMZN", "NIO", "MSFT", 
    "AAPL", "GOOG", "NFLX", "COIN"
]

# Portfolios for testing
PORTFOLIOS = {
    "Portfolio1": ["TSLA", "MSFT", "PFE"],
    "Portfolio2": ["AMZN", "GM", "LLY"]
}

# Training settings
MAX_EPISODES = 4
INITIAL_INVESTMENT = 1000000  # $1M initial investment

# Evaluation metrics settings
RISK_FREE_RATE = 0.01  # 1% risk-free rate for Sharpe Ratio calculation

# File paths for data sources
STOCK_DATA_PATH = os.path.join(DATA_DIR, "stock_data")
NEWS_DATA_PATH = os.path.join(DATA_DIR, "news_data")
FILINGS_PATH = os.path.join(DATA_DIR, "filings")
ECC_AUDIO_PATH = os.path.join(DATA_DIR, "ecc_audio")

# Create necessary directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STOCK_DATA_PATH, exist_ok=True)
os.makedirs(NEWS_DATA_PATH, exist_ok=True)
os.makedirs(FILINGS_PATH, exist_ok=True)
os.makedirs(ECC_AUDIO_PATH, exist_ok=True)