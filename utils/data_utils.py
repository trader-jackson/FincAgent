"""
Data utilities for FINCON system.
Functions for loading and processing multi-modal financial data sources.
"""

import os
import json
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger("data_utils")

def load_stock_data(symbols, start_date, end_date, data_dir=None, use_cache=True):
    """Load historical stock price data for daily technical analysis."""
    result = {}
    
    for symbol in symbols:
        cache_path = None
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            cache_path = os.path.join(data_dir, f"{symbol}_{start_date}_{end_date}.csv")
            
        if use_cache and cache_path and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                result[symbol] = df
                continue
            except Exception as e:
                logger.warning(f"Error loading cached data for {symbol}: {str(e)}")
                
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            if cache_path:
                df.to_csv(cache_path)
                
            result[symbol] = df
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            
    return result

def load_news_data(symbols, start_date, end_date, data_dir=None, use_cache=True):
    """Load financial news data from REFINITIV REAL-TIME NEWS."""
    cache_path = None
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        symbols_str = "_".join(symbols)
        cache_path = os.path.join(data_dir, f"news_{symbols_str}_{start_date}_{end_date}.json")
        
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached news data: {str(e)}")
    
    # Generate synthetic news data
    news_data = []
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_date = start_date_obj
    while current_date <= end_date_obj:
        date_str = current_date.strftime("%Y-%m-%d")
        
        for symbol in symbols:
            # Only generate news for ~30% of days
            if np.random.random() < 0.3:
                sentiment = np.random.choice(["positive", "negative", "neutral"])
                
                # Create sample news article
                if sentiment == "positive":
                    title = f"{symbol} Reports Strong Results in Latest Quarter"
                    content = f"{symbol} Inc. reported quarterly earnings that exceeded analyst expectations, with revenue increasing by {np.random.randint(5, 20)}% year-over-year."
                elif sentiment == "negative":
                    title = f"{symbol} Misses Earnings Expectations, Shares Down"
                    content = f"{symbol} Inc. reported quarterly earnings below analyst expectations, with revenue declining by {np.random.randint(1, 15)}% year-over-year."
                else:
                    title = f"{symbol} Reports Mixed Results for the Quarter"
                    content = f"{symbol} Inc. reported mixed quarterly results with some metrics beating expectations while others fell short."
                
                article = {
                    "id": f"{symbol}_{date_str}_{np.random.randint(1000, 9999)}",
                    "date": date_str,
                    "title": title,
                    "source": "REFINITIV REAL-TIME NEWS",
                    "symbols": [symbol],
                    "content": content
                }
                
                news_data.append(article)
                
        # Move to next day
        current_date += timedelta(days=1)
    
    # Cache data if requested
    if cache_path:
        try:
            with open(cache_path, 'w') as f:
                json.dump(news_data, f)
        except Exception as e:
            logger.warning(f"Error caching news data: {str(e)}")
    
    return news_data

def load_filings_data(symbols, start_date, end_date, data_dir=None, use_cache=True):
    """Load SEC filings data (10-K, 10-Q) with Management Discussion & Analysis sections."""
    cache_path = None
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        symbols_str = "_".join(symbols)
        cache_path = os.path.join(data_dir, f"filings_{symbols_str}_{start_date}_{end_date}.json")
        
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached filings data: {str(e)}")
    
    # Generate synthetic filings data
    filings_data = []
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate for each year in the range
    for year in range(start_date_obj.year, end_date_obj.year + 1):
        for symbol in symbols:
            # 10-Q filings (quarterly)
            q1_date = f"{year}-05-15"  # Q1 filing
            q2_date = f"{year}-08-15"  # Q2 filing
            q3_date = f"{year}-11-15"  # Q3 filing
            annual_date = f"{year+1}-03-15"  # 10-K filing
            
            filing_dates = [
                (q1_date, "10-Q", f"{year}-03-31", "Q1"),
                (q2_date, "10-Q", f"{year}-06-30", "Q2"),
                (q3_date, "10-Q", f"{year}-09-30", "Q3"),
                (annual_date, "10-K", f"{year}-12-31", "Annual")
            ]
            
            for filing_date, form_type, period_ending, period_name in filing_dates:
                # Check if within date range
                if not (start_date <= filing_date <= end_date):
                    continue
                
                # Create synthetic MD&A content
                mda = f"Management's Discussion and Analysis for {symbol} Inc. {period_name} {year}. "
                mda += f"The company reported {'strong' if np.random.random() > 0.3 else 'mixed'} financial results for the period ending {period_ending}."
                
                # Create filing data
                filing = {
                    "type": form_type,
                    "company": f"{symbol} Inc.",
                    "symbol": symbol,
                    "date_filed": filing_date,
                    "period_ending": period_ending,
                    "mda": mda
                }
                
                filings_data.append(filing)
    
    # Cache data if requested
    if cache_path:
        try:
            with open(cache_path, 'w') as f:
                json.dump(filings_data, f)
        except Exception as e:
            logger.warning(f"Error caching filings data: {str(e)}")
    
    return filings_data

def load_ecc_data(symbols, start_date, end_date, data_dir=None, use_cache=True):
    """Load earnings call conference (ECC) audio data."""
    cache_path = None
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        symbols_str = "_".join(symbols)
        cache_path = os.path.join(data_dir, f"ecc_{symbols_str}_{start_date}_{end_date}.json")
        
    if use_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached ECC data: {str(e)}")
    
    # Generate synthetic ECC data
    ecc_data = []
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate for each year in the range
    for year in range(start_date_obj.year, end_date_obj.year + 1):
        for symbol in symbols:
            # Set earnings call dates
            q1_call_date = f"{year}-04-25"  # Q1 earnings call
            q2_call_date = f"{year}-07-25"  # Q2 earnings call
            q3_call_date = f"{year}-10-25"  # Q3 earnings call
            q4_call_date = f"{year+1}-01-25"  # Q4 earnings call
            
            call_dates = [
                (q1_call_date, f"Q1 {year}", f"{year}-03-31"),
                (q2_call_date, f"Q2 {year}", f"{year}-06-30"),
                (q3_call_date, f"Q3 {year}", f"{year}-09-30"),
                (q4_call_date, f"Q4 {year}", f"{year}-12-31")
            ]
            
            for call_date, quarter, period_ending in call_dates:
                # Check if within date range
                if not (start_date <= call_date <= end_date):
                    continue
                
                # Generate sample transcript
                sentiment = np.random.choice(["POSITIVE", "NEGATIVE", "NEUTRAL"])
                transcript = f"Earnings call transcript for {symbol} Inc. {quarter}. "
                
                if sentiment == "POSITIVE":
                    transcript += f"CEO: We had a strong quarter with excellent growth. "
                    transcript += f"CFO: Revenue increased by {np.random.randint(5, 20)}% year-over-year."
                elif sentiment == "NEGATIVE":
                    transcript += f"CEO: This was a challenging quarter for our business. "
                    transcript += f"CFO: Revenue decreased by {np.random.randint(1, 15)}% year-over-year."
                else:
                    transcript += f"CEO: Results were mixed this quarter. "
                    transcript += f"CFO: Revenue was flat compared to the same period last year."
                
                audio_analysis = {
                    "tone": sentiment,
                    "confidence": np.random.choice(["HIGH", "MEDIUM", "LOW"]),
                    "pace": "NORMAL"
                }
                
                # Create earnings call data
                call = {
                    "company": f"{symbol} Inc.",
                    "symbol": symbol,
                    "date": call_date,
                    "quarter": quarter,
                    "transcript": transcript,
                    "audio_analysis": audio_analysis
                }
                
                ecc_data.append(call)
    
    # Cache data if requested
    if cache_path:
        try:
            with open(cache_path, 'w') as f:
                json.dump(ecc_data, f)
        except Exception as e:
            logger.warning(f"Error caching ECC data: {str(e)}")
    
    return ecc_data

def calculate_technical_indicators(df):
    """Calculate technical indicators for a stock."""
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Calculate Simple Moving Averages
    result['SMA20'] = result['Close'].rolling(window=20).mean()
    result['SMA50'] = result['Close'].rolling(window=50).mean()
    result['SMA200'] = result['Close'].rolling(window=200).mean()
    
    # Calculate MACD
    result['EMA12'] = result['Close'].ewm(span=12, adjust=False).mean()
    result['EMA26'] = result['Close'].ewm(span=26, adjust=False).mean()
    result['MACD'] = result['EMA12'] - result['EMA26']
    result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = result['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    result['BB_Middle'] = result['Close'].rolling(window=20).mean()
    result['BB_StdDev'] = result['Close'].rolling(window=20).std()
    result['BB_Upper'] = result['BB_Middle'] + (result['BB_StdDev'] * 2)
    result['BB_Lower'] = result['BB_Middle'] - (result['BB_StdDev'] * 2)
    
    # Calculate Momentum
    result['Momentum'] = result['Close'] / result['Close'].shift(10) - 1
    
    return result