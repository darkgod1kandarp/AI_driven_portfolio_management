"""
This script creates a master dataset by merging stock price data, analyst ratings, macroeconomic indicators, and sentiment scores from news and social media. The resulting dataset is saved as a CSV file for use in modeling and analysis.
"""


import pandas as pd  
import numpy as np
from pathlib import Path



DATA_DIR = Path("meta_data")


# Input files
STOCK_FILE = DATA_DIR / "stock_prices_with_technical_indicators.csv"   
ANALYST_FILE = DATA_DIR / "analyst_ratings.csv"   
MACRO_FILE = DATA_DIR / "macro_data_2015_2020.csv"   

# Sentimental files   
NEWS_SENTIMENTAL_FILE = DATA_DIR /  "financial_news_2015_2020_with_sentiment.csv"   
SOCIAL_MEDIA_SENTIMENTAL_FILE = DATA_DIR / "social_media_2015_2020_with_sentiment.csv"   

# Output file 
OUTPUT_FILE = DATA_DIR / "master_dataset.csv"   

# Tickers
TICKERS = [
        "AAPL","TSLA","GME","AMC","AMZN",
        "MSFT","NVDA","AMD","PLTR","BB",
        "GOOGL","META","NFLX","SPY","QQQ"
]

def load_stock_data(filepath: Path = STOCK_FILE) -> pd.DataFrame:
    """Load stock price and technical indicator data."""
    df = pd.read_csv(filepath, parse_dates=["date"])   
    expected_cols =  expected_cols = ["date", "ticker", "open", "high", "low", "close", "volume",
                     "RSI_14", "MACD", "SMA_50", "SMA_200", "ATR_14"]   

    missing =  set(expected_cols) - set(df.columns)   
    if missing:
        print(f"Warning: Missing columns in stock data: {missing}")
        
    return df
    
def load_and_fill_analyst_rating(filepath: Path, stock_df: pd.DataFrame) -> pd.DataFrame:
    
    print(f"Loading analyst ratings from {filepath}...")
    if not filepath.exists():
        print(f"Error: Analyst ratings file not found at {filepath}")
        return None  
    
    df = pd.read_csv(filepath, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()   # strip time component → midnight

    # Get all unique tickers from stock data
    all_dates = stock_df["date"].unique()
    all_tickers = stock_df["ticker"].unique()
    
    print("Filling analyst ratings for tickers:", all_tickers)
    
    filled_data = []   
    for ticker in all_tickers:
        
        ticker_ratings = df[df["ticker"] == ticker].sort_values("date")
        print(f"  Processing {ticker}: {len(ticker_ratings)} ratings found")
                
        # Create a complete date range for this ticker
        ticker_dates = pd.DataFrame({"date": all_dates})
        ticker_dates["ticker"] = ticker   
        
        print(ticker_ratings['sentiment_score'].sum())
        if len(ticker_ratings) == 0: 
            ticker_dates["analyst_sentiment"] = 0.0   
        else:   
            ticker_dates = ticker_dates.merge(
                ticker_ratings[["date", "sentiment_score"]],
                on="date",
                how="left"
            )
            
            print(ticker_dates["sentiment_score"].sum())           
            # Forward-fill: each rating persists until next one
            ticker_dates["analyst_sentiment"] = ticker_dates["sentiment_score"].ffill()
            # Fill any leading NaN with 0 (before first rating)
            ticker_dates["analyst_sentiment"] = ticker_dates["analyst_sentiment"].fillna(0.0)
            ticker_dates = ticker_dates.drop(columns=["sentiment_score"])
            print(f"    Filled {ticker_dates['analyst_sentiment'].isna().sum()} missing values")
        filled_data.append(ticker_dates) 
    
    return pd.concat(filled_data, ignore_index=True)  

 

def load_sentiment_data(filepath: Path, name:str, date_col: str = "date", 
                            ticker_col: str = "ticker", score_col: str = "sentiment_score") -> pd.DataFrame:
    if not filepath.exists():
        print(f"Error: Sentiment data file not found at {filepath}")
        return None
    
    df = pd.read_csv(filepath, parse_dates=[date_col])
    
    df = df.rename(columns={
        date_col: "date",
        ticker_col: "ticker",
        score_col: f"{name}_sentiment"
    })
    
    # Aggregate to daily mean per ticker
    daily = df.groupby(["date", "ticker"])[f"{name}_sentiment"].mean().reset_index()   
    daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize(None).dt.normalize()
    daily[f"{name}_sentiment"] = daily[f"{name}_sentiment"].fillna(0.0)
    return daily


def load_macro_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        print(f"Error: Macro data file not found at {filepath}")
        return None  
    
    df = pd.read_csv(filepath, parse_dates=["date"])
    
    expected_cols = ["date", "VIX", "FED_RATE", "CPI", "CPI_YOY"]   
    missing = set(expected_cols) - set(df.columns)   
    if missing:
        print(f"Warning: Missing columns in macro data: {missing}")
        
    return df

def build_master_dataset():   
    
     # Step 1: Load stock data (the base)
    stock_df  = load_stock_data(STOCK_FILE)      
    
    # Step 2: Load macro data
    macro_df = load_macro_data(MACRO_FILE)
    
    # Step 3: Load and fill analyst ratings    
    analyst_df = load_and_fill_analyst_rating(ANALYST_FILE, stock_df)
    
    
    news_df = load_sentiment_data(NEWS_SENTIMENTAL_FILE, "news") if NEWS_SENTIMENTAL_FILE else None
  
    
    stock_df = stock_df.merge(analyst_df, on=["date", "ticker"],  how="left")
    stock_df = stock_df.merge(news_df, on=["date", "ticker"],  how="left")
    stock_df['news_sentiment'] = stock_df['news_sentiment'].fillna(0.0)
    stock_df = stock_df.merge(macro_df, on="date", how="left")
    stock_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Master dataset saved to {OUTPUT_FILE}")
    




build_master_dataset()