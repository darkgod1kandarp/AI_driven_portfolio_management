"""
Getting stock meta data such as historical prices, technical indicators, and analyst ratings. This data will be used to create a master dataset for modeling stock price movements. The script downloads data from Yahoo Finance and processes it to compute technical indicators and fill in analyst ratings over time.
"""

import pandas as pd  
import yfinance as yf  
import pandas_ta as ta 
from datetime import datetime  

TICKERS = [
    "AAPL", "TSLA", "GME",  "AMC",  "AMZN",
    "MSFT", "NVDA", "AMD",  "PLTR", "BB",
    "GOOGL","META", "NFLX", "SPY",  "QQQ"
]  



START_DATE = "2014-01-01"   
END_DATE = "2020-12-31"   
FINAL_START = "2015-01-01"


def download_single_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data for a single ticker and calculate technical indicators."""   
    
    print(f"Downloading {ticker} data...")   
    
    try:  
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"Warning: No data found for {ticker}. Skipping.")
            return None 
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make Date a column
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        
        # Keep only OHLCV columns
        df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
        
        # Rename to lowercase
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        
        # Add ticker column
        df["ticker"] = ticker
        
        print(f"✓ {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None   
    
def compute_indicator(df:pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for a given DataFrame."""
    df = df.copy()  
    # RSI - Relative Strength Index (14 periods)
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    
    # MACD - Moving Average Convergence Divergence (12, 26, 9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"] = macd.iloc[:, 0]        # MACD line
        df["MACD_signal"] = macd.iloc[:, 1]  # Signal line
        df["MACD_hist"] = macd.iloc[:, 2]    # Histogram
    else:
        df["MACD"] = None
        df["MACD_signal"] = None
        df["MACD_hist"] = None   
    
     # SMA - Simple Moving Averages
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)
    
    # ATR - Average True Range (14 periods)
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    return df

def main(output_path: str):
    
    all_data = []
    for ticker in TICKERS:
        ticker_df = download_single_ticker(ticker, START_DATE, END_DATE)
        if ticker_df is not None:
                all_data.append(ticker_df)   
    
    print("Combining data...")
    processed_data = []   
    for df in all_data:
        ticker = df["ticker"].iloc[0]
        
        df = df.sort_values("date").reset_index(drop=True)
        df = compute_indicator(df)  
        
        df_filtered = df[df["date"] >= FINAL_START].copy()   
        
        processed_data.append(df_filtered)  
    
    df_final = pd.concat(processed_data, ignore_index=True)
    df_final = df_final.sort_values(["ticker", "date"]).reset_index(drop=True)   
    columns_order = ["date", "ticker", "open", "high", "low", "close", "volume",
                        "RSI_14", "MACD", "MACD_signal", "MACD_hist",
                        "SMA_50", "SMA_200", "ATR_14"]
    
    df_final = df_final[columns_order]    
    df_final.to_csv(output_path, index=False)   
    
    
    # Summary statistics
    print("\nRows per ticker:")
    print(df_final.groupby("ticker").size().to_string())
    
    print("\nMissing values per column:")
    print(df_final.isna().sum().to_string())   
        

if __name__ == "__main__":
    output_file = "meta_data/stock_prices_with_technical_indicators.csv"
    main(output_file)    

