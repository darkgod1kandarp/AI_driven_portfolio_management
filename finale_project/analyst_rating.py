#!/usr/bin/env python3
"""
Download analyst upgrade/downgrade ratings from Yahoo Finance and convert to numeric scores.

Tickers: AAPL, TSLA, GME, AMC, AMZN, MSFT, NVDA, AMD, PLTR, BB, GOOGL, META, NFLX
Date range: 2015-01-01 to 2020-12-31

Fixes:
- META → FB alias (Facebook renamed to Meta in 2021)
- Handles missing tickers gracefully

Output: analyst_ratings.csv (with numeric sentiment_score column)
"""

import pandas as pd  
import yfinance as yf  
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
TICKERS = [
    "AAPL", "TSLA", "GME", "AMC", "AMZN", "MSFT", "NVDA", "AMD",
    "PLTR", "BB", "GOOGL", "META", "NFLX", "SPY", "QQQ"
]

# ETFs don't have analyst ratings
ETFS = ["SPY", "QQQ"]

# Ticker aliases for historical data (company name changes)
TICKER_ALIASES = {
    "META": "FB",   # Facebook → Meta (renamed Oct 2021)
    # Add more aliases if needed:
    # "GOOGL": "GOOG",  # If GOOGL doesn't work
}

START_DATE = "2015-01-01"
END_DATE = "2020-12-31"

OUTPUT_PATH = r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\analyst_ratings.csv"


# ============================================================
# GRADE TO NUMERIC MAPPING
# ============================================================
GRADE_MAPPING = {
    # Bullish (+1.0 to +0.75)
    "Strong Buy": 1.0,
    "Buy": 1.0,
    "Outperform": 0.75,
    "Overweight": 0.75,
    "Accumulate": 0.75,
    "Add": 0.75,
    "Positive": 0.75,
    "Long-Term Buy": 1.0,
    "Top Pick": 1.0,
    "Conviction Buy": 1.0,
    "Sector Outperform": 0.75,
    "Market Outperform": 0.75,
    
    # Neutral (0.0)
    "Hold": 0.0,
    "Neutral": 0.0,
    "Equal-Weight": 0.0,
    "Equal Weight": 0.0,
    "Market Perform": 0.0,
    "Sector Perform": 0.0,
    "In-Line": 0.0,
    "Peer Perform": 0.0,
    "Fair Value": 0.0,
    "Mixed": 0.0,
    "Sector Weight": 0.0,
    "Market Weight": 0.0,
    
    # Bearish (-0.75 to -1.0)
    "Sell": -1.0,
    "Strong Sell": -1.0,
    "Underperform": -0.75,
    "Underweight": -0.75,
    "Reduce": -0.75,
    "Negative": -0.75,
    "Avoid": -1.0,
    "Sector Underperform": -0.75,
    "Market Underperform": -0.75,
}


def grade_to_score(grade: str) -> float:
    """Convert analyst grade string to numeric score."""
    if pd.isna(grade) or grade == "":
        return 0.0  # Unknown = neutral
    
    # Try exact match first
    if grade in GRADE_MAPPING:
        return GRADE_MAPPING[grade]
    
    # Try case-insensitive match
    grade_lower = grade.lower().strip()
    for key, value in GRADE_MAPPING.items():
        if key.lower() == grade_lower:
            return value
    
    # Keyword-based fallback
    if any(word in grade_lower for word in ["buy", "outperform", "overweight", "positive", "accumulate"]):
        return 0.75
    elif any(word in grade_lower for word in ["sell", "underperform", "underweight", "negative", "reduce"]):
        return -0.75
    elif any(word in grade_lower for word in ["hold", "neutral", "equal", "perform", "inline"]):
        return 0.0
    
    # Unknown grade - log it
    print(f"    ⚠ Unknown grade: '{grade}' → defaulting to 0.0")
    return 0.0


def download_analyst_ratings(ticker: str, display_ticker: str = None) -> pd.DataFrame:
    """
    Download analyst ratings for a single ticker.
    
    Args:
        ticker: The ticker to download from Yahoo Finance
        display_ticker: The ticker name to use in output (for aliases like FB→META)
    """
    if display_ticker is None:
        display_ticker = ticker
        
    print(f"  Fetching {display_ticker}", end="")
    if ticker != display_ticker:
        print(f" (using {ticker})", end="")
    print("...", end=" ")
    
    try:
        stock = yf.Ticker(ticker)
        ratings = stock.upgrades_downgrades  
        if ticker
        
        if ratings is None or ratings.empty:
            print("⚠ No ratings found")
            return None  
        
        ratings = ratings.reset_index()
        ratings = ratings.rename(columns={
            "GradeDate": "date", 
            "Firm": "firm", 
            "ToGrade": "to_grade", 
            "FromGrade": "from_grade",
            "Action": "action"
        })
        
        # Use display ticker (e.g., META instead of FB)
        ratings["ticker"] = display_ticker
        
        # Convert date to datetime and filter to date range
        ratings["date"] = pd.to_datetime(ratings["date"])
        ratings = ratings[
            (ratings["date"] >= START_DATE) & 
            (ratings["date"] <= END_DATE)
        ]
        
        if ratings.empty:
            print("⚠ No ratings in date range")
            return None
        
        print(f"✓ {len(ratings)} ratings")
        return ratings
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    
def main():
    print("=" * 60)
    print("ANALYST RATINGS DOWNLOADER (with META→FB fix)")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Tickers: {len(TICKERS)} (excluding ETFs: {ETFS})")
    print("=" * 60)
    
    # Step 1: Download all ratings
    print("\n[1/3] Downloading from Yahoo Finance...")
    all_ratings = []
    success_count = 0
    fail_count = 0
    
    for ticker in TICKERS:
        if ticker in ETFS:  
            print(f"  Skipping {ticker} (ETF - no analyst ratings)")
            continue
        
        
        df = download_analyst_ratings(actual_ticker, display_ticker=ticker)
        if df is not None:
            all_ratings.append(df)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed/empty")
    
    if not all_ratings:
        print("\n❌ No ratings data collected!")
        return
    
    # Step 2: Combine all data
    print("\n[2/3] Combining data...")
    df_final = pd.concat(all_ratings, ignore_index=True)
    df_final = df_final.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Step 3: Convert grades to numeric scores
    print("\n[3/3] Converting grades to numeric scores...")
    df_final["sentiment_score"] = df_final["to_grade"].apply(grade_to_score)
    
    # Reorder columns
    column_order = ["date", "ticker", "firm", "to_grade", "from_grade", "action", "sentiment_score"]
    df_final = df_final[column_order]
    
    # Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"✓ SAVED: {OUTPUT_PATH}")
    print(f"  Total ratings: {len(df_final):,}")
    print(f"  Date range: {df_final['date'].min().date()} to {df_final['date'].max().date()}")
    print(f"  Tickers with data: {df_final['ticker'].nunique()}")
    print(f"{'=' * 60}")
    
    # Ratings per ticker
    print("\nRatings per ticker:")
    ticker_counts = df_final.groupby("ticker").size().sort_values(ascending=False)
    print(ticker_counts.to_string())
    
    # Score distribution
    print("\nSentiment score distribution:")
    score_counts = df_final["sentiment_score"].value_counts().sort_index()
    print(score_counts.to_string())
    
    # Average score per ticker
    print("\nAverage sentiment score per ticker:")
    avg_scores = df_final.groupby("ticker")["sentiment_score"].mean().round(2).sort_values(ascending=False)
    print(avg_scores.to_string())
    
    # Unique grades found
    print("\nUnique grades found:")
    unique_grades = df_final["to_grade"].value_counts().head(20)
    print(unique_grades.to_string())
    
    # Preview
    print("\nPreview (first 10 rows):")
    print(df_final.head(10).to_string(index=False))
    
    # Check for any missing tickers
    expected = set(TICKERS) - set(ETFS)
    actual = set(df_final["ticker"].unique())
    missing = expected - actual
    if missing:
        print(f"\n⚠ Tickers with no data: {missing}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()