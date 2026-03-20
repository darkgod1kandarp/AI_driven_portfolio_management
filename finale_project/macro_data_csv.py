"""
This script fetches macroeconomic data from the FRED API for the period 2015-2020, including the VIX index, 
Fed funds rate, and CPI index. The data is merged into a single DataFrame, with missing values forward-filled and backward-filled as needed.
A new column for year-over-year CPI change is calculated and added to the dataset. Finally, the cleaned and processed dataset is saved as a CSV file for use in further analysis.
"""

import pandas as pd  
import requests  
from datetime import datetime   

FRED_API_KEY = ""  


START_DATE = "2015-01-01"  
END_DATE = "2020-12-31"


SERIES = {
    "VIX": "VIXCLS",        # Daily volatility index
    "FED_RATE": "FEDFUNDS", # Monthly Fed funds rate
    "CPI": "CPIAUCSL"       # Monthly CPI index
}


OUTPUT_PATH = r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\macro_data_2015_2020.csv"  


def fetch_fred_data(series_id: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from FRED API for a given series ID and date range."""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    observations = data.get("observations", [])
    
    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    
    return df[["date", "value"]].copy()


def main():
    
    if not FRED_API_KEY:
        print("Error: Please set your FRED API key in the FRED_API_KEY variable.")
        return
    
    data_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')   
    df_master = pd.DataFrame({"date": data_range})  
    
    for col_name, series_id in SERIES.items():
        print(f"Fetching {col_name} data...")
        df_series = fetch_fred_data(series_id, FRED_API_KEY, START_DATE, END_DATE)
        df_series = df_series.rename(columns={"value": col_name})
        df_master = pd.merge(df_master, df_series, on="date", how="left")
    
    # Forward fill, then backward fill for any leading NaNs
    for col in ["VIX", "FED_RATE", "CPI"]:
        df_master[col] = df_master[col].ffill().bfill()
        
    df_master["CPI_YOY"] = (
        (df_master["CPI"] - df_master["CPI"].shift(365)) 
        / df_master["CPI"].shift(365) 
        * 100
    ).round(2)
    
    first_valid = df_master["CPI_YOY"].first_valid_index()
    if first_valid:
        first_value = df_master.loc[first_valid, "CPI_YOY"]
        df_master["CPI_YOY"] = df_master["CPI_YOY"].fillna(first_value)

    df_master.to_csv(OUTPUT_PATH, index=False)   
    
if __name__ == "__main__":
    main()
