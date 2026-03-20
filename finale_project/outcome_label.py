import numpy as np
import pandas as pd

INPUT_FILE = r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\master_dataset.csv"
OUTPUT_FILE = r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\master_dataset_with_labels.csv"

# Label thresholds
BUY_THRESHOLD = 0.02    # +2%
SHORT_THRESHOLD = -0.02  # -2%

def read_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

def calculate_return(group):  
    group = group.sort_values("date")
    group["return_1d"] = group["close"].pct_change(1).shift(-1)
    group["return_5d"] = group["close"].pct_change(5).shift(-5)
    group["return_30d"] = group["close"].pct_change(30).shift(-30)
    return group


def assign_label(return_30d):
    """Assign buy/hold/short label based on 30-day return."""
    if pd.isna(return_30d):
        return np.nan
    elif return_30d > BUY_THRESHOLD:
        return "buy"
    elif return_30d < SHORT_THRESHOLD:
        return "short"
    else:
        return "hold"
    
def main(df: pd.DataFrame) -> pd.DataFrame:
    # Ensuring sorting the data
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)      
    df = df.groupby("ticker").apply(calculate_return)
    df["label"] = df["return_30d"].apply(assign_label)
    df_clean = df.dropna(subset=["label"])
    return df


def graph(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x="label", data=df)
    plt.title("Distribution of Buy/Hold/Short Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

    
    
if __name__ == "__main__":
    main_df = read_data(INPUT_FILE)
    labeled_df = main(main_df)
    labeled_df.to_csv(OUTPUT_FILE, index=False)
    graph(labeled_df)
    
    

    
