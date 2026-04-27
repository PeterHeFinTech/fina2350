import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


ESTIMATION_LOOKBACK = 252  

BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
PRICE_DIR = Path("/Users/lin/Desktop/fetch/category C/nvda_price_data")

NEWS_FILE = BASE_DIR / "category_C_news_cleaned.csv"
NVDA_FILE = PRICE_DIR / "NVDA_5Y.csv"
QQQ_FILE = PRICE_DIR / "QQQ_5Y.csv"
OUT_FILE = BASE_DIR / "category_C_car_sentiment.csv"


def load_data(path, name):
    try:
        df = pd.read_csv(path)
       
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df[name] = pd.to_numeric(df["Close"], errors="coerce")
        return df[["Date", name]].dropna()
    except Exception as e:
        print(f" Error loading {path.name}: {e}")
        return pd.DataFrame()



def run_analysis():
    print(" Starting Category C CAR calculation...")

    nvda = load_data(NVDA_FILE, "p_nvda")
    qqq = load_data(QQQ_FILE, "p_qqq")
    
    if nvda.empty or qqq.empty:
        print(" FAILED: Price CSV files not found or empty. Please check your folder.")
        return

    merged = pd.merge(nvda, qqq, on="Date").sort_values("Date")
    merged["r_nvda"] = merged["p_nvda"].pct_change()
    merged["r_qqq"] = merged["p_qqq"].pct_change()
    merged = merged.dropna().reset_index(drop=True)
    
    try:
        news = pd.read_csv(NEWS_FILE)
        news["publish_date"] = pd.to_datetime(news["publish_date"])
    except Exception as e:
        print(f" FAILED: News CSV not found: {e}")
        return
    
    results = []
    
    for _, row in news.iterrows():
        event_date = row["publish_date"].normalize()
    
        idx_list = merged.index[merged["Date"] == event_date].tolist()
        if not idx_list: 
            pos = merged["Date"].searchsorted(event_date)
            if pos >= len(merged): continue
            t0_idx = pos
        else:
            t0_idx = idx_list[0]
        
        hist = merged.iloc[max(0, t0_idx - ESTIMATION_LOOKBACK):t0_idx]
        if len(hist) < 30: continue
        
        beta = np.cov(hist["r_nvda"], hist["r_qqq"])[0, 1] / np.var(hist["r_qqq"])
        alpha = hist["r_nvda"].mean() - beta * hist["r_qqq"].mean()
        
        try:
            window = merged.iloc[t0_idx-1 : t0_idx+3].copy()
            if len(window) < 4: continue
            
            window["AR"] = window["r_nvda"] - (alpha + beta * window["r_qqq"])
            ar = window["AR"].values
            
            results.append({
                "CAR-1": ar[0],                  
                "CAR0":  ar[0:2].sum(),           
                "CAR1":  ar[0:3].sum(),            
                "CAR2":  ar[0:4].sum(),           
                "sentiment": row["summary_sentiment_score"]
            })
        except:
            continue


    if results:
        pd.DataFrame(results).to_csv(OUT_FILE, index=False)
        print(f"SUCCESS: Linked {len(results)} samples. Output: {OUT_FILE}")
    else:
        print("FAILED: No matches found between news and trading dates.")

if __name__ == "__main__":
    run_analysis()
