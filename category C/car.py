import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =================================================================
# 1. GLOBAL CONFIGURATION
# =================================================================
ESTIMATION_LOOKBACK = 252 
RELEVANCE_THRESHOLD = 0.20

BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
PRICE_DIR = BASE_DIR / "nvda_price_data"

NEWS_FILE = BASE_DIR / "category_C_news_cleaned.csv"
NVDA_FILE = PRICE_DIR / "NVDA_5Y.csv"
QQQ_FILE = PRICE_DIR / "QQQ_5Y.csv"
OUT_FILE = BASE_DIR / "category_C_car_sentiment.csv"

# =================================================================
# 2. DATA LOADING UTILITIES
# =================================================================

def load_market_prices(file_path, ticker_symbol):
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
        df[ticker_symbol] = pd.to_numeric(df["Close"], errors="coerce")
        return df[["Date", ticker_symbol]].dropna()
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return pd.DataFrame()

# =================================================================
# 3. ANALYSIS PIPELINE
# =================================================================

def run_event_study():
    print("Status: Initializing Event Study with Automatic Aggregation...")

    # A. Load Price Data
    nvda = load_market_prices(NVDA_FILE, "p_nvda")
    qqq = load_market_prices(QQQ_FILE, "p_qqq")
    
    if nvda.empty or qqq.empty:
        print("Failure: Market data could not be loaded.")
        return

    market_data = pd.merge(nvda, qqq, on="Date").sort_values("Date")
    market_data["r_nvda"] = market_data["p_nvda"].pct_change()
    market_data["r_qqq"] = market_data["p_qqq"].pct_change()
    market_data = market_data.dropna().reset_index(drop=True)
    
    # B. Load News Data + Automatic Aggregation
    try:
        news = pd.read_csv(NEWS_FILE)
        news["publish_date"] = pd.to_datetime(news["publish_date"], dayfirst=True, errors='coerce')
        news = news.dropna(subset=["publish_date"])
        
        print(f"Original news items: {len(news)}")
        
        # === AUTOMATIC AGGREGATION ===
        # Keep only the news with the highest nvda_relevance for each event_date
        news = news.loc[news.groupby('publish_date')['nvda_relevance'].idxmax()].reset_index(drop=True)
        print(f"After aggregation: {len(news)} unique events")
        
    except Exception as e:
        print(f"Failure: News file error: {e}")
        return
    
    results = []
    skipped_count = 0
    
    # C. Event Loop
    print(f"Status: Processing {len(news)} events...")
    for _, row in news.iterrows():
        rel_score = row.get("nvda_relevance", 0)
        if rel_score < RELEVANCE_THRESHOLD:
            skipped_count += 1
            continue
            
        event_dt = row["publish_date"].normalize()
        
        trading_idx = market_data.index[market_data["Date"] == event_dt].tolist()
        if not trading_idx:
            pos = market_data["Date"].searchsorted(event_dt)
            if pos >= len(market_data): continue
            t0 = pos
        else:
            t0 = trading_idx[0]
        
        hist = market_data.iloc[max(0, t0 - ESTIMATION_LOOKBACK):t0]
        if len(hist) < 30: continue
        
        beta = np.cov(hist["r_nvda"], hist["r_qqq"])[0, 1] / np.var(hist["r_qqq"])
        alpha = hist["r_nvda"].mean() - beta * hist["r_qqq"].mean()
        
        try:
            window = market_data.iloc[t0-1 : t0+3].copy()
            if len(window) < 4: continue
            
            window["AR"] = window["r_nvda"] - (alpha + beta * window["r_qqq"])
            ar = window["AR"].values
            
            results.append({
                "event_date": event_dt.strftime('%Y-%m-%d'),
                "CAR_minus_1": ar[0],                  
                "CAR_0":  ar[0:2].sum(),           
                "CAR_1":  ar[0:3].sum(),            
                "CAR_2":  ar[0:4].sum(),
                "nvda_relevance": rel_score,
                "nvda_sentiment": row.get("nvda_sentiment_score", 0),
                "overall_sentiment": row.get("overall_sentiment_score", 0),
                "policy_type": row.get("policy_shock_type", "N/A")
            })
        except:
            continue

    # D. Output Generation
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv(OUT_FILE, index=False)
        print(f"Success! Filtered out {skipped_count} low-relevance items.")
        print(f"Final dataset: {len(output_df)} events saved to {OUT_FILE.name}")
    else:
        print("Failure: No events passed the relevance filter.")

if __name__ == "__main__":
    run_event_study()