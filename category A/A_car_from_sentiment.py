import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from datetime import time
import statsmodels.api as sm

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
NEWS_FILE = BASE_DIR / "finbert_ai_analysis_report.csv" 
NVDA_FILE = BASE_DIR.parent / "stock_price" / "NVDA_5Y.csv"
QQQ_FILE = BASE_DIR.parent / "stock_price" / "QQQ_5Y.csv"
OUT_CSV = BASE_DIR / "nvda_sentiment_regression_results.csv"

MARKET_CLOSE = time(16, 0)
CAPM_LOOKBACK = 252 

def load_data():
    nvda = pd.read_csv(NVDA_FILE)
    qqq = pd.read_csv(QQQ_FILE)
    for df in [nvda, qqq]:
        df['Date'] = pd.to_datetime(df.iloc[:, 0])
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    merged = pd.merge(nvda[['Date', 'Close']], qqq[['Date', 'Close']], on="Date", suffixes=('_nvda', '_qqq'))
    merged = merged.sort_values("Date").dropna()
    merged['r_nvda'] = merged['Close_nvda'].pct_change()
    merged['r_qqq'] = merged['Close_qqq'].pct_change()
    return merged.dropna().reset_index(drop=True)

def run_regression(df, target_col):
    """Runs OLS: CAR = alpha + beta(Sentiment) + error"""
    Y = df[target_col]
    X = df['sentiment_score']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model

def main():
    returns_df = load_data()
    news = pd.read_csv(NEWS_FILE)
    news['publish_date'] = pd.to_datetime(news['publish_date'])
    
    event_results = []

    for _, row in news.iterrows():
        event_dt = row['publish_date'].normalize()
        if row['publish_date'].time() >= MARKET_CLOSE:
            event_dt += pd.Timedelta(days=1)
        
        idx_match = returns_df.index[returns_df['Date'] >= event_dt].tolist()
        if not idx_match: continue
        t_idx = idx_match[0]
        
        # Market Model Estimation
        start_idx = max(0, t_idx - CAPM_LOOKBACK)
        hist = returns_df.iloc[start_idx : t_idx]
        if len(hist) < 30: continue
        
        beta, alpha = np.polyfit(hist['r_qqq'], hist['r_nvda'], 1)
        
        # Calculate AR for window: index 0=T-1, 1=T, 2=T+1, 3=T+2
        window_indices = range(t_idx - 1, t_idx + 3)
        if window_indices[-1] >= len(returns_df) or window_indices[0] < 0: continue
        
        ar = [returns_df.loc[i, 'r_nvda'] - (alpha + beta * returns_df.loc[i, 'r_qqq']) for i in window_indices]
        
        event_results.append({
            'Event_Date': event_dt.date(),
            'sentiment_score': row.get('sentiment_score', 0),
            'CAR_minus_1': ar[0],               # Abnormal return the day BEFORE (T-1)
            'CAR_0': ar[1],                     # Abnormal return on day OF (T)
            'CAR_1': ar[1] + ar[2],             # Sum of Day 0 and Day +1
            'CAR_2': ar[1] + ar[2] + ar[3]      # Sum of Day 0, +1, and +2
        })

    results_df = pd.DataFrame(event_results)
    
    # --- REGRESSION ANALYSIS ---
    print("\n" + "="*40)
    print("OŠI-STYLE REGRESSION RESULTS (N=17)")
    print("="*40)
    
    # Analyze all windows including CAR_minus_1
    windows = ['CAR_minus_1', 'CAR_0', 'CAR_1', 'CAR_2']
    
    for window in windows:
        model = run_regression(results_df, window)
        print(f"\n>>> ANALYSIS FOR {window} <<<")
        print(f"Alpha (Intercept): {model.params['const']:.4f}")
        print(f"Beta (Sentiment):  {model.params['sentiment_score']:.4f}")
        print(f"P-Value:           {model.pvalues['sentiment_score']:.4f}")
        print(f"R-Squared:         {model.rsquared:.4f}")
        
        # Quick significance check
        if model.pvalues['sentiment_score'] < 0.05:
            print("STATUS: STATISTICALLY SIGNIFICANT (*)")
        else:
            print("STATUS: Not Significant")

    # Save to CSV
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[Success] All windows and regressions exported to {OUT_CSV}")

if __name__ == "__main__":
    main()