import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# =================================================================
# 1. 配置路徑與參數 (已根據你的路徑設定)
# =================================================================
# 這裡定義了剛才報錯缺失的變量
ESTIMATION_LOOKBACK = 252  # 估計 Alpha/Beta 的回測天數（1年）

BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
PRICE_DIR = Path("/Users/lin/Desktop/fetch/category C/nvda_price_data")

NEWS_FILE = BASE_DIR / "category_C_news_cleaned.csv"
NVDA_FILE = PRICE_DIR / "NVDA_5Y.csv"
QQQ_FILE = PRICE_DIR / "QQQ_5Y.csv"
OUT_FILE = BASE_DIR / "category_C_car_sentiment.csv"

# =================================================================
# 2. 數據加載函數
# =================================================================

def load_data(path, name):
    try:
        df = pd.read_csv(path)
        # 自動識別日期列並統一格式
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df[name] = pd.to_numeric(df["Close"], errors="coerce")
        return df[["Date", name]].dropna()
    except Exception as e:
        print(f"❌ Error loading {path.name}: {e}")
        return pd.DataFrame()

# =================================================================
# 3. 核心計算邏輯
# =================================================================

def run_analysis():
    print("🚀 Starting Category C CAR calculation...")

    # A. 載入股價與指數並計算回報率
    nvda = load_data(NVDA_FILE, "p_nvda")
    qqq = load_data(QQQ_FILE, "p_qqq")
    
    if nvda.empty or qqq.empty:
        print("❌ FAILED: Price CSV files not found or empty. Please check your folder.")
        return

    merged = pd.merge(nvda, qqq, on="Date").sort_values("Date")
    merged["r_nvda"] = merged["p_nvda"].pct_change()
    merged["r_qqq"] = merged["p_qqq"].pct_change()
    merged = merged.dropna().reset_index(drop=True)
    
    # B. 載入已清洗的新聞數據
    try:
        news = pd.read_csv(NEWS_FILE)
        news["publish_date"] = pd.to_datetime(news["publish_date"])
    except Exception as e:
        print(f"❌ FAILED: News CSV not found: {e}")
        return
    
    results = []
    
    # C. 遍歷每個政策事件
    for _, row in news.iterrows():
        # 標準化日期
        event_date = row["publish_date"].normalize()
        
        # 尋找交易日索引
        idx_list = merged.index[merged["Date"] == event_date].tolist()
        if not idx_list: 
            # 如果當天不是交易日，找最近的一個交易日
            pos = merged["Date"].searchsorted(event_date)
            if pos >= len(merged): continue
            t0_idx = pos
        else:
            t0_idx = idx_list[0]
        
        # 估計窗口：利用前 252 天計算 Alpha 和 Beta
        # 公式: R_nvda = Alpha + Beta * R_qqq
        hist = merged.iloc[max(0, t0_idx - ESTIMATION_LOOKBACK):t0_idx]
        if len(hist) < 30: continue
        
        # 計算統計指標 (不含 Risk-free)
        beta = np.cov(hist["r_nvda"], hist["r_qqq"])[0, 1] / np.var(hist["r_qqq"])
        alpha = hist["r_nvda"].mean() - beta * hist["r_qqq"].mean()
        
        # D. 計算事件窗口 [-1, 0, 1, 2] 的 Abnormal Returns (AR)
        # AR = 實際回報 - (Alpha + Beta * 市場回報)
        try:
            window = merged.iloc[t0_idx-1 : t0_idx+3].copy()
            if len(window) < 4: continue
            
            window["AR"] = window["r_nvda"] - (alpha + beta * window["r_qqq"])
            ar = window["AR"].values
            
            results.append({
                "CAR-1": ar[0],                   # T-1
                "CAR0":  ar[0:2].sum(),            # T-1 到 T0
                "CAR1":  ar[0:3].sum(),            # T-1 到 T1
                "CAR2":  ar[0:4].sum(),            # T-1 到 T2
                "sentiment": row["summary_sentiment_score"]
            })
        except:
            continue

    # E. 保存結果
    if results:
        pd.DataFrame(results).to_csv(OUT_FILE, index=False)
        print(f"✅ SUCCESS: Linked {len(results)} samples. Output: {OUT_FILE}")
    else:
        print("❌ FAILED: No matches found between news and trading dates.")

if __name__ == "__main__":
    run_analysis()