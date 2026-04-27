import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from transformers import pipeline

# =================================================================
# 1. SETTINGS & CONFIGURATION
# =================================================================
API_TOKEN = 'aJIeaXHo9cCmu83T92tT0nwgCMm4SLhIQzRTUDN7' 
TARGET_TICKER = "NVDA"

# Target Events (Covering both Restrictions and Subsidies)
EVENTS = [
    {"date": "2022-08-09", "name": "CHIPS Act Signed (Subsidy)"},
    {"date": "2022-08-31", "name": "A100/H100 Export Ban"},
    {"date": "2022-10-07", "name": "Export Control Expansion"},
    {"date": "2023-10-17", "name": "H800/A800 Loophole Closure"},
    {"date": "2024-05-07", "name": "Huawei Export License Revocation"}
]

# AI Classification Settings
AI_LABELS = ["Government Policy and Regulation", "Market News", "Product Review"]
AI_THRESHOLD = 0.30  # Lowered slightly to catch hybrid news (Policy + Earnings)

# Expanded Safety Net to include Subsidy-related terms
HARD_KEYWORDS = [
    "ban", "restrict", "export", "chips act", "control", "order", 
    "subsidy", "grant", "funding", "incentive", "china", "biden"
]

# =================================================================
# 2. PHASE 1: BROAD RAW FETCHING (category_C.csv)
# =================================================================

def fetch_raw_data():
    raw_records = []
    for ev in EVENTS:
        # Broad search to ensure we catch any mention of NVDA on event dates
        url = f"https://api.marketaux.com/v1/news/all?search=nvidia&published_on={ev['date']}&language=en&api_token={API_TOKEN}"
        try:
            r = requests.get(url)
            articles = r.json().get('data', [])
            for art in articles:
                # Extract NVDA-specific sentiment score
                score = 0.0
                if art.get('entities'):
                    for entity in art['entities']:
                        if entity['symbol'] == TARGET_TICKER:
                            score = entity.get('sentiment_score', 0.0)
                            break
                
                raw_records.append({
                    "publish_date": art.get('published_at'),
                    "source": art.get('source'),
                    "url": art.get('url'),
                    "title": art.get('title'),
                    "summary": art.get('description'),
                    "raw_sentiment": score,
                    "event_tag": ev['name']
                })
            time.sleep(0.5) 
        except Exception:
            continue

    df_raw = pd.DataFrame(raw_records)
    df_raw.to_csv("category_C.csv", index=False, encoding='utf-8-sig')
    return df_raw

# =================================================================
# 3. PHASE 2: HYBRID CLEANING & TEMPORAL ALIGNMENT (news_cleaned.csv)
# =================================================================

def clean_and_align_data(df_raw):
    # Lightweight model optimized for Mac Air CPU
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=-1)
    cleaned_list = []

    for _, row in df_raw.iterrows():
        title = str(row['title'])
        summary = str(row['summary'])
        combined_text = (title + " " + summary).lower()
        
        # Check A: Keyword Safety Net (Ensures CHIPS Act/Subsidy are kept)
        keyword_hit = any(kw in combined_text for kw in HARD_KEYWORDS)
        
        # Check B: AI Semantic Classification
        res = classifier(combined_text, AI_LABELS)
        is_policy_ai = (res['labels'][0] == "Government Policy and Regulation" and res['scores'][0] > AI_THRESHOLD)

        if keyword_hit or is_policy_ai:
            # 16:00 ET Cutoff Logic (UTC 20:00 = 16:00 ET)
            dt_utc = datetime.strptime(row['publish_date'][:19], '%Y-%m-%dT%H:%M:%S')
            dt_final = dt_utc + timedelta(days=1) if dt_utc.hour >= 20 else dt_utc
            
            # Match Category B Standard Schema
            cleaned_list.append({
                "publish_date": dt_final.strftime('%Y-%m-%d %H:%M:%S'),
                "source": row['source'],
                "url": row['url'],
                "title": title,
                "summary": summary,
                "summary_sentiment_score": row['raw_sentiment']
            })

    df_cleaned = pd.DataFrame(cleaned_list)
    df_cleaned.to_csv("category_C_news_cleaned.csv", index=False, encoding='utf-8-sig')
    return df_cleaned

# =================================================================
# 4. EXECUTION
# =================================================================

if __name__ == "__main__":
    print("--- Start Data Process ---")
    
    raw_df = fetch_raw_data()
    print(f"Phase 1: {len(raw_df)} raw records saved.")
    
    if not raw_df.empty:
        cleaned_df = clean_and_align_data(raw_df)
        print(f"Phase 2: {len(cleaned_df)} cleaned records aligned.")
    
    print("--- Process Complete ---")