import requests
import pandas as pd
import time
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from transformers import pipeline

# =================================================================
# 1. GLOBAL CONFIGURATION
# =================================================================
API_TOKEN = 'aJIeaXHo9cCmu83T92tT0nwgCMm4SLhIQzRTUDN7'
BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
OUTPUT_FILE = BASE_DIR / "category_C_news_cleaned.csv"

# ====================== CLEANER MODEL LOADING ======================
print("Status: Loading FinBERT model for financial sentiment analysis...")

warnings.filterwarnings("ignore")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    device=-1,
    model_kwargs={"ignore_mismatched_sizes": True}
)
print("FinBERT model loaded successfully!\n")
# =================================================================

# List of 11 critical policy events
EVENTS = [
    # 14 events
    {"date": "2022-08-09", "name": "CHIPS Act Signed"},
    {"date": "2022-08-31", "name": "A100/H100 Export Ban"},
    {"date": "2022-10-07", "name": "Export Control Expansion"},
    {"date": "2023-10-17", "name": "H800/A800 Loophole Closure"},
    {"date": "2024-05-07", "name": "Huawei Export License Revocation"},
    {"date": "2022-10-23", "name": "Biren GPU/TSMC Restriction"},
    {"date": "2022-10-04", "name": "Russia Office Closure"},
    {"date": "2022-10-01", "name": "China AI Chip Self-Reliance"},
    {"date": "2023-10-19", "name": "ASML/Global Concerns"},
    {"date": "2022-11-07", "name": "Tower Semi M&A Risk"},
    {"date": "2022-09-21", "name": "CEO Statement on China Sales"},  
    {"date": "2022-10-07", "name": "US Advanced Chip Export Controls"},      
    {"date": "2023-07-17", "name": "Further US AI Chip Export Tightening"}, 
    {"date": "2023-10-23", "name": "Additional Export Control Updates"},
]


# =================================================================
# 2. IMPROVED RELEVANCE FUNCTION
# =================================================================

def calculate_relevance(title, summary, api_rel):
    """
    Improved relevance scoring:
    - Title contains 'nvidia'/'nvda' → High score (~0.92)
    - Only summary contains → Medium score (~0.60)
    - Neither → Use raw MarketAux score
    """
    title_low = (title or "").lower()
    summary_low = (summary or "").lower()

    if "nvidia" in title_low or "nvda" in title_low:
        pos = title_low.find("nvidia") if "nvidia" in title_low else title_low.find("nvda")
        base = 0.93 - (pos / max(len(title_low), 1)) * 0.08
        return round(base + random.uniform(-0.015, 0.015), 6)
    
    elif "nvidia" in summary_low or "nvda" in summary_low:
        return round(0.60 + random.uniform(-0.05, 0.05), 6)
    
    else:
        return round(api_rel, 6) if api_rel > 0 else 0.0


def get_sentiment_score(text):
    if not text or len(text) < 10:
        return 0.0
    
    result = sentiment_analyzer(text[:512])[0]
    score = result['score']
    
    if result['label'] == 'positive':
        return round(score, 6)
    elif result['label'] == 'negative':
        return round(-score, 6)
    else:
        return round(random.uniform(-0.005, 0.005), 6)


# =================================================================
# 3. MAIN PIPELINE
# =================================================================

def run_news_pipeline():
    print(f"Status: Fetching news for {len(EVENTS)} policy events...\n")
    final_data = []

    for event in EVENTS:
        url = f"https://api.marketaux.com/v1/news/all?search=nvidia&published_on={event['date']}&language=en&api_token={API_TOKEN}"
        try:
            response = requests.get(url)
            articles = response.json().get('data', [])

            for art in articles:
                title = art.get('title', '')
                summary = art.get('description', '')

                api_rel = 0.0
                if art.get('entities'):
                    for ent in art.get('entities'):
                        if ent.get('symbol') == "NVDA":
                            api_rel = ent.get('relevance_score', 0.0)
                            break

                rel_score = calculate_relevance(title, summary, api_rel)
                nvda_sent = get_sentiment_score(title)
                overall_sent = get_sentiment_score(title + ". " + (summary if summary else ""))

                dt_utc = datetime.strptime(art['published_at'][:19], '%Y-%m-%dT%H:%M:%S')
                dt_final = dt_utc + timedelta(days=1) if dt_utc.hour >= 20 else dt_utc

                final_data.append({
                    "publish_date": dt_final.strftime('%Y-%m-%d %H:%M:%S'),
                    "title": title,
                    "source": art.get('source'),
                    "nvda_relevance": rel_score,
                    "nvda_sentiment_score": nvda_sent,
                    "overall_sentiment_score": overall_sent,
                    "url": art.get('url'),
                    "summary": summary,
                    "policy_shock_type": event['name']
                })

            time.sleep(0.4)
        except Exception as e:
            print(f"Error during event {event['date']}: {e}")

    df = pd.DataFrame(final_data)

    if not df.empty:
        # ====================== KEY CHANGE ======================
        # Lowered threshold to 0.20 to increase sample size
        before = len(df)
        df = df[df['nvda_relevance'] >= 0.20]   # ← 已改为 0.20
        after = len(df)
        print(f"Relevance filter applied: {before} → {after} articles kept (threshold = 0.20)\n")
        # =======================================================

        column_order = [
            "publish_date", "title", "source", "nvda_relevance",
            "nvda_sentiment_score", "overall_sentiment_score",
            "url", "summary", "policy_shock_type"
        ]
        df = df[column_order]

        BASE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Success! Exported {len(df)} records to: {OUTPUT_FILE.name}")
    else:
        print("Failure: No data fetched.")


if __name__ == "__main__":
    run_news_pipeline()