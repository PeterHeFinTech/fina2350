
import requests
import json
import os
import time
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- CONFIGURATION ---
API_KEY = "PSIZQXUGVLDPAY6J" 
BASE_URL = "https://www.alphavantage.co/query"
OUT_FILE = Path(__file__).resolve().parent / "hardware_arch_shocks_alphavantage.json"
CSV_FILE = Path(__file__).resolve().parent / "nvda_sentiment_news.csv"

# Env vars with safer defaults for debugging
TARGET_TICKER = os.getenv("TARGET_TICKER", "NVDA").upper()
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "20"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_SECONDS", "12"))

# Tickers have been updated to parent/proxy companies for the API call
# (e.g., MSFT for OpenAI, GOOG/GOOGL for Google/Deepmind, blank for purely private like Midjourney)
EVENTS = [
    # --- The 2022-2023 Hype Cycle ---
    {"event_date": "2022-11-30", "event_title": "ChatGPT Launch", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2023-03-14", "event_title": "GPT-4 Release", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2023-05-24", "event_title": "Nvidia Massive Guidance Beat", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2023-12-06", "event_title": "Google Gemini 1.0", "tickers": "GOOG,GOOGL", "topics": "technology"},

    # --- The 2024 Scale-Up ---
    {"event_date": "2024-02-15", "event_title": "OpenAI Sora Preview", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2024-03-18", "event_title": "Nvidia Blackwell Reveal", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2024-04-18", "event_title": "Meta Llama 3 Release", "tickers": "META", "topics": "technology"},
    {"event_date": "2024-12-09", "event_title": "OpenAI Sora Public Release", "tickers": "MSFT", "topics": "technology"},

    # --- The 2025 Efficiency Shock ---
    {"event_date": "2025-01-20", "event_title": "DeepSeek-R1 Shock", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2025-03-24", "event_title": "OpenAI Sora App Shutdown", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2025-09-30", "event_title": "Sora 2 / Social AI launch", "tickers": "MSFT", "topics": "technology"}
]

# --- UTILS ---

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def format_publish_date(value: str | None) -> str | None:
    if not value: return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value

def extract_ticker_sentiment(item: dict, ticker: str) -> tuple[float | None, float | None]:
    for ticker_data in item.get("ticker_sentiment", []):
        if str(ticker_data.get("ticker", "")).upper() == ticker.upper():
            ticker_score = to_float(ticker_data.get("ticker_sentiment_score"))
            relevance = to_float(ticker_data.get("relevance_score"))
            return ticker_score, relevance
    return None, None

def month_window(event_date: str) -> tuple[str, str]:
    if len(event_date) == 7:
        anchor = datetime.strptime(event_date + "-15", "%Y-%m-%d")
    else:
        anchor = datetime.strptime(event_date, "%Y-%m-%d")
    start = anchor - timedelta(days=45)
    end = anchor + timedelta(days=45)
    return start.strftime("%Y%m%dT0000"), end.strftime("%Y%m%dT2359")

# --- CORE LOGIC ---

def fetch_event_news(event: dict) -> dict:
    time_from, time_to = month_window(event["event_date"])
    
    # Base parameters (no tickers by default so it supports empty strings for private orgs)
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": event["topics"],
        "time_from": time_from,
        "time_to": time_to,
        "sort": "RELEVANCE", 
        "limit": "50",
        "apikey": API_KEY,
    }
    
    # Add tickers dynamically if the event specifies them
    if event.get("tickers"):
        params["tickers"] = event["tickers"]

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        payload = response.json()
    except Exception as e:
        print(f"  [ERROR] Network/JSON issue: {e}")
        return {"articles": []}

    # API Rate Limit or Error Check
    if "Note" in payload:
        print(f"  [LIMIT] Alpha Vantage says: {payload.get('Note')}")
        return {"event_date": event["event_date"], "error": "Rate limit reached", "articles": []}
    
    if "ErrorMessage" in payload:
        print(f"  [ERROR] API Error: {payload.get('ErrorMessage')}")
        return {"articles": []}

    feed = payload.get("feed", [])
    if not feed:
        print(f"  [INFO] No articles found for this time window ({time_from} to {time_to})")

    articles = []
    for item in feed:
        # We always check for NVDA sentiment, even if we searched for MSFT or GOOG
        score, relevance = extract_ticker_sentiment(item, TARGET_TICKER)
        
        # We NO LONGER filter out articles missing NVDA. 
        # If NVDA isn't mentioned, we record 0.0 for relevance and sentiment.
        articles.append({
            "publish_date": format_publish_date(item.get("time_published")),
            "title": item.get("title"),
            "source": item.get("source"),
            "nvda_relevance": relevance if relevance is not None else 0.0,
            "nvda_sentiment_score": score if score is not None else 0.0,
            "overall_sentiment_score": to_float(item.get("overall_sentiment_score")),
            "url": item.get("url"),
            "summary": item.get("summary")
        })

    return {
        "event_date": event["event_date"],
        "event_title": event["event_title"],
        "article_count": len(articles),
        "articles": articles,
    }

def main() -> None:
    selected_events = EVENTS[:MAX_EVENTS]
    all_results = {"generated_at": datetime.now(timezone.utc).isoformat(), "events": []}

    print(f"Starting generic event fetch for AI milestones (Targeting {TARGET_TICKER} indirect sentiment)...")

    for idx, event in enumerate(selected_events, start=1):
        print(f"[{idx}/{len(selected_events)}] Processing: {event['event_title']} ({event['event_date']})")
        event_data = fetch_event_news(event)
        all_results["events"].append(event_data)
        
        if event_data.get("article_count", 0) > 0:
            print(f"  [SUCCESS] Found {event_data['article_count']} relevant articles.")
        
        # Prevent rate limiting
        if idx < len(selected_events):
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Save JSON
    OUT_FILE.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Save CSV
    csv_fields = [
        "publish_date", "title", "source", "nvda_relevance", 
        "nvda_sentiment_score", "overall_sentiment_score", "url", "summary"
    ]

    csv_rows = []
    for event in all_results["events"]:
        for article in event.get("articles", []):
            csv_rows.append({field: article.get(field) for field in csv_fields})

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("-" * 30)
    print(f"Done! Created {CSV_FILE} with {len(csv_rows)} rows.")

if __name__ == "__main__":
    main()
