import requests
import json
import os
import time
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load API key from file
API_KEY_FILE = Path(__file__).resolve().parent / "api_key.txt"
if API_KEY_FILE.exists():
    API_KEY = API_KEY_FILE.read_text().strip()
else:
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not API_KEY:
        print("ERROR: api_key.txt not found and ALPHAVANTAGE_API_KEY env var not set")
        exit(1)

BASE_URL = "https://www.alphavantage.co/query"

# 👑 更新了输出文件名，专门为 Category C 定制
OUT_FILE = Path(__file__).resolve().parent / "category_c_policy_shocks.json"
CSV_FILE = Path(__file__).resolve().parent / "nvda_sentiment_cat_c.csv"

TARGET_TICKER = os.getenv("TARGET_TICKER", "NVDA").upper()
RELEVANCE_THRESHOLD = float(os.getenv("NVDA_RELEVANCE_THRESHOLD", "0.3"))

MAX_EVENTS = int(os.getenv("MAX_EVENTS", "20"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_SECONDS", "12")) # 保护 API 额度的完美设计

# ==========================================
# 👑 CATEGORY C: POLICY / REGULATORY SHOCKS
# 全部精确到历史 T=0 爆发日，适配 [-1, +2] 窗口
# ==========================================
EVENTS = [
    {"event_date": "2022-08-09", "event_title": "US Passes CHIPS and Science Act", "tickers": "NVDA,INTC,AMD", "topics": "technology,economy_macro"},
    {"event_date": "2022-10-07", "event_title": "US Imposes Export Controls on Advanced AI Chips (A100)", "tickers": "NVDA,AMD", "topics": "technology,economy_macro"},
    {"event_date": "2023-03-22", "event_title": "FLI Open Letter to Pause Giant AI Experiments", "tickers": "NVDA,MSFT,GOOG", "topics": "technology"},
    {"event_date": "2023-07-13", "event_title": "China Gen AI Temporary Regulations", "tickers": "NVDA,BIDU", "topics": "technology,economy_macro"},
    {"event_date": "2023-10-30", "event_title": "Biden Executive Order on Safe AI", "tickers": "NVDA,MSFT,GOOG", "topics": "technology,economy_macro"},
    {"event_date": "2023-10-17", "event_title": "US Tightens Chip Export Controls (H800 Loophole)", "tickers": "NVDA,INTC,AMD", "topics": "technology,economy_macro"},
    {"event_date": "2023-11-01", "event_title": "UK AI Safety Summit & Bletchley Declaration", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2024-02-08", "event_title": "US Dept of Commerce Establishes AISIC", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2024-03-13", "event_title": "European Parliament Adopts EU AI Act", "tickers": "NVDA", "topics": "technology,economy_macro"},
    {"event_date": "2024-05-07", "event_title": "US Revokes Intel/Qualcomm Export Licenses to Huawei", "tickers": "NVDA,INTC,QCOM", "topics": "technology,economy_macro"},
    {"event_date": "2024-08-01", "event_title": "EU AI Act Officially Enters Into Force", "tickers": "NVDA", "topics": "technology,economy_macro"},
    {"event_date": "2024-09-29", "event_title": "California Governor Vetoes SB 1047", "tickers": "NVDA,META", "topics": "technology"},
    {"event_date": "2025-02-02", "event_title": "EU AI Act Prohibitions on Unacceptable Risk Take Effect", "tickers": "NVDA", "topics": "technology,economy_macro"},
]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_publish_date(value: str | None) -> str | None:
    if not value:
        return None
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


# 严格的事件研究窗口：[-1, +2]
def exact_event_window(event_date: str) -> tuple[str, str]:
    anchor = datetime.strptime(event_date, "%Y-%m-%d")
    start = anchor - timedelta(days=1)
    end = anchor + timedelta(days=2)
    return start.strftime("%Y%m%dT0000"), end.strftime("%Y%m%dT2359")


def fetch_event_news(event: dict) -> dict:
    time_from, time_to = exact_event_window(event["event_date"])

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": event["tickers"],
        "topics": event["topics"],
        "time_from": time_from,
        "time_to": time_to,
        "sort": "RELEVANCE",
        "limit": "50",
        "apikey": API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    payload = response.json()

    if "Note" in payload:
        return {
            "event_date": event["event_date"],
            "event_title": event["event_title"],
            "time_from": time_from,
            "time_to": time_to,
            "tickers": event["tickers"],
            "error": payload.get("Note"),
            "articles": [],
        }

    feed = payload.get("feed", []) if isinstance(payload, dict) else []
    articles = []
    for item in feed:
        nvda_sentiment_score, nvda_relevance = extract_ticker_sentiment(item, TARGET_TICKER)
        if nvda_relevance is None or nvda_relevance <= RELEVANCE_THRESHOLD:
            continue

        articles.append(
            {
                "publish_date": format_publish_date(item.get("time_published")),
                "title": item.get("title"),
                "source": item.get("source"),
                "nvda_relevance": nvda_relevance,
                "nvda_sentiment_score": nvda_sentiment_score,
                "overall_sentiment_score": to_float(item.get("overall_sentiment_score")),
                "url": item.get("url"),
                "summary": item.get("summary"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
            }
        )

    return {
        "event_date": event["event_date"],
        "event_title": event["event_title"],
        "time_from": time_from,
        "time_to": time_to,
        "tickers": event["tickers"],
        "article_count": len(articles),
        "articles": articles,
    }


def main() -> None:
    selected_events = EVENTS[:MAX_EVENTS]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Alpha Vantage NEWS_SENTIMENT",
        "events": [],
    }

    for idx, event in enumerate(selected_events, start=1):
        print(f"[{idx}/{len(selected_events)}] {event['event_title']} (T=0: {event['event_date']})")
        event_result = fetch_event_news(event)
        result["events"].append(event_result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON: {OUT_FILE}")

    csv_fields = [
        "event_date",
        "publish_date",
        "title",
        "source",
        "nvda_relevance",
        "nvda_sentiment_score",
        "overall_sentiment_score",
        "url",
        "summary",
    ]

    csv_rows = []
    for event in result["events"]:
        for article in event.get("articles", []):
            row_data = {field: article.get(field) for field in csv_fields if field != "event_date"}
            row_data["event_date"] = event["event_date"]
            csv_rows.append(row_data)

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"Saved CSV: {CSV_FILE} (rows={len(csv_rows)}, ticker={TARGET_TICKER}, relevance>{RELEVANCE_THRESHOLD})"
    )


if __name__ == "__main__":
    main()
