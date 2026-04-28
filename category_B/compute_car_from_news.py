from pathlib import Path
from datetime import time

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PRICE_DIR = BASE_DIR.parent / "nvda_price_data"

NEWS_FILE = BASE_DIR / "category_B.csv"
NVDA_FILE = PRICE_DIR / "NVDA_5Y.csv"
QQQ_FILE = PRICE_DIR / "QQQ_5Y.csv"
OUT_FILE = BASE_DIR / "category_B_car_sentiment.csv"
MARKET_CLOSE = time(16, 0)
CAPM_LOOKBACK = 252
RISK_FREE_RATE = 0.0


def load_price_series(path: Path, close_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df[close_name] = pd.to_numeric(df["Close"], errors="coerce")
    return df[["Date", close_name]].dropna().sort_values("Date")


def build_abnormal_returns() -> pd.DataFrame:
    nvda = load_price_series(NVDA_FILE, "Close_NVDA")
    qqq = load_price_series(QQQ_FILE, "Close_QQQ")

    merged = pd.merge(nvda, qqq, on="Date", how="inner").sort_values("Date")
    merged["r_nvda"] = merged["Close_NVDA"].pct_change()
    merged["r_qqq"] = merged["Close_QQQ"].pct_change()
    merged = merged.dropna(subset=["r_nvda", "r_qqq"]).reset_index(drop=True)
    return merged


def get_event_date(news_dt: pd.Timestamp) -> pd.Timestamp | None:
    if pd.isna(news_dt):
        return None

    event_date = news_dt.normalize()
    if news_dt.time() >= MARKET_CLOSE:
        event_date = event_date + pd.Timedelta(days=1)
    return event_date


def find_event_index(trading_dates: pd.Series, news_date: pd.Timestamp) -> int | None:
    pos = trading_dates.searchsorted(news_date)
    if pos >= len(trading_dates):
        return None
    return int(pos)


def estimate_capm_beta(returns_df: pd.DataFrame, end_idx: int, lookback: int = CAPM_LOOKBACK) -> float | None:
    start_idx = max(0, end_idx - lookback)
    hist = returns_df.loc[start_idx:end_idx - 1, ["r_nvda", "r_qqq"]].dropna()
    if len(hist) < 2:
        return None

    r_stock = hist["r_nvda"] - RISK_FREE_RATE
    r_market = hist["r_qqq"] - RISK_FREE_RATE
    market_var = float(((r_market - r_market.mean()) ** 2).sum())
    if market_var <= 0:
        return None

    cov = float(((r_stock - r_stock.mean()) * (r_market - r_market.mean())).sum())
    return cov / market_var


def capm_abnormal_return(r_nvda: float, r_qqq: float, beta: float) -> float:
    expected = RISK_FREE_RATE + beta * (r_qqq - RISK_FREE_RATE)
    return r_nvda - expected


def compute_car_row(returns_df: pd.DataFrame, t_idx: int) -> tuple[float | None, float | None, float | None, float | None]:
    start = t_idx - 1
    end = t_idx + 2

    if start < 0 or end >= len(returns_df):
        return None, None, None, None

    beta = estimate_capm_beta(returns_df, t_idx)
    if beta is None:
        return None, None, None, None

    window = returns_df.loc[start:end, ["r_nvda", "r_qqq"]].reset_index(drop=True)
    ar_window = window.apply(lambda x: capm_abnormal_return(float(x["r_nvda"]), float(x["r_qqq"]), beta), axis=1)

    car_m1 = float(ar_window.iloc[0])
    car_0 = float(ar_window.iloc[0:2].sum())
    car_1 = float(ar_window.iloc[0:3].sum())
    car_2 = float(ar_window.iloc[0:4].sum())
    return car_m1, car_0, car_1, car_2


def main() -> None:
    news = pd.read_csv(NEWS_FILE)
    news["publish_date"] = pd.to_datetime(news["publish_date"], errors="coerce")
    news["sentiment"] = pd.to_numeric(news["summary_sentiment_score"], errors="coerce")

    returns_df = build_abnormal_returns()
    trading_dates = returns_df["Date"]

    rows = []
    for _, row in news.iterrows():
        if pd.isna(row["publish_date"]):
            continue

        event_date = get_event_date(row["publish_date"])
        if event_date is None:
            continue

        t_idx = find_event_index(trading_dates, event_date)
        if t_idx is None:
            continue

        car_m1, car_0, car_1, car_2 = compute_car_row(returns_df, t_idx)
        if car_m1 is None:
            continue

        rows.append(
            {
                "CAR-1": car_m1,
                "CAR0": car_0,
                "CAR1": car_1,
                "CAR2": car_2,
                "sentiment": row["sentiment"],
            }
        )

    out_df = pd.DataFrame(rows, columns=["CAR-1", "CAR0", "CAR1", "CAR2", "sentiment"])
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
