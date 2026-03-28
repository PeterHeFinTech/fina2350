import yfinance as yf
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=5*365)
df = yf.download("NVDA", start=start, end=end)
df.to_csv("NVDA_5Y.csv")