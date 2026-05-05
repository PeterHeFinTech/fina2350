import pandas as pd
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

# ====================== CONFIGURATION ======================
BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
CSV_PATH = BASE_DIR / "category_C_car_sentiment.csv"      # ← 已改为最终文件
# ===========================================================

def calculate_p_values():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        return

    print("=" * 60)
    print("CATEGORY C: STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 60)

    # 1. One-sample T-test
    t_stat, p_val_mean = stats.ttest_1samp(df['CAR_2'], 0)
    print(f"\n[1] Mean CAR_2 Significance Test:")
    print(f"Average CAR_2: {df['CAR_2'].mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val_mean:.4f}")
    if p_val_mean < 0.05:
        print("→ Result: SIGNIFICANT at 5% level")
    else:
        print("→ Result: NOT significant at 5% level")

    # 2. Multivariate Regression
    X = df[['nvda_sentiment', 'overall_sentiment']]
    X = sm.add_constant(X)
    y = df['CAR_2']

    model = sm.OLS(y, X).fit()

    print(f"\n[2] Multivariate Regression:")
    print(f"NVDA Sentiment Coefficient: {model.params['nvda_sentiment']:.4f}")
    print(f"NVDA Sentiment P-value: {model.pvalues['nvda_sentiment']:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")

    if model.pvalues['nvda_sentiment'] < 0.05:
        print("→ NVDA-specific sentiment is STATISTICALLY SIGNIFICANT.")
    else:
        print("→ NVDA-specific sentiment is NOT statistically significant.")

    print("\n" + "=" * 60)
    print("FULL REGRESSION SUMMARY")
    print("=" * 60)
    print(model.summary())


if __name__ == "__main__":
    calculate_p_values()