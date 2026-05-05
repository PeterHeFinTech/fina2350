import pandas as pd
import statsmodels.api as sm
from scipy import stats
from pathlib import Path


BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
CSV_PATH = BASE_DIR / "category_C_car_sentiment.csv"   


def calculate_p_values_all_windows():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        return


    car_windows = ['CAR_minus_1', 'CAR_0', 'CAR_1', 'CAR_2']
    
    print("=" * 70)
    print("CATEGORY C: STATISTICAL SIGNIFICANCE REPORT (All CAR Windows)")
    print("=" * 70)

    for window in car_windows:
        print(f"\n{'='*70}")
        print(f"【{window}】")
        print(f"{'='*70}")

    
        t_stat, p_val_ttest = stats.ttest_1samp(df[window], 0)
        print(f"\n[1] One-sample T-test (H0: Mean {window} = 0)")
        print(f"    Mean {window}: {df[window].mean():.4f}")
        print(f"    T-statistic: {t_stat:.4f}")
        print(f"    P-value: {p_val_ttest:.4f}")
        if p_val_ttest < 0.05:
            print("    → Result: SIGNIFICANT at 5% level")
        else:
            print("    → Result: NOT significant at 5% level")

        
        X = df[['nvda_sentiment', 'overall_sentiment']]
        X = sm.add_constant(X)
        y = df[window]

        model = sm.OLS(y, X).fit()

        print(f"\n[2] Multivariate Regression: {window} ~ NVDA Sentiment + Overall Sentiment")
        print(f"    R-squared: {model.rsquared:.4f}")
        print(f"    Adj. R-squared: {model.rsquared_adj:.4f}")
        print(f"    F-statistic p-value: {model.f_pvalue:.4f}")

        # NVDA-specific sentiment
        p_nvda = model.pvalues['nvda_sentiment']
        coef_nvda = model.params['nvda_sentiment']
        print(f"\n    NVDA Sentiment:")
        print(f"      Coefficient: {coef_nvda:.4f}")
        print(f"      P-value: {p_nvda:.4f}")
        if p_nvda < 0.05:
            print("      → SIGNIFICANT at 5% level")
        else:
            print("      → NOT significant at 5% level")

        # Overall sentiment
        p_overall = model.pvalues['overall_sentiment']
        print(f"\n    Overall Sentiment P-value: {p_overall:.4f}")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    calculate_p_values_all_windows()
