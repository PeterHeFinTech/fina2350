import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ====================== CONFIGURATION ======================
BASE_DIR = Path("/Users/lin/Desktop/fetch/category C")
CSV_PATH = BASE_DIR / "category_C_car_sentiment.csv"      # ← 已改为最终文件
OUTPUT_IMAGE = BASE_DIR / "nvda_full_timeline_analysis.png"
# ===========================================================

def generate_full_timeline_plot():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        return

    fig, axes = plt.subplots(1, 4, figsize=(26, 7), sharey=True)

    targets = [
        ("CAR_minus_1", "Pre-Event: Information Leakage Check (T-1)"),
        ("CAR_0", "Immediate: Market Shock (Day 0)"),
        ("CAR_1", "Intermediate: Digestion Period (Day 1)"),
        ("CAR_2", "Final: Full Incorporation (Day 2)")
    ]

    plt.style.use('ggplot')

    for ax, (col, title) in zip(axes, targets):
        r_nvda = df['nvda_sentiment'].corr(df[col])
        r_market = df['overall_sentiment'].corr(df[col])

        sns.regplot(data=df, x='nvda_sentiment', y=col, ax=ax,
                    label=f'NVDA Specific (r={r_nvda:.3f})',
                    scatter_kws={'alpha':0.5, 'color':'#d62728', 's':60},
                    line_kws={'color':'#d62728', 'lw':3})

        sns.regplot(data=df, x='overall_sentiment', y=col, ax=ax,
                    label=f'Market General (r={r_market:.3f})',
                    scatter_kws={'alpha':0.2, 'color':'#95a5a6', 's':40},
                    line_kws={'color':'#95a5a6', 'lw':2, 'ls':'--'})

        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Sentiment Score (FinBERT)", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Cumulative Abnormal Return (CAR)", fontsize=13)
        ax.axhline(0, color='black', lw=1, ls='-')
        ax.axvline(0, color='black', lw=1, ls='-')
        ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=9)

    fig.suptitle("The Life Cycle of a Policy Shock: From Information Leakage Test to Final Price Discovery",
                 fontsize=20, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {OUTPUT_IMAGE.name}")


if __name__ == "__main__":
    generate_full_timeline_plot()