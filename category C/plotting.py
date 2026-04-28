import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_full_timeline_plot(csv_path="category_C_car_sentiment.csv", out_img="nvda_full_timeline_analysis.png"):
    # 1. 載入數據
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 2. 設置繪圖 (1x4 佈局，展示完整時間線)
    fig, axes = plt.subplots(1, 4, figsize=(26, 7), sharey=True)
    
    # 定義四個時間窗口
    targets = [
        ("CAR_minus_1", "Pre-Event: Information Leakage Check (T-1)"),
        ("CAR_0", "Immediate: Market Shock (Day 0)"), 
        ("CAR_1", "Intermediate: Digestion Period (Day 1)"), 
        ("CAR_2", "Final: Full Incorporation (Day 2)")
    ]
    
    plt.style.use('ggplot')
    
    # 3. 循環繪製每個子圖
    for ax, (col, title) in zip(axes, targets):
        # 計算相關係數
        r_nvda = df['nvda_sentiment'].corr(df[col])
        r_market = df['overall_sentiment'].corr(df[col])
        
        # 繪製英偉達特有情緒 (核心信號)
        sns.regplot(data=df, x='nvda_sentiment', y=col, ax=ax, 
                    label=f'NVDA Specific (r={r_nvda:.3f})',
                    scatter_kws={'alpha':0.5, 'color':'#d62728', 's':60},
                    line_kws={'color':'#d62728', 'lw':3})
        
        # 繪製大盤整體情緒 (基準/噪音)
        sns.regplot(data=df, x='overall_sentiment', y=col, ax=ax, 
                    label=f'Market General (r={r_market:.3f})',
                    scatter_kws={'alpha':0.2, 'color':'#95a5a6', 's':40},
                    line_kws={'color':'#95a5a6', 'lw':2, 'ls':'--'})
        
        # 格式化
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
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Success: Full timeline plot saved as {out_img}")

if __name__ == "__main__":
    generate_full_timeline_plot()