import pandas as pd

# ====================== 请根据你的实际路径修改 ======================
csv_path = "/Users/lin/Desktop/fetch/category C/category_C_news_cleaned.csv"
# =================================================================

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"错误：找不到文件 {csv_path}")
    exit()

print("=" * 60)
print("Category C - nvda_relevance 分布检查")
print("=" * 60)

print(f"\n总文章数量: {len(df)} 条")

print("\n【relevance 基础统计】")
print(df['nvda_relevance'].describe().round(4))

print("\n【按区间统计文章数量】")
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels = ['0~0.3 (低)', '0.3~0.5 (中低)', '0.5~0.7 (中)', '0.7~1.0 (高)']
df['relevance_bin'] = pd.cut(df['nvda_relevance'], bins=bins, labels=labels, include_lowest=True)
print(df['relevance_bin'].value_counts().sort_index())

print("\n【高相关文章示例】（relevance > 0.6）")
high_rel = df[df['nvda_relevance'] > 0.6][['title', 'nvda_relevance', 'policy_shock_type']].head(5)
print(high_rel.to_string(index=False))

print("\n" + "=" * 60)