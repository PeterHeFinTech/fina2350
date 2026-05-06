import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- 1. DATA LOADING & ROBUST CLEANING ---
def load_and_clean(file_path, cat_id):
    """Loads and cleans dates, handling mixed formats and corrupted 'D1' strings[cite: 2]."""
    df = pd.read_csv(file_path)
    # Fix the "D1" and mixed date format issues found in CSVs[cite: 2]
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    df = df.dropna(subset=['date'])
    
    # Select sentiment and all 3 CAR targets[cite: 2]
    cols = ['date', 'sentiment', 'CAR_0', 'CAR_1', 'CAR_2']
    return df[cols].rename(columns={
        'sentiment': f'sent_{cat_id}',
        'CAR_0': f'car0_{cat_id}',
        'CAR_1': f'car1_{cat_id}',
        'CAR_2': f'car2_{cat_id}'
    })

# Loading the three news category files used in your analysis[cite: 2]
cat1 = load_and_clean("category_A_car_sentiment.csv", "C1")
cat2 = load_and_clean("category_C_car_sentiment.csv", "C2")
cat3 = load_and_clean("category_D_car_sentiment.csv", "C3")

# --- 2. MULTI-DATE MERGE ---
# Merging via outer join to preserve news dates across all categories[cite: 2]
df = pd.merge(cat1, cat2, on='date', how='outer')
df = pd.merge(df, cat3, on='date', how='outer')

# Consolidate the 3 CAR targets by averaging across categories for each date[cite: 2]
for t in [0, 1, 2]:
    df[f'CAR_{t}'] = df[[f'car{t}_C1', f'car{t}_C2', f'car{t}_C3']].mean(axis=1)

# Fill sparse gaps: 0 sentiment for days without category-specific news[cite: 2]
sent_cols = ['sent_C1', 'sent_C2', 'sent_C3']
df[sent_cols] = df[sent_cols].fillna(0)
df = df.dropna(subset=['CAR_0', 'CAR_1', 'CAR_2']).sort_values('date')

# --- 3. TRAINING & VISUALIZATION FOR EACH HORIZON ---
targets = ['CAR_0', 'CAR_1', 'CAR_2']
fig, axes = plt.subplots(len(targets), 2, figsize=(15, 18))

for i, t_col in enumerate(targets):
    # Binary Target: 1 if Positive Abnormal Return, 0 if Negative[cite: 2]
    y = (df[t_col] > 0).astype(int)
    X = df[sent_cols]
    
    # Chronological Split: 25% for testing[cite: 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Random Forest with BALANCED weights to fix Class 0 (Fall) detection issues[cite: 2]
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Plot A: Confusion Matrix for classification validation[cite: 2]
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=axes[i, 0])
    axes[i, 0].set_title(f'Confusion Matrix: {t_col}')
    
    # --- FIXING THE CROSSING LINES ---
    # Create test_df and sort by date strictly to ensure lines move left-to-right
    test_df = df.iloc[X_test.index].copy()
    test_df = test_df.sort_values('date') 
    
    # Re-align predictions with the sorted test_df using index matching
    pred_series = pd.Series(preds, index=X_test.index)
    test_df['aligned_preds'] = pred_series
    
    # Calculate strategy return based on model direction (Long if 1, Short if 0)[cite: 2]
    test_df['strat_ret'] = np.where(test_df['aligned_preds'] == 1, test_df[t_col], -test_df[t_col])
    
    # Plot B: Cumulative Strategy Return versus Market Benchmark[cite: 2]
    axes[i, 1].plot(test_df['date'], (1 + test_df['strat_ret']).cumprod(), label='Model Strategy', color='green', marker='o', markersize=3)
    axes[i, 1].plot(test_df['date'], (1 + test_df[t_col]).cumprod(), label='Actual Market', color='grey', alpha=0.5, linestyle='--')
    axes[i, 1].set_title(f'Financial Performance: {t_col}')
    axes[i, 1].legend()

plt.tight_layout()
plt.show()

print("\nPerformance Report:")
print(classification_report(y_test, preds))
