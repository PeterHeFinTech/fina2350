import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. DATA LOADING & PATH CONFIGURATION ---
def load_news_cat(folder, filename, cat_id):
    """Loads news data using existing CAR columns from the news_ABCD folder."""
    file_path = os.path.join(folder, filename)
    df = pd.read_csv(file_path)
    
    # Standardize date format and handle errors
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    df = df.dropna(subset=['date'])
    
    # Directly select the pre-existing columns: CAR-1, CAR0, CAR1, CAR2
    cols = ['date', 'sentiment', 'CAR-1', 'CAR0', 'CAR1', 'CAR2']
    
    # Rename to include category ID to prevent collision during merge[cite: 2]
    return df[cols].rename(columns={
        'sentiment': f'sent_{cat_id}',
        'CAR-1': f'car_m1_{cat_id}',
        'CAR0': f'car0_{cat_id}',
        'CAR1': f'car1_{cat_id}',
        'CAR2': f'car2_{cat_id}'
    })

# Define your new folder paths[cite: 3]
news_dir = "news_ABCD"
stock_dir = "stock_price"

# Load all four categories (A, B, C, D)[cite: 3]
catA = load_news_cat(news_dir, "category_A_car_sentiment.csv", "A")
catB = load_news_cat(news_dir, "category_B_car_sentiment.csv", "B")
catC = load_news_cat(news_dir, "category_C_car_sentiment.csv", "C")
catD = load_news_cat(news_dir, "category_D_car_sentiment.csv", "D")

# --- 2. MERGING ---
# Combine all categories into one master dataframe[cite: 2, 3]
df = pd.merge(catA, catB, on='date', how='outer')
df = pd.merge(df, catC, on='date', how='outer')
df = pd.merge(df, catD, on='date', how='outer')

# Consolidate the targets by averaging the CARs provided in each file[cite: 2]
target_cols = {'CAR_-1': 'car_m1', 'CAR_0': 'car0', 'CAR_1': 'car1', 'CAR_2': 'car2'}
for final_name, orig_prefix in target_cols.items():
    cols_to_avg = [f'{orig_prefix}_{c}' for c in ['A', 'B', 'C', 'D']]
    df[final_name] = df[cols_to_avg].mean(axis=1)

# Set up features (Sentiments) and fill missing values for days with no news[cite: 2]
sent_features = ['sent_A', 'sent_B', 'sent_C', 'sent_D']
df[sent_features] = df[sent_features].fillna(0)
df = df.dropna(subset=list(target_cols.keys())).sort_values('date')

# --- 3. EVALUATION & PLOTTING ---
targets = list(target_cols.keys())
fig, axes = plt.subplots(len(targets), 2, figsize=(16, 20))

for i, t_col in enumerate(targets):
    # Binary Target: 1 for Rise, 0 for Fall[cite: 2]
    y = (df[t_col] > 0).astype(int)
    X = df[sent_features]
    
    # Split data (30% test size)[cite: 3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Train Balanced Random Forest[cite: 2]
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Confusion Matrix Plot[cite: 2]
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=axes[i, 0])
    axes[i, 0].set_title(f'Confusion Matrix: {t_col}')
    
    # Correcting the "Looping/Crossing" Lines by sorting test_df chronologically[cite: 2]
    test_df = df.iloc[X_test.index].copy()
    test_df = test_df.sort_values('date')
    
    # Align predictions with sorted dates
    pred_series = pd.Series(preds, index=X_test.index)
    test_df['final_preds'] = pred_series
    
    # Strategy Return: Long on 1, Short on 0[cite: 2]
    test_df['strat_ret'] = np.where(test_df['final_preds'] == 1, test_df[t_col], -test_df[t_col])
    
    # Performance Plot[cite: 2]
    axes[i, 1].plot(test_df['date'], (1 + test_df['strat_ret']).cumprod(), label='AI Strategy', color='green', marker='o', markersize=2)
    axes[i, 1].plot(test_df['date'], (1 + test_df[t_col]).cumprod(), label='Market CAR', color='grey', alpha=0.4)
    axes[i, 1].set_title(f'Strategy Backtest: {t_col}')
    axes[i, 1].legend()

plt.tight_layout()
plt.show()