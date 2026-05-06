import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# --- CONFIGURATION ---
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Fix 1: Map labels dynamically based on the model's own config
# ProsusAI/finbert usually maps: 0 -> positive, 1 -> negative, 2 -> neutral
labels = model.config.id2label 
POS_IDX = next(k for k, v in labels.items() if v.lower() == 'positive')
NEG_IDX = next(k for k, v in labels.items() if v.lower() == 'negative')

def get_finbert_score(raw_text):
    """
    Calculates a sentiment score using raw text (preserving punctuation).
    """
    if not raw_text or len(str(raw_text).strip()) == 0:
        return 0.0
    
    # Fix 3: Using raw_text (FinBERT handles max 512 tokens)
    inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = softmax(outputs.logits.numpy()[0])
    
    # Calculate score: Positive probability minus Negative probability
    # This gives a range of -1 to +1
    score = probs[POS_IDX] - probs[NEG_IDX]
    return round(float(score), 4)

def analyze_ai_trends(raw_input_csv, final_output_csv):
    df = pd.read_csv(raw_input_csv)
    
    # 1. Correct the date interpretation (Day First for July 11th)
    df['publish_date'] = pd.to_datetime(df['publish_date'], dayfirst=True)
    
    ai_keywords = {
        'ai', 'genai', 'llm', 'nlp', 'ml', 'agi', 'gpt', 'gpt4', 'chatgpt', 
        'claude', 'gemini', 'llama', 'mistral', 'deepseek', 'nvidia', 'gpu', 
        'h100', 'cuda', 'transformer', 'finetuning', 'rag'
    }

    def process_row(row):
        raw_text = str(row['summary'])
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', raw_text).lower()
        words = clean_text.split()
        if not words: return 0.0, 0.0
        matches = sum(1 for word in words if word in ai_keywords)
        relevance = round((matches / len(words)) * 100, 2)
        sentiment_score = get_finbert_score(raw_text)
        return relevance, sentiment_score

    print("Running FinBERT Analysis...")
    results = df.apply(process_row, axis=1)
    df['ai_relevance_score'], df['sentiment_score'] = zip(*results)
    
    # --- AGGREGATION WITH EARLIEST TIME ---

    # Create a temporary column that is just the Date (no time) for grouping
    df['date_only'] = df['publish_date'].dt.date

    final_report = df.groupby('date_only').agg({
        'publish_date': 'min',           # <--- Keep the EARLIEST time of the day
        'ai_relevance_score': 'mean',    # Average the relevance
        'sentiment_score': 'mean',       # Average the sentiment
        'summary': ' '.join              # Combine all summaries
    }).reset_index()

    # Drop the helper column
    final_report = final_report.drop(columns=['date_only'])

    # 2. Format the earliest timestamp to YYYY-MM-DD HH:MM:SS
    final_report['publish_date'] = final_report['publish_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    final_report.to_csv(final_output_csv, index=False)
    print(f"Success! report saved to {final_output_csv}")

analyze_ai_trends('ai_launch_news.csv', 'finbert_ai_analysis_report.csv')