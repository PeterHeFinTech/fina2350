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
    
    # Use exact word matching list
    ai_keywords = {
        'ai', 'genai', 'llm', 'nlp', 'ml', 'agi', 'gpt', 'gpt4', 'chatgpt', 
        'claude', 'gemini', 'llama', 'mistral', 'deepseek', 'nvidia', 'gpu', 
        'h100', 'cuda', 'transformer', 'finetuning', 'rag'
        # ... add more specific terms here
    }

    def process_row(row):
        raw_text = str(row['summary'])
        
        # --- Fix 2: Better Relevance Logic ---
        # 1. Clean for counting: lower case and remove non-alphanumeric except spaces
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', raw_text).lower()
        words = clean_text.split()
        
        if not words: 
            return 0.0, 0.0

        # Exact word matching to avoid "rain" matching "ai"
        matches = sum(1 for word in words if word in ai_keywords)
        relevance = round((matches / len(words)) * 100, 2)

        # --- Fix 3: Sentiment Logic ---
        # Pass the UNTOUCHED raw_text to FinBERT
        sentiment_score = get_finbert_score(raw_text)
        
        return relevance, sentiment_score

    print("Running FinBERT Analysis with fixed indexing and matching...")
    results = df.apply(process_row, axis=1)
    df['ai_relevance_score'], df['sentiment_score'] = zip(*results)
    
    # Aggregation
    final_report = df.groupby('publish_date').agg({
        'ai_relevance_score': 'mean',
        'sentiment_score': 'mean',
        'summary': ' '.join 
    }).reset_index()

    final_report.to_csv(final_output_csv, index=False)
    print(f"Success! Report saved to {final_output_csv}")

analyze_ai_trends('ai_launch_news.csv', 'finbert_ai_analysis_report.csv')