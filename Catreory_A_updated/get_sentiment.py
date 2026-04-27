import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER is downloaded
nltk.download('vader_lexicon', quiet=True)

def analyze_ai_trends(raw_input_csv, final_output_csv):
    # 1. Load the original raw data
    df = pd.read_csv(raw_input_csv)
    sia = SentimentIntensityAnalyzer()
    
    # 2. Define AI Keywords (Expanded for better accuracy)
    ai_keywords = {
    # --- General AI Categories ---
    'ai', 'genai', 'generative', 'artificial', 'intelligence', 'llm', 'nlp', 
    'ml', 'machinelearning', 'deeplearning', 'neural', 'agi', 'multimodal', 
    'chatbot', 'transformer', 'algorithm',

    # --- Major Models & Products ---
    'gpt', 'gpt3', 'gpt4', 'gpt4o', 'o1', 'chatgpt', 'claude', 'gemini', 'llama', 
    'mistral', 'sora', 'dalle', 'midjourney', 'stablediffusion', 'grok', 'falcon', 
    'palm', 'whisper', 'bert', 'ernie', 'qwen', 'deepseek', 'r1', 'v3', 'copilot',

    # --- Companies & Organizations ---
    'openai', 'anthropic', 'deepmind', 'google', 'microsoft', 'meta', 'nvidia', 
    'mistralai', 'cohere', 'perplexity', 'xai', 'baidu', 'alibaba', 'tencent', 
    'huggingface', 'databricks', 'stabilityai', 'intel', 'amd', 'apple',

    # --- Technical & Training Terms ---
    'finetuning', 'rag', 'rlhf', 'inference', 'tokens', 'parameters', 
    'weights', 'pretraining', 'vector', 'embedding', 'prompt', 'context',

    # --- Hardware & Infrastructure ---
    'gpu', 'h100', 'a100', 'b200', 'cuda', 'tpu', 'npu', 'compute'}

    def process_row(row):
        # GET RAW TEXT for Sentiment
        raw_text = str(row['summary'])
        
        # CREATE CLEAN TEXT for Relevance Matching
        # We do this on-the-fly so we don't lose the raw text
        clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', raw_text).lower()
        
        # --- Relevance Logic (Using Clean Text) ---
        words = clean_text.split()
        if not words: return 0.0, 0.0
        matches = sum(1 for word in words if any(k in word for k in ai_keywords))
        relevance = round((matches / len(words)) * 100, 2)

        # --- Sentiment Logic (Using Raw Text) ---
        # VADER performs better with punctuation and case!
        sentiment_score = sia.polarity_scores(raw_text)['compound']
        
        return relevance, sentiment_score

    print("Processing raw text for maximum accuracy...")
    
    # Apply logic
    results = df.apply(process_row, axis=1)
    df['ai_relevance_score'], df['sentiment_score'] = zip(*results)
    
    # Labeling
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )

    # 3. Final Aggregation
    # Since you wanted daily trends, we group here AFTER calculating scores
    final_report = df.groupby('publish_date').agg({
        'title': 'first',
        'ai_relevance_score': 'mean',
        'sentiment_score': 'mean',
        'summary': ' '.join # Keep the raw summaries together for record
    }).reset_index()

    final_report.to_csv(final_output_csv, index=False)
    print(f"Done! Final report generated: {final_output_csv}")

# EXECUTION
# Use the file you exported from .numbers to .csv
analyze_ai_trends('ai_launch_news.csv', 'final_ai_analysis_report.csv')