import pandas as pd
import re

def clean_text(text):
    """
    Improved cleaning: 
    1. Keeps letters and numbers (essential for AI versions like GPT-4).
    2. Removes special symbols and non-ASCII noise.
    3. Standardizes spacing and case.
    """
    if not isinstance(text, str):
        return ""
    
    # Updated regex: Keeps alphanumeric characters and spaces
    # This prevents 'GPT-4' from becoming 'GPT'
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    
    # Convert to lowercase for uniform analysis
    return text.lower().strip()

def prepare_data(input_filename, output_filename):
    try:
        # Load the data (Assumes file was exported from .numbers to .csv)
        df = pd.read_csv(input_filename)
        
        # Verify required columns exist
        required_cols = ['publish_date', 'title', 'summary']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Missing one of the required columns: {required_cols}")
            return

        # 1. Cleaning
        # Fill empty summaries to avoid errors and apply the new cleaning rule
        df['summary'] = df['summary'].fillna('')
        df['cleaned_summary'] = df['summary'].apply(clean_text)
        
        # 2. Aggregation
        # Group by date and combine all news summaries from that day
        print(f"Aggregating {len(df)} news entries by date...")
        
        df_agg = df.groupby('publish_date').agg({
            'title': 'first',             # Keep the first headline as a reference
            'cleaned_summary': ' '.join   # Combine all cleaned text for that date
        }).reset_index()
        
        # Save to a new CSV
        df_agg.to_csv(output_filename, index=False)
        print(f"Success! Processed data saved to: {output_filename}")
        print(f"Total unique dates: {len(df_agg)}")

    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found. Please export your .numbers file to .csv.")

# Run the cleaning script
# Ensure your file is named 'ai_launch_news.csv' in the same folder
prepare_data('ai_launch_news.csv', 'ai_news_ready.csv')