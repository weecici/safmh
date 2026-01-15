import pandas as pd
import joblib
import sys
import os
from ..backend.pipe import full_preprocess

MODEL_PATH = "../../models/svc_pipeline.pkl"
INPUT_FILE = "../data/voz_data_english.csv"
OUTPUT_FILE = "../data/processed_data_final.csv"

def generate():
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    
    print("Loading English data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        # Create a dummy one for demo structure if missing
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Check needed columns
    if 'translated_text' not in df.columns:
        print("Column 'translated_text' missing.")
        return
        
    print("Preprocessing and Predicting...")
    # Apply preprocessing from pipe.py
    # We must treat NaN
    df['translated_text'] = df['translated_text'].fillna("")
    
    # Process
    df['processed_text'] = df['translated_text'].apply(full_preprocess)
    
    # Predict
    # Scikit learn predict takes list or array
    predictions = model.predict(df['processed_text'].tolist())
    
    df['sentiment'] = predictions
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated {OUTPUT_FILE} with {len(df)} rows.")
    print(df[['translated_text', 'sentiment']].head())

if __name__ == "__main__":
    generate()
