from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from pipe import full_preprocess, crawl_reddit_live, translate_text
import logging

app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# Load Model
MODEL_PATH = "../../models/svc_pipeline.pkl"
try:
    # We load the pipeline. Remember this pipeline has TfidfVectorizer + LinearSVC
    # It expects *preprocessed* text (as string) if trained that way.
    # Based on notebook, X was 'statement_processed'.
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model_pipeline = None

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "ok", "message": "Sentiment Analysis Backend Live"}

@app.post("/predict")
def predict_sentiment(request: PredictionRequest):
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Preprocess
    processed_text = full_preprocess(request.text)
    
    # 2. Predict
    # Scikit-learn pipeline expects iterable
    prediction = model_pipeline.predict([processed_text])[0]
    
    return {"text": request.text, "processed": processed_text, "sentiment": prediction}

@app.get("/crawl_live")
def trigger_live_crawl():
    """
    Crawls ~5-10 posts, translates, and predicts.
    Returns the list of analyzed posts.
    """
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # 1. Crawl
        df_new = crawl_reddit_live(limit=20)
        
        if df_new.empty:
            return {"message": "No new relevant posts found.", "data": []}
        
        results = []
        
        for _, row in df_new.iterrows():
            original_text = row['full_text']
            
            # 2. Translate
            translated_text = translate_text(original_text)
            
            # 3. Preprocess
            processed_text = full_preprocess(translated_text)
            
            # 4. Predict
            sentiment = model_pipeline.predict([processed_text])[0]
            
            results.append({
                "id": row['id'],
                "date": str(row['date_readable']),
                "original_text": original_text[:100] + "...", # Snippet
                "translated_text": translated_text[:100] + "...",
                "sentiment": sentiment
            })
            
        return {"message": f"Successfully analyzed {len(results)} posts.", "data": results}

    except Exception as e:
        logger.error(f"Live process failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
