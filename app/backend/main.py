"""
File: ./app/backend/main.py
Description:
- Main entry point for FastAPI backend
- Loads trained SVC pipeline from models/svc_pipeline.pkl
- Endpoints for health check and inference
"""
from fastapi import FastAPI, HTTPException
from schemas import SentimentRequest, SentimentResponse
import joblib
import os

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Path to the model file
MODEL_FILE = "svc_pipeline.pkl"
MODEL_PATH = f"/models/{MODEL_FILE}"
model = None

@app.on_event("startup")
async def startup_event():
    """
    Load the trained model from disk on startup.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/v1/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment using the loaded SVC pipeline.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        input_data = [request.text]
        prediction = model.predict(input_data)[0]

        return SentimentResponse(
            text=request.text,
            sentiment=str(prediction)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")