"""
File: ./app/backend/schemas.py
Description:
- Pydantic models for request and response validation
- Defines structure for Sentiment input and Prediction output
"""
from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
