# Sentiment Analysis Demo

This app demonstrates a Sentiment Analysis system.
It takes a sentence as input and classifies the sentiment.

## Folder Structure
- `app/`: Main application code
  - `backend/`: FastAPI service
  - `frontend/`: Streamlit interface
  - `docker-compose.yml`: Orchestration file
- `models/`: Folder for model files (.pkl)

## Features
- **Backend**: FastAPI with Pydantic validation.
- **Frontend**: Streamlit.
- **Docker**: Containerized services.

## How to Run
1. Navigate to the `app` directory.
```bash
cd .\app\
```
2. Run `docker-compose up --build`.
```bash
docker-compose up --build -d
```
3. Open Frontend at `http://localhost:8501`.
4. Backend documentation available at `http://localhost:8000/docs`.
