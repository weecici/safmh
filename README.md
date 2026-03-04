<div align="center">
  <h1>🧠 Sentiment Analysis for Mental Health (SAFMH)</h1>
  <p><i>A simple ML application to analyze and predict mental health sentiments from text.</i></p>

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/Managed_with-uv-purple.svg)](https://github.com/astral-sh/uv)

</div>

---

## 📖 Overview

**Sentiment Analysis for Mental Health (SAFMH)** is a comprehensive machine learning project and web application. It uses a robust Natural Language Processing (NLP) pipeline to evaluate text and categorize it by mental health sentiment.

We trained three different ML models (Multinomial Naive Bayes, Softmax Regression and Linear SVC) and fine-tuned BERT (bert-base-uncased) on a [Kaggle dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) of social media posts. The dataset consists of over 50,000 labeled samples, which divided into 7 classes. For simplicity, we only use samples from 3 classes: Normal, Depression, and Suicidal, for training (fine-tuning) and evaluation our models.

The best-performing model (BERT), achieved an macro F1 score of **84.14%**. The runner-up model is Linear SVC, which achieved a macro F1 score of **80.44%**. Due to good performance with low computational cost (training Linear SVC take ~10 minutes, while fine-tuning BERT takes ~30 minutes per epoch), **Linear SVC is deployed in a FastAPI backend**, while a Streamlit frontend provides an interactive interface for users to input text and receive sentiment predictions in real time.

The project features:

- **FastAPI Backend**: A high-performance RESTful API serving model inference.
- **Streamlit Frontend**: An interactive, user-friendly web interface for real-time text analysis.
- **Data Engineering Utilities**: Custom scripts for crawling data, translating text, and generating historical datasets.
- **Containerized Architecture**: Easily deployable with Docker Compose.

---

## 👥 Team Members

| Student ID | Student Name    | GitHub Profile                                 |
| ---------- | --------------- | ---------------------------------------------- |
| 23520199   | Cuong Nguyen    | [@weecici](https://github.com/weecici)         |
| 23520623   | Minh-Huy Ngo    | [@MinhHuy1507](https://github.com/MinhHuy1507) |
| 23521734   | Thong-Tue Duong | [@tueduong05](https://github.com/tueduong05)   |

---

## 🏗️ Project Structure

```text
safmh/
├── app/
│   ├── backend/        # FastAPI application (main.py, inference pipe.py)
│   ├── frontend/       # Streamlit user interface (app.py)
│   ├── utils/          # Scripts for crawling, translation, and historical data
│   └── data/           # Mounted volume for frontend/backend data sharing
├── models/             # Directory for model weights (Requires manual download)
├── notebooks/          # Jupyter notebooks for EDA and model training
├── studies/            # Optuna studies save directory
├── pyproject.toml      # Python dependencies managed via uv
└── compose.yml         # Docker Compose configuration
```

---

## 🚀 Getting Started

Follow these steps to run the application locally using Docker.

### 1. Clone the Repository

```bash
git clone https://github.com/weecici/safmh.git
cd safmh
```

### 2. Download Model Weights (Required)

The application requires pre-trained model weights to run predictions.

1. Download the weights from this [Google Drive Link](https://drive.google.com/drive/folders/1fNVNcRGaZT654APwCWLIg0DhEXBdvIg6?usp=sharing).
2. Place the downloaded `.pkl` file (e.g., `svc_pipeline.pkl`) inside the `models/` directory in the project root. Ensure the path matches the one defined in `compose.yml` (`/models/svc_pipeline.pkl`).

### 3. Run with Docker Compose

Ensure you have Docker and Docker Compose installed, then spin up the services:

```bash
docker compose up --build
```

This will start both the backend API and the frontend UI concurrently.

---

## 🖥️ Usage

Once the Docker containers are successfully running, you can access the applications via your browser:

- **Frontend User Interface (Streamlit)**:
  👉 [http://localhost:8501](http://localhost:8501)
- **Backend API Docs (FastAPI Swagger UI)**:
  👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Local Development (Without Docker)

If you prefer to run the application locally for development, we use [`uv`](https://github.com/astral-sh/uv) for fast dependency management. (Nix users can also enter the shell using `nix develop`).

```bash
# Sync dependencies and activate virtual environment
uv sync
source .venv/bin/activate

# Run backend (from app/backend directory)
export MODEL_PATH=../../models/svc_pipeline.pkl
uvicorn main:app --reload --port 8000

# Run frontend (from app/frontend directory in a new terminal)
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

---

## 📜 License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for more details.
