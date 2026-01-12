"""
File: ./app/frontend/app.py
Description:
- Streamlit frontend application
- User interface for text input and result display (Updated for Mental Health Sentiment)
"""
import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://backend:8000/api/v1"

st.set_page_config(
    page_title="Mental Health Sentiment Analysis",
    layout="centered"
)

st.title("Mental Health Sentiment Analysis")
st.markdown("Analyze text for mental health related sentiment (**Normal**, **Depression**, **Suicidal**).")

# Input area
user_input = st.text_area("Input Text", height=150, placeholder="Example: I feel really down and hopeless lately...")

if st.button("Analyze", type="primary"):
    if user_input.strip():
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"text": user_input}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result.get("sentiment")
                    
                    color_map = {
                        "Normal": "green",
                        "Depression": "orange", 
                        "Suicidal": "red"
                    }
                    lbl_color = color_map.get(sentiment, "blue")

                    st.success("Analysis Complete!")
                    
                    st.markdown(f"### Prediction: :{lbl_color}[{sentiment}]")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend service. Ensure Docker is running.")
    else:
        st.warning("Please enter some text to analyze.")