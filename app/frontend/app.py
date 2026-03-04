import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATA_FILE = "../data/processed_data_final.csv"

st.set_page_config(page_title="Student Sentiment Dashboard", layout="wide")

st.title("Student Sentiment Analysis Dashboard")
st.markdown("Monitoring mental health trends & academic pressure on Social Media (Reddit).")

# --- 1. Load Historical Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['date_readable'] = pd.to_datetime(df['date_readable'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# --- 2. Live Fetch Control ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Historical Trends (Past Year)")
    
with col2:
    st.write("### Live Actions")
    if st.button("🔴 Fetch Latest Data"):
        with st.spinner("Crawling Reddit & Analyzing..."):
            try:
                response = requests.get(f"{BACKEND_URL}/crawl_live")
                if response.status_code == 200:
                    data = response.json()
                    new_items = data.get("data", [])
                    st.success(data.get("message"))
                    
                    if new_items:
                        # Show new items
                        st.write("#### Just In:")
                        new_df = pd.DataFrame(new_items)
                        st.dataframe(new_df[['date', 'sentiment', 'original_text']])
                        
                        # In a real app, we would append to 'df' and saving back to csv
                        # For demo, the graph won't update automatically unless we manipulate session state
                        # but showing the table proves the concept.
                else:
                    st.error("Backend Error")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- 3. Dashboard Visualization ---
if not df.empty:
    # A. Sentiment Distribution
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("#### Sentiment Distribution")
        fig_pie = px.pie(df, names='sentiment', title='Overall Sentiment Split', 
                         color='sentiment',
                         color_discrete_map={
                             'Normal': '#2ecc71',
                             'Depression': '#3498db', 
                             'Suicidal': '#e74c3c'
                         })
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.write("#### Monthly Trend")
        # Ensure date column
        if 'date_readable' in df.columns:
            df['month_year'] = df['date_readable'].dt.to_period('M').astype(str)
            trend_df = df.groupby(['month_year', 'sentiment']).size().reset_index(name='count')
            
            fig_line = px.line(trend_df, x='month_year', y='count', color='sentiment', 
                               title='Sentiment Trends Over Time',
                               markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
            
    # B. Detailed View
    st.markdown("---")
    st.write("#### Raw Data Explorer")

    # Filter by Sentiment
    sentiment_options = ['Normal', 'Depression', 'Suicidal']
    selected_sentiments = st.multiselect("Filter by Sentiment", options=sentiment_options, default=sentiment_options)
    
    filtered_df = df[df['sentiment'].isin(selected_sentiments)]
    
    st.dataframe(filtered_df[['date_readable', 'sentiment', 'full_text']].sort_values('date_readable', ascending=False).head(500))
    
else:
    st.warning("Processed data not found. Please run the setup script.")
