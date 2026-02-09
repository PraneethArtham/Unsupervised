import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Mall Customer Clustering", layout="wide")

# Load resources
import os

@st.cache_resource
def load_model():
    return joblib.load("unsupervised/kmeans_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("unsupervised/scaler.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("unsupervised/Mall_customers.csv")

@st.cache_data
def load_clustered():
    return pd.read_csv("unsupervised/clustered_mall_customers.csv")


model = load_model()
scaler = load_scaler()
df = load_data()
clustered_df = load_clustered()

st.title("🛍️ Mall Customer Clustering")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), 30)
    income = st.slider(
        "Annual Income (k$)",
        int(df['Annual Income (k$)'].min()),
        int(df['Annual Income (k$)'].max()),
        50
    )
    spending = st.slider("Spending Score", 1, 100, 50)

with col2:
    st.metric("Customers", len(df))
    st.metric("Avg Age", round(df.Age.mean(), 1))
    st.metric("Avg Income", round(df['Annual Income (k$)'].mean(), 1))

if st.button("Predict Cluster"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    })

    scaled_input = scaler.transform(input_df)
    cluster = int(model.predict(scaled_input)[0])

    st.success(f"Predicted Cluster: {cluster}")

    # Visualization
    fig = px.scatter_3d(
        clustered_df,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color='Cluster',
        title="Customer Clusters"
    )

    fig.add_scatter3d(
        x=[age],
        y=[income],
        z=[spending],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Input'
    )

    st.plotly_chart(fig, use_container_width=True)




