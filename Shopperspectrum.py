import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Load models and data ---
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("rfm_scaler.pkl", "rb"))
similarity_df = pd.read_pickle("product_similarity.pkl")

# --- Page setup ---
st.set_page_config(page_title="Shopper Spectrum App", layout="centered")
st.title("🛍️ Shopper Spectrum: Smart Retail Assistant")

# --- Navigation ---
menu = st.sidebar.selectbox("Choose Feature", ["Product Recommendation", "Customer Segmentation"])

# --- Module 1: Product Recommendation ---
if menu == "Product Recommendation":
    st.subheader("Find Similar Products")
    product_name = st.text_input("Enter a Product Name")

    if st.button("Get Recommendations"):
        if product_name in similarity_df.columns:
            sim_scores = similarity_df[product_name].sort_values(ascending=False)
            top_similar = sim_scores.iloc[1:6]  # skip itself
            st.success("Top 5 Similar Products:")
            for i, (prod, score) in enumerate(top_similar.items(), 1):
                st.write(f"{i}. {prod} (Similarity: {score:.2f})")
        else:
            st.error("Product not found. Please try another.")

# --- Module 2: Customer Segmentation ---
if menu == "Customer Segmentation":
    st.subheader("Predict Customer Segment (RFM Based)")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary Value (total spend)", min_value=1.0, value=100.0)

    if st.button("Predict Customer Segment"):
        rfm_input = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_input)
        cluster = kmeans.predict(rfm_scaled)[0]

        st.write("Predicted Cluster ID:", cluster)  

        # cluster-label mapping 
        cluster_labels = {
             0: 'Occasional',
             1: 'At-Risk',
             2: 'High-Value',
             3: 'Regular'
        }

        segment = cluster_labels.get(cluster, f"Cluster {cluster} (Not Labeled)")
        st.success(f"Predicted Customer Segment: **{segment}**")
