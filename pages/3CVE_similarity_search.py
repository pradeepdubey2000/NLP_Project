import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the dataset with CVE details
dataset_path = r'C:\Users\pradeep dubey\Desktop\NLP_Project\Data\merged_cve_data.csv'  # Adjust the path accordingly
cve_data = pd.read_csv(dataset_path)

# Check that the embeddings columns (0-767) exist
embedding_columns = [str(i) for i in range(768)]  # Columns 0 to 767
assert all(col in cve_data.columns for col in embedding_columns), "Embedding columns not found!"

# Load the SentenceTransformer model for encoding input descriptions
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # You can replace with your specific model

# Function to encode the input description into an embedding vector
def encode_input(description):
    # Encode the description using the SentenceTransformer model
    return model.encode(description).reshape(1, -1)  # Return as 2D array for cosine similarity

# Function to find the top 5 similar CVEs based on description input
def find_similar_cves(description, n_similar=5):
    # Encode the input description into an embedding
    input_embedding = encode_input(description)
    
    # Extract embeddings from dataset (columns 0-767)
    dataset_embeddings = cve_data[embedding_columns].values

    # Calculate cosine similarity
    similarities = cosine_similarity(input_embedding, dataset_embeddings).flatten()

    # Get top n_similar results
    top_indices = similarities.argsort()[-n_similar:][::-1]
    similar_cves = cve_data.iloc[top_indices]

    # Return results with CVE_ID at the start, including 'Description' and other requested columns
    output = similar_cves[['CVE_ID', 'Description', 'Impact_Score', 'Base_Score', 
                           'Exploitability_Score', 'Access_Complexity', 'Access_Vector', 
                           'Availability_Impact', 'Confidentiality_Impact', 'Integrity_Impact', 
                           'Published_Date']]
    
    # Convert to JSON
    return output.to_json(orient='records', lines=False)

# Streamlit UI for similarity search
st.set_page_config(page_title="🔍 CVE Similarity Search", page_icon=":guardsman:", layout="wide")

st.title("🔍 CVE Similarity Search")
st.markdown("""
Enter a vulnerability description below, and we will find the most similar CVE records from the database. This tool helps in identifying vulnerabilities based on descriptions.
""")
st.markdown("---")

# Text input for the description with styling
description = st.text_area("Enter vulnerability description:", height=150, max_chars=1000)

# Button to trigger search
if st.button("Find Similar CVEs", key="search_button"):
    if description.strip():
        similar_cves_json = find_similar_cves(description)
        
        # Display results in JSON format
        st.markdown("### Top 5 Similar CVEs (JSON Format)")
        st.json(similar_cves_json)

        st.markdown("---")
        st.markdown("""
        This analysis is based on the CVE descriptions and their associated impact scores, exploitability metrics, and other related factors.
        """)

    else:
        st.warning("Please enter a description to find similar CVEs.")

# Optional: Add a footer or additional UI elements if needed
st.markdown("""
---
Created with ❤️ by Your Name. For more information, visit [Your GitHub](https://github.com/yourprofile).
""")
