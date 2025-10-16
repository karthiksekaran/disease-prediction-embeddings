import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Drug-Condition Predictor", page_icon="")
st.title("Drug-Condition Predictor")
st.write("Enter a drug name to see the Top 5 most similar drugs and their conditions.")

@st.cache_resource
def load_model():
    return SentenceTransformer("models/all-MiniLM-L6-v2", device="cpu")

@st.cache_data
def load_precomputed():
    df = pd.read_csv("drugLibTrain_prepared.csv")
    embeddings = torch.load("drug_embeddings.pt", map_location="cpu")
    return df, embeddings

with st.spinner("Loading precomputed data..."):
    model = load_model()
    df, drug_embeddings = load_precomputed()

def predict_top_k(drug_name, top_k=5):
    if not drug_name.strip():
        return []
    
    query_emb = model.encode(drug_name, convert_to_tensor=True, device="cpu")
    cos_scores = util.cos_sim(query_emb, drug_embeddings)[0]  # shape (N,)
    
    topk_scores, topk_idx = torch.topk(cos_scores, k=top_k)
    
    results = []
    for idx, score in zip(topk_idx.tolist(), topk_scores.tolist()):
        results.append({
            "Drug Name": df.iloc[idx]["urlDrugName"],
            "Condition": df.iloc[idx]["condition"],
            "Similarity": round(score, 3)
        })
    return results

drug_input = st.text_input("Enter a drug name:", "")

if st.button("Predict Top 5 Conditions"):
    if drug_input.strip():
        with st.spinner("Predicting..."):
            top_matches = predict_top_k(drug_input, top_k=5)
        
        if top_matches:
            st.success("Top 5 similar drugs and their conditions:")
            st.table(top_matches)
        else:
            st.warning("No prediction available.")
    else:
        st.warning("Please enter a valid drug name.")

with st.expander("Sample of the dataset"):
    st.dataframe(df[['urlDrugName', 'condition']].head(10))