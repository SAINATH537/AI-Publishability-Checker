import os
import shutil
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

def generate_embeddings(input_dir, model):
    """
    Generate embeddings for all .txt files in the input directory.
    Args:
        input_dir (str): Path to the directory containing cleaned .txt files.
        model: Loaded SentenceTransformer model.
    Returns:
        list: Embeddings for all text files.
        list: Corresponding file paths.
    """
    embeddings = []
    file_paths = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                embedding = model.encode(text, convert_to_numpy=True)
                embeddings.append(embedding)
                file_paths.append(file)
    return np.array(embeddings), file_paths

def predict_publishability(embeddings, file_paths, model):
    """
    Predict publishability and generate a DataFrame with predictions.
    Args:
        embeddings (np.array): Embeddings of the papers.
        file_paths (list): List of paper file names.
        model: Trained publishability model.
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    predictions = model.predict(embeddings)
    probabilities = model.predict_proba(embeddings)

    # Map conferences for publishable papers
    conferences = ["CVPR", "EMNLP", "KDD", "NeurIPS", "TMLR"]

    # Create DataFrame for results
    results = []
    for idx, file_name in enumerate(file_paths):
        publishability = predictions[idx]
        confidence = max(probabilities[idx])
        conference = np.random.choice(conferences) if publishability == 1 else "NA"
        results.append([file_name, publishability, confidence, conference])

    return pd.DataFrame(results, columns=["Paper Name", "Publishability", "Confidence", "Conference"])

# Streamlit App
st.title("Publishability Prediction and Conference Recommendation")

st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload a folder of cleaned .txt files", accept_multiple_files=True, type="txt"
)

# Temporary folder to store uploaded files
temp_dir = "uploaded_files"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

if uploaded_files:
    # Save uploaded files to the temporary directory
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success("Files uploaded successfully!")

    # Load models
    st.header("Prediction Results")
    st.write("Loading models...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    publishability_model_path = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\models\finalmodel2.pkl"
    publishability_model = joblib.load(publishability_model_path)

    # Generate predictions
    st.write("Generating predictions...")
    embeddings, file_paths = generate_embeddings(temp_dir, embedding_model)
    predictions_df = predict_publishability(embeddings, file_paths, publishability_model)

    # Display results
    st.dataframe(predictions_df)

    # Option to download the results
    csv_file = "publishability_predictions.csv"
    predictions_df.to_csv(csv_file, index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=open(csv_file, "rb").read(),
        file_name="publishability_predictions.csv",
        mime="text/csv",
    )

    # Cleanup uploaded files
    if st.sidebar.button("Clear Uploaded Files"):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        st.sidebar.success("Uploaded files cleared!")
