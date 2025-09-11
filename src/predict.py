import os
import csv
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

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

def predict_and_generate_csv(embeddings, file_paths, publishability_model, output_csv):
    """
    Predict publishability and generate a CSV file with predictions.
    Args:
        embeddings (np.array): Embeddings of the papers.
        file_paths (list): List of paper file names.
        publishability_model: Trained publishability model.
        output_csv (str): Path to save the predictions CSV.
    """
    predictions = publishability_model.predict(embeddings)
    probabilities = publishability_model.predict_proba(embeddings)

    # Map conferences for publishable papers
    conferences = ["CVPR", "EMNLP", "KDD", "NeurIPS", "TMLR"]

    # Generate CSV rows
    rows = [["Paper Name", "Publishability", "Conference"]]
    for idx, file_name in enumerate(file_paths):
        publishability = predictions[idx]
        conference = np.random.choice(conferences) if publishability == 1 else "NA"
        rows.append([file_name, publishability, conference])

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    # Define paths
    cleaned_papers_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\cleaned\papers"
    publishability_model_path = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\models\finalmodel2.pkl"
    output_csv_path =r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\output\predictions.csv"

    # Load SentenceTransformer model
    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings for the 135 papers
    print("Generating embeddings for papers...")
    embeddings, file_paths = generate_embeddings(cleaned_papers_dir, embedding_model)

    # Load the trained publishability model
    print(f"Loading publishability model from: {publishability_model_path}")
    publishability_model = joblib.load(publishability_model_path)

    # Predict and generate CSV
    print("Predicting publishability and generating CSV...")
    predict_and_generate_csv(embeddings, file_paths, publishability_model, output_csv_path)
