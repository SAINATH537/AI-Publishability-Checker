import os
import joblib
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_dir, output_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for all .txt files in the input directory and save them as .pkl files.
    Args:
        input_dir (str): Path to the directory containing cleaned .txt files.
        output_dir (str): Path to save the generated embeddings.
        model_name (str): Name of the SentenceTransformer model to use.
    """
    # Load the SentenceTransformer model
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)
                with open(input_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Generate embedding
                embedding = model.encode(text, convert_to_numpy=True)

                # Save embedding as .pkl
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".pkl")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    joblib.dump(embedding, f)
                print(f"Generated embedding for {input_path} -> {output_path}")

if __name__ == "__main__":
    # Define directories
    cleaned_papers_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\cleaned\papers"
    cleaned_reference_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\cleaned\refrence"
    embeddings_papers_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\embeddings\papers"
    embeddings_reference_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\embeddings\refrence"

    # Generate embeddings for papers
    print("Generating embeddings for papers...")
    generate_embeddings(cleaned_papers_dir, embeddings_papers_dir)

    # Generate embeddings for reference
    print("Generating embeddings for reference...")
    generate_embeddings(cleaned_reference_dir, embeddings_reference_dir)
