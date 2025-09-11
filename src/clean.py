import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Initialize stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Clean the input text by removing noise, normalizing, and tokenizing.
    Args:
        text (str): The raw text to clean.
    Returns:
        str: The cleaned text.
    """
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Lowercase the text
    text = text.lower()

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(cleaned_tokens)

def process_cleaning(input_dir, output_dir):
    """
    Process all .txt files in the input directory, clean the text, and save in the output directory.
    Args:
        input_dir (str): Path to the directory containing raw text files.
        output_dir (str): Path to the directory to save cleaned text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)
                with open(input_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                cleaned_text = clean_text(raw_text)

                # Save the cleaned text
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                print(f"Cleaned {input_path} -> {output_path}")

if __name__ == "__main__":
    # Define directories for input and output
    processed_papers_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\processed_papers"
    processed_reference_dir =r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\reference"
    cleaned_papers_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\cleaned\papers"
    cleaned_reference_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\cleaned\refrence"

    # Clean raw papers
    print("Cleaning papers...")
    process_cleaning(processed_papers_dir, cleaned_papers_dir)

    # Clean reference papers
    print("Cleaning reference papers...")
    process_cleaning(processed_reference_dir, cleaned_reference_dir)
