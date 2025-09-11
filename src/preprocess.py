import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def process_pdfs(input_dir, output_dir):
    """
    Process all PDFs in the input directory and save extracted text as .txt files in the output directory.
    Args:
        input_dir (str): Path to the directory containing PDF files.
        output_dir (str): Path to save extracted text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                
                if text.strip():  # Ensure the text is not empty
                    # Preserve folder structure
                    relative_path = os.path.relpath(pdf_path, input_dir)
                    output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".txt")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Processed {pdf_path} -> {output_path}")
                else:
                    print(f"No text extracted from {pdf_path}")

if __name__ == "__main__":
    # Directories for processing
    papers_dir = r"c:\Users\SAINATH\OneDrive\Desktop\IITKHARG\data\raw\KDSH_2025_Dataset\Papers"  # 135 PDFs
    reference_dir = r"C:\Users\SAINATH\OneDrive\Desktop\IITKHARG\data\raw\KDSH_2025_Dataset\Reference"  # Publishable/Non-Publishable PDFs
    processed_papers_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\processed_papers"
    processed_reference_dir = r"c:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\reference"

    # Process papers
    print("Processing papers...")
    process_pdfs(papers_dir, processed_papers_dir)

    # Process reference data
    print("Processing reference data...")
    process_pdfs(reference_dir, processed_reference_dir)
