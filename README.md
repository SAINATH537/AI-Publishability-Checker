# AI-Publishability-Checker

This project is an AI-powered tool to classify research papers as **Publishable** or **Non-Publishable** and recommend suitable conferences for publishable papers. It combines machine learning, natural language processing, and an intuitive UI for efficient paper evaluation.

---

## **Features**
1. **Publishability Classification**:
   - Classifies research papers into "Publishable" and "Non-Publishable".
   - Trained using labeled reference data with state-of-the-art embeddings.

2. **Conference Recommendation**:
   - For publishable papers, suggests relevant conferences such as **CVPR**, **EMNLP**, **KDD**, **NeurIPS**, and **TMLR**.

3. **Intuitive UI**:
   - Upload research papers via a simple web app built with **Streamlit**.
   - View predictions and download results as a CSV.

4. **Robust Machine Learning**:
   - Uses advanced algorithms (Random Forest with SMOTE) for balanced and accurate predictions.

---

## **Dataset Structure**
The dataset is organized as follows:
- **papers/**: Contains 135 unlabeled research papers to be classified.
- **reference/**: Labeled reference data for training.
  - **publishable/**: 10 papers in 5 subfolders (e.g., `CVPR`, `EMNLP`).
  - **non_publishable/**: 5 directly stored papers.

---

## **Project Structure**
---

## **Setup Instructions**
### **Prerequisites**
- Python 3.8 or higher
- Pip (Python package manager)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-Publishability-Checker.git
   cd AI-Publishability-Checker
#to run the application

streamlit run src/app.py

