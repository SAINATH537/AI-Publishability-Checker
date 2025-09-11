import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def load_embeddings_and_labels(embeddings_dir):
    """
    Load embeddings and their corresponding labels, handling nested folder structures.
    Args:
        embeddings_dir (str): Path to the directory containing reference embeddings.
    Returns:
        np.array: Embeddings array.
        np.array: Labels array (1 for publishable, 0 for non-publishable).
    """
    embeddings = []
    labels = []

    for root, _, files in os.walk(embeddings_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                embedding = joblib.load(file_path)
                embeddings.append(embedding)

                # Assign label based on folder structure
                if "non_publishable" in root.lower():
                    labels.append(0)
                elif "publishable" in root.lower():
                    labels.append(1)

    print(f"Loaded {len(embeddings)} embeddings with {len(labels)} labels.")
    return np.array(embeddings), np.array(labels)

def train_publishability_model(X, y, output_path):
    """
    Train a Random Forest model with SMOTE to classify publishability.
    Args:
        X (np.array): Feature matrix (embeddings).
        y (np.array): Labels.
        output_path (str): Path to save the trained model.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42, k_neighbors=2)  # Adjusted for small minority class
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled training samples: {len(X_train_resampled)}")

    # Train Random Forest model
    print("Training the Random Forest publishability model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, zero_division=1))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the model
    joblib.dump(model, output_path)
    print(f"Model saved at: {output_path}")

if __name__ == "__main__":
    embeddings_dir = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\data\embeddings\refrence"
    model_output_path = r"C:\Users\SAINATH\OneDrive\Desktop\PAPERSDS\IITKHARG\models\finalmodel2.pkl"

    # Load embeddings and labels
    print("Loading embeddings and labels...")
    X, y = load_embeddings_and_labels(embeddings_dir)

    # Train and save the model
    train_publishability_model(X, y, model_output_path)
