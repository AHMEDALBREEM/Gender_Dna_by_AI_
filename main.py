import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

# Step 1: Load and preprocess data
def load_data(file_path):
    """
    Load DNA sequences and gender information from a FASTA file.

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        sequences (list): List of DNA sequences.
        genders (list): List of gender labels (0 for female, 1 for male).
    """
    sequences = []
    genders = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        # Assuming gender is the second field after splitting by "|"
        description_parts = record.description.split("|")
        if len(description_parts) > 1:  # Ensure there are enough parts
            gender = description_parts[1].strip().lower()  # Normalize gender to lowercase
            genders.append(gender)
        else:
            logging.warning(f"Skipping record: {record.id} (missing gender information)")
    gender_map = {"female": 0, "male": 1}
    genders = [gender_map.get(g, -1) for g in genders]  # Use -1 for unknown genders
    return sequences, genders

# Step 2: Feature extraction
def get_kmer_counts(sequence, k=6):
    """
    Count k-mers in a DNA sequence.

    Args:
        sequence (str): DNA sequence.
        k (int): Length of k-mers.

    Returns:
        Counter: Dictionary of k-mer counts.
    """
    return Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def calculate_gc_content(sequence):
    """
    Calculate the GC content of a DNA sequence.

    Args:
        sequence (str): DNA sequence.

    Returns:
        float: GC content as a percentage.
    """
    return (sequence.count("G") + sequence.count("C")) / len(sequence)

def extract_features(sequences):
    """
    Extract features from DNA sequences.

    Args:
        sequences (list): List of DNA sequences.

    Returns:
        pd.DataFrame: DataFrame of features.
    """
    kmer_counts = [get_kmer_counts(seq) for seq in sequences]
    kmers = set().union(*kmer_counts)  # Get all unique k-mers
    features = pd.DataFrame({
        "GC_Content": [calculate_gc_content(seq) for seq in sequences],
        "Sequence_Length": [len(seq) for seq in sequences],
        **{f"kmer_{kmer}": [count.get(kmer, 0) for count in kmer_counts] for kmer in kmers}
    })
    return features

# Step 3: Train and evaluate the model
def train_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model and evaluate its performance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (list): Training labels.
        X_test (pd.DataFrame): Testing features.
        y_test (list): Testing labels.

    Returns:
        model: Trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Step 4: Visualize results
def plot_confusion_matrix(y_test, y_pred):
    """
    Plot a confusion matrix.

    Args:
        y_test (list): True labels.
        y_pred (list): Predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Female", "Male"], yticklabels=["Female", "Male"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def predict_gender(model, dna_sequence, kmers):
    """
    Predict the gender of a given DNA sequence using the trained model.

    Args:
        model: Trained Random Forest model.
        dna_sequence (str): DNA sequence to predict.
        kmers (set): Set of k-mers used during training.

    Returns:
        str: Predicted gender ("Female" or "Male").
    """
    # Extract features from the input DNA sequence using the saved k-mers
    features = extract_features([dna_sequence], kmers=kmers)

    # Predict the gender
    prediction = model.predict(features)[0]

    # Map the prediction to the corresponding gender
    gender_map = {0: "Female", 1: "Male"}
    return gender_map.get(prediction, "Unknown")


def extract_features(sequences, kmers=None):
    """
    Extract features from DNA sequences.

    Args:
        sequences (list): List of DNA sequences.
        kmers (set): Set of k-mers to use for feature extraction. If None, all k-mers in the sequences are used.

    Returns:
        pd.DataFrame: DataFrame of features.
    """
    kmer_counts = [get_kmer_counts(seq) for seq in sequences]
    if kmers is None:
        kmers = set().union(*kmer_counts)  # Get all unique k-mers if not provided
    features = pd.DataFrame({
        "GC_Content": [calculate_gc_content(seq) for seq in sequences],
        "Sequence_Length": [len(seq) for seq in sequences],
        **{f"kmer_{kmer}": [count.get(kmer, 0) for count in kmer_counts] for kmer in kmers}
    })
    return features

def main():
    # Step 1: Load data
    sequences, genders = load_data("human_dna.fasta")

    # Filter out sequences with unknown genders
    valid_indices = [i for i, g in enumerate(genders) if g != -1]
    sequences = [sequences[i] for i in valid_indices]
    genders = [genders[i] for i in valid_indices]

    if not sequences:
        logging.error("No valid sequences with gender information found.")
        return

    # Step 2: Extract features
    X = extract_features(sequences)
    y = genders

    # Save the k-mers used during training
    kmers = {col.replace("kmer_", "") for col in X.columns if col.startswith("kmer_")}

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train model
    model = train_model(X_train, y_train, X_test, y_test)

    # Step 5: Visualize results
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    # Step 6: Save model and k-mers
    joblib.dump(model, "gender_predictor.pkl")
    joblib.dump(kmers, "training_kmers.pkl")
    logging.info("Model saved as gender_predictor.pkl")
    logging.info("Training k-mers saved as training_kmers.pkl")

    while True:
    # Step 7: Predict gender for a new DNA sequence
        new_dna_sequence = input("Enter a DNA sequence to predict its gender: ").strip()
        if (new_dna_sequence == "exit"):
            break
        # Load the k-mers used during training
        kmers = joblib.load("training_kmers.pkl")
        # Predict gender using the model and saved k-mers
        predicted_gender = predict_gender(model, new_dna_sequence, kmers)
        print(f"The predicted gender for the given DNA sequence is: {predicted_gender}")


if __name__ == "__main__":
    main()