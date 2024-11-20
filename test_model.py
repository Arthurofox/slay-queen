import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_processing import load_data, preprocess_data

def load_trained_objects():
    """
    Load the trained model, TF-IDF vectorizer, and scaler from the 'models' directory.
    """
    if not all(os.path.exists(f'models/{obj}') for obj in ['best_model.pkl', 'tfidf_vectorizer.pkl', 'scaler.pkl']):
        raise FileNotFoundError("Trained model or preprocessing objects are missing in 'models/' directory.")
    
    model = joblib.load('models/best_model.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, tfidf_vectorizer, scaler

def evaluate_model_performance(model, X_test, y_test, y_columns):
    """
    Evaluate the model and generate a detailed performance report.
    """
    print("\nEvaluating the model...")
    
    # Predict probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=y_columns)
    print("\nClassification Report:\n", report)
    
    return y_pred, y_pred_proba, accuracy, f1, precision, recall

def plot_roc_curves(y_test, y_pred_proba, y_columns):
    """
    Plot ROC curves for each target variable.
    """
    plt.figure(figsize=(12, 8))
    
    for i, column in enumerate(y_columns):
        fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_proba[i][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{column} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multi-Label Classification")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def main():
    # Load the trained model and preprocessing objects
    print("Loading the trained model and preprocessing objects...")
    model, tfidf_vectorizer, scaler = load_trained_objects()

    # Load and preprocess the dataset
    df = load_data('datasets/cosmetics.csv')
    X, y, _, _ = preprocess_data(df)
    
    # Split the dataset into training and testing sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate the model performance
    y_pred, y_pred_proba, accuracy, f1, precision, recall = evaluate_model_performance(
        model, X_test, y_test, y.columns
    )
    
    # Plot ROC curves
    print("\nGenerating ROC curves...")
    plot_roc_curves(y_test, y_pred_proba, y.columns)
    
    # Save metrics to a file
    with open('model_performance_report.txt', 'w') as file:
        file.write(f"Model Performance Report\n")
        file.write(f"------------------------\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"F1-Score: {f1:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(classification_report(y_test, y_pred, target_names=y.columns))
    print("\nModel performance report saved as 'model_performance_report.txt'.")

if __name__ == "__main__":
    main()
