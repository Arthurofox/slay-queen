import os
import joblib
from data_processing import load_data, preprocess_data
from model_training import train_models, evaluate_model, save_model
from recommendation import recommend_products
from sklearn.model_selection import train_test_split


def main():
    # Check if the model already exists in the 'models' directory
    models_dir = 'models'
    model_path = os.path.join(models_dir, 'best_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    model_exists = os.path.exists(model_path) and \
                   os.path.exists(vectorizer_path) and \
                   os.path.exists(scaler_path)

    if model_exists:
        print("Loading the existing trained model and preprocessing objects from 'models/' directory...")
        # Load the model and preprocessing objects
        model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        scaler = joblib.load(scaler_path)

        # Load the dataset for recommendations
        df = load_data('datasets/cosmetics.csv')
    else:
        print("No pre-trained model found. Training model for the first time...")
        # Load and preprocess the data
        df = load_data('datasets/cosmetics.csv')
        X, y, tfidf_vectorizer, scaler = preprocess_data(df)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = train_models(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test, y.columns)

        # Save the model and preprocessing objects
        save_model(model, tfidf_vectorizer, scaler)

    # User interaction for recommendations
    print("\nWelcome to the Skincare Recommendation System!")
    print("Please select your skin type(s) from the following options:")
    print("1. Combination")
    print("2. Dry")
    print("3. Normal")
    print("4. Oily")
    print("5. Sensitive")

    # Get user input
    choices = input("Enter the numbers corresponding to your skin types, separated by commas: ")
    choice_map = {'1': 'Combination', '2': 'Dry', '3': 'Normal', '4': 'Oily', '5': 'Sensitive'}
    selected_types = [choice_map.get(choice.strip()) for choice in choices.split(',') if choice.strip() in choice_map]

    if not selected_types:
        print("Invalid selection. Please try again.")
    else:
        # Get recommendations
        recommendations = recommend_products(df, selected_types, model, tfidf_vectorizer, scaler)
        print("\nRecommended Products:")
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
