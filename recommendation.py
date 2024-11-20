# recommendation.py

import numpy as np
import pandas as pd
from scipy.sparse import hstack

def recommend_products(df, skin_types, model, vectorizer, scaler, num_recommendations=5):
    """
    Recommend products based on the selected skin types.

    Parameters:
    - df: Original DataFrame containing product data.
    - skin_types: List of skin types selected by the user.
    - model: Trained machine learning model.
    - vectorizer: Trained TF-IDF vectorizer.
    - scaler: Trained scaler for numerical features.
    - num_recommendations: Number of products to recommend.

    Returns:
    - DataFrame containing recommended products.
    """
    # Filter the products suitable for the given skin types
    condition = df[skin_types].any(axis=1)
    suitable_products = df[condition]

    # If no products found, return a message
    if suitable_products.empty:
        return "No suitable products found for the selected skin type(s)."

    # Prepare features for prediction
    X_suitable_ingredients = vectorizer.transform(suitable_products['Ingredients'])
    X_suitable_price_rank = suitable_products[['Price', 'Rank']].values

    # Normalize 'Price' and 'Rank' using the same scaler
    X_suitable_price_rank = pd.DataFrame(X_suitable_price_rank, columns=['Price', 'Rank'])
    X_suitable_price_rank = scaler.transform(X_suitable_price_rank)


    from scipy.sparse import csr_matrix

    # Combine features
    X_suitable = csr_matrix(hstack([X_suitable_ingredients, X_suitable_price_rank]))


    # Predict probabilities
    y_probs = model.predict_proba(X_suitable)

    # Average the probabilities across the selected skin types
    skin_type_indices = [list(df.columns).index(skin) - 6 for skin in skin_types]  # Adjust index for y columns
    avg_probs = np.mean([y_probs[i][:, 1] for i in skin_type_indices], axis=0)

    # Add probabilities to DataFrame
    suitable_products = suitable_products.copy()
    suitable_products['Score'] = avg_probs

    # Recommend top products based on Score
    recommendations = suitable_products.sort_values('Score', ascending=False)
    return recommendations[['Brand', 'Name', 'Price', 'Rank', 'Score']].head(num_recommendations)
