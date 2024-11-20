import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data: feature engineering and combining features.
    """
    # Define the target variables (multi-labels)
    y = df[['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']]

    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

    # Fit and transform the 'Ingredients' column
    X_ingredients = tfidf.fit_transform(df['Ingredients'])

    # Normalize 'Price' and 'Rank' columns using MinMaxScaler
    scaler = MinMaxScaler()
    df[['Price', 'Rank']] = scaler.fit_transform(df[['Price', 'Rank']])

    # Extract the normalized 'Price' and 'Rank' as numpy arrays
    X_price_rank = df[['Price', 'Rank']].values

    # Combine all features into a single feature set
    X = hstack([X_ingredients, X_price_rank])

    return X, y, tfidf, scaler
