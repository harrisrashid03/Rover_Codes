import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from the specified filepath."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Drop missing values if any
    df = df.dropna()
    
    # Separate features and target variable
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardize the feature variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    # Example usage
    filepath = '../data/credit_card_transactions.csv'
    df = load_data(filepath)
    X_scaled, y = preprocess_data(df)
    print("Data preprocessing completed.")
