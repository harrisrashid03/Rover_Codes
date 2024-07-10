import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Load the dataset from the specified filepath."""
    return pd.read_excel(filepath)

def preprocess_data(df):
    """Clean and preprocess the data."""
    # Drop missing values if any
    df = df.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    # Separate features and target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Standardize the feature variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    # Example usage
    filepath = '../data/Bank Customer Dataset for Churn prediction.xlsx'
    df = load_data(filepath)
    X_scaled, y = preprocess_data(df)
    print("Data preprocessing completed.")
