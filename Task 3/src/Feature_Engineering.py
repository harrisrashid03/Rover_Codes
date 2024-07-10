import pandas as pd

def create_features(df):
    """Create new features from existing data."""
    # Example: Create a new feature combining existing features
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    return df

if __name__ == "__main__":
    # Example usage
    filepath = '../data/Bank Customer Dataset for Churn prediction.xlsx'
    df = pd.read_excel(filepath)
    df = create_features(df)
    print("Feature engineering completed.")
