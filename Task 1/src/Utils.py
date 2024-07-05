import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    df = pd.read_csv(file_path)
    X = df['clean_plot']
    y = df['genre']
    return train_test_split(X, y, test_size=0.2, random_state=42)
