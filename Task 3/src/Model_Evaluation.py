import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X, y):
    """Evaluate the trained model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix

if __name__ == "__main__":
    # Example usage
    filepath = '../data/Bank Customer Dataset for Churn prediction.xlsx'
    df = pd.read_excel(filepath)
    from data_preprocessing import preprocess_data
    X, y, _ = preprocess_data(df)
    from model_training import train_models
    model_accuracies = train_models(X, y)
    
    # Assuming Random Forest is selected for evaluation
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    report, matrix = evaluate_model(rf_model, X, y)
    
    print("Model evaluation completed.")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)
