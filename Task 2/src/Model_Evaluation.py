from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return accuracy, precision, recall, and F1 score."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def print_evaluation_metrics(model_name, metrics):
    """Print the evaluation metrics."""
    print(f"{model_name} - Accuracy: {metrics[0]}, Precision: {metrics[1]}, Recall: {metrics[2]}, F1 Score: {metrics[3]}")

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    from model_training import train_models
    from sklearn.model_selection import train_test_split
    
    filepath = '../data/credit_card_transactions.csv'
    df = load_data(filepath)
    X_scaled, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        print_evaluation_metrics(model_name, metrics)
    print("Model evaluation completed.")
