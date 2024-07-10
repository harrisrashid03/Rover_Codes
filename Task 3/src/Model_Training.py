import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_models(X, y):
    """Train different machine learning models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    # Gradient Boosting
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)

    return {
        "Logistic Regression": lr_accuracy,
        "Random Forest": rf_accuracy,
        "Gradient Boosting": gb_accuracy
    }

if __name__ == "__main__":
    # Example usage
    filepath = '../data/Bank Customer Dataset for Churn prediction.xlsx'
    df = pd.read_excel(filepath)
    from data_preprocessing import preprocess_data
    X, y, _ = preprocess_data(df)
    accuracies = train_models(X, y)
    print("Model training completed.")
    print(accuracies)
