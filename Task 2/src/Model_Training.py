from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    """Train Logistic Regression, Decision Tree, and Random Forest models."""
    models = {}
    
    # Train a Logistic Regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Train a Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    
    # Train a Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    return models

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    filepath = '../data/credit_card_transactions.csv'
    df = load_data(filepath)
    X_scaled, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)
    print("Model training completed.")
