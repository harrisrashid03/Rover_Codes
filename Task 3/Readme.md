# Customer Churn Prediction

This project aims to develop a machine learning model to predict customer churn for a subscription-based service or business. The dataset contains historical customer data, including features like usage behavior and customer demographics. The goal is to use algorithms like Logistic Regression, Random Forests, or Gradient Boosting to predict churn.

## Directory and File Descriptions

- **src/**: Contains Python scripts for various tasks.
  - `data_preprocessing.py`: Script to handle data loading and preprocessing.
    - Load the dataset.
    - Handle missing values.
    - Encode categorical variables.
    - Standardize the feature variables.
  - `feature_engineering.py`: Script to perform feature engineering on the dataset.
    - Create new features from existing data if needed.
  - `model_training.py`: Script to train different machine learning models.
    - Train models using Logistic Regression, Random Forest, and Gradient Boosting.
  - `model_evaluation.py`: Script to evaluate the trained models.
    - Evaluate models using metrics like accuracy, precision, recall, and F1 score.
  - `utils.py`: Utility functions for the project.
    - Copy data files.
    - Other helper functions.

- **README.md**: This readme file explaining the project.

## Data Info

The data set was downloaded from Kaggle using the following link: [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).
