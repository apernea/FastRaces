"""Main Pipeline Script
This script orchestrates the entire pipeline: data fetching, preprocessing, and model training.
It ensures that each step is executed in the correct order and handles any necessary data flow between steps"""

from create_datasets import fetch_all_data
from model.data_preprocessor import preprocess_data
from model.train import train_model

def main():
    print("--- Starting Step 1: Data Fetching ---")
    fetch_all_data()
    print("--- Data Fetching Completed ---")

    print("\n--- Starting Step 2: Data Preprocessing ---")
    try:
        X_train, X_test, y_train, y_test = preprocess_data()
    except FileNotFoundError:
        print("Raw data file not found. Please run create_datasets.py first.")
        return

    print("\n--- Starting Step 3: Model Training ---")
    train_model(X_train, X_test, y_train, y_test)
    
    print("\n--- Pipeline Finished Successfully! ---")
    print("Check the 'models' directory for the saved model and the project root for the feature importance plot.")

if __name__ == '__main__':
    main()