"""Main Pipeline Script
This script orchestrates the entire pipeline: data fetching, preprocessing, and model training.
It ensures that each step is executed in the correct order and handles any necessary data flow between steps.
This script can be run with the '--fetch' flag to trigger the data fetching process, which should only be done when necessary."""

from create_datasets import fetch_all_data
from model.data_preprocessor import preprocess_data
from model.train import train_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="FastRaces F1 Prediction Pipeline")

    parser.add_argument(
        '--fetch', 
        action='store_true', 
        help="Run the long data fetching process to populate the database. Use this only for the first run or to refresh data."
    )
    
    args = parser.parse_args()

    if args.fetch:
        print("--- Starting Step 1: Data Fetching (as requested) ---")
        fetch_all_data()
        print("--- Data Fetching Finished ---")
    else:
        print("--- Step 1: Data Fetching Skipped ---")
        print("(Use 'python main.py --fetch' to run the data collection process)")


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