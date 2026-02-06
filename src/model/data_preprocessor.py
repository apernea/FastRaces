"""Data Preprocessor Module
This module handles data loading, cleaning, feature engineering, and splitting for the XGB model"""

import pandas as pd
import config

def preprocess_data():
    """
    Loads, cleans, and splits the data for modeling.
    """
    df = pd.read_csv(config.RAW_DATA_PATH)

    # --- Cleaning ---
    df[config.TARGET_VARIABLE] = pd.to_numeric(df[config.TARGET_VARIABLE], errors='coerce')
    df.dropna(subset=[config.TARGET_VARIABLE], inplace=True)
    df[config.TARGET_VARIABLE] = df[config.TARGET_VARIABLE].astype(int)

    # --- Feature Engineering & Selection ---
    df_model = df.drop(columns=['Round', 'Event', 'Driver', 'Status', 'Points'])
    df_model = pd.get_dummies(df_model, columns=['Team'], prefix='Team')
    df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = df_model.drop(config.TARGET_VARIABLE, axis=1)
    y = df_model[config.TARGET_VARIABLE]

    # --- Splitting ---
    train_indices = df[df['Season'] != config.TEST_SEASON].index
    test_indices = df[df['Season'] == config.TEST_SEASON].index

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    
    # Drop Season column after splitting
    X_train = X_train.drop(columns=['Season'])
    X_test = X_test.drop(columns=['Season'])
    
    print(f"Data preprocessed. Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test