"""Data Preprocessor Module
This module handles data loading, cleaning, feature engineering, and splitting for the XGB model"""

import pandas as pd
import config
import glob

def preprocess_data():
    """
    Loads, cleans, and splits the data for modeling.
    """
    csv_files = glob.glob('datasets/*.csv')  # Adjust path if needed
    df_list = []
    
    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(csv_files)} files with {len(df)} total rows")
    print(f"Seasons in data: {sorted(df['Season'].unique())}")

    # --- Cleaning ---
    df[config.TARGET_VARIABLE] = pd.to_numeric(df[config.TARGET_VARIABLE], errors='coerce')
    df.dropna(subset=[config.TARGET_VARIABLE], inplace=True)
    df[config.TARGET_VARIABLE] = df[config.TARGET_VARIABLE].astype(int)

    # --- Feature Engineering & Selection ---
    df_model = df.drop(columns=['Round', 'Event', 'Driver', 'Status', 'Points'])
    df_model = pd.get_dummies(df_model, columns=['Team'], prefix='Team')
    df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = df_model.drop(config.TARGET_VARIABLE, axis=1)
    y = df_model[config.TARGET_VARIABLE] - 1

    # --- Splitting ---
    test_season = config.TEST_SEASON
    train_indices = df[df['Season'] < test_season].index
    test_indices = df[df['Season'] == test_season].index

    X_train_full, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train_full, y_test = y.loc[train_indices], y.loc[test_indices]

    test_metadata = df.loc[test_indices, ['Season', 'Round', 'Event', 'Driver']].reset_index(drop=True)

    # --- Validation Split (temporal: last 20% of training data) ---
    val_size = int(len(X_train_full) * config.VALIDATION_SPLIT)
    X_train = X_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_train = y_train_full.iloc[:-val_size]
    y_val = y_train_full.iloc[-val_size:]

    # Drop Season column after splitting
    X_train = X_train.drop(columns=['Season'])
    X_val = X_val.drop(columns=['Season'])
    X_test = X_test.drop(columns=['Season'])

    print(f"Data preprocessed. Training shape: {X_train.shape}, Validation shape: {X_val.shape}, Testing shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, test_metadata