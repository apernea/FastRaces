"""Config file"""
import os

YEARS_TO_PROCESS = [2019,2020,2021,2022,2023,2024]
DATA_PATH = "datasets/"
DATA_FILE_NAME = "{year}_race_data_w_weather.csv"

TARGET_VARIABLE = "Final_Pos"
MODEL_PATH = 'models/'
PREDICTIONS_PATH = 'results/'
FEATURE_IMPORTANCE_DIR = 'feature_importance'

TEST_SEASON = 2025

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PREDICTIONS_PATH, exist_ok=True)
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

# Training Configuration
VALIDATION_SPLIT = 0.2  # 20% of historical data for validation
MIN_RACES_FOR_TRAINING = 5  # Minimum number of races needed before making predictions

XGB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'reg_alpha': 0.1,       # L1 regularization (encourages sparsity in feature weights)
    'reg_lambda': 1.5,      # L2 regularization (penalizes large weights)
    'min_child_weight': 3,  # Minimum sum of instance weight needed in a leaf
}
EARLY_STOPPING_ROUNDS = 50

# QUANTILE PARAMERETERS
LOWER_QUANTILE = 0.1
UPPER_QUANTILE = 0.9