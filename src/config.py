"""Config file"""
import os

YEARS_TO_PROCESS = [2023,2024]
DATA_PATH = "datasets/"
DATA_FILE_NAME = "{year}_race_data_w_weather.csv"

TARGET_VARIABLE = "Final_Pos"
MODEL_PATH = 'models/'
PREDICTIONS_PATH = 'results/'
FEATURE_IMPORTANCE_DIR = 'feature_importance'

TEST_SEASON = 2024

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
    'n_jobs': -1,
    'random_state': 42
}
EARLY_STOPPING_ROUNDS = 50