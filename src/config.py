"""Config file"""

YEARS_TO_PROCESS = [2025]
RAW_DATA_PATH = "datasets/2025_race_data_w_weather.csv"

TARGET_VARIABLE = "Final_Pos"
TEST_SEASON = 2025
MODEL_PATH = 'model/xgb_f1_model.json'
FEATURE_IMPORTANCE_PLOT_PATH = 'feature_importance.png'


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