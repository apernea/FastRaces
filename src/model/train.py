"""Model Training Module
This module trains the XGBoost model, evaluates it, and saves artifacts."""

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import config

def train_model(X_train, X_test, y_train, y_test, test_metadata):
    """
    Trains the XGBoost model, evaluates it, and saves artifacts.
    """
    model = xgb.XGBRegressor(**config.XGB_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        n_estimators=config.XGB_PARAMS['n_estimators'],
        max_depth=config.XGB_PARAMS['max_depth'],
        learning_rate=config.XGB_PARAMS['learning_rate'],
        subsample=config.XGB_PARAMS['subsample'],
        colsample_bytree=config.XGB_PARAMS['colsample_bytree'],
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    # --- Evaluation ---
    raw_predictions = model.predict(X_test)
    predictions = np.clip(raw_predictions, 0, 19).round().astype(int)  # Ensure predictions are between 0 and 19

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print("--------------------------------")

    results_df = pd.DataFrame({
        'Season': config.TEST_SEASON,
        'Round': test_metadata['Round'],
        'Event': test_metadata['Event'],
        'Driver': test_metadata['Driver'],
        'Actual_Pos': y_test.values + 1,
        'Predicted_Pos': predictions + 1,
        # 'Confidence (%)': confidence.astype(int)
    }) 
    results_df.to_csv(f"{config.PREDICTIONS_PATH}/predictions_{config.TEST_SEASON}.csv", index=False)
    print(f"Evaluation results saved to {config.PREDICTIONS_PATH}")

    #Feature importance plot
    save_feature_importance(model, X_train.columns)

    # --- Save Model ---
    model.save_model("models/xgb_model.ubj")
    print(f"Model saved to {config.MODEL_PATH}")

    results_df

    return model

def save_feature_importance(model, feature_names):
    """
    Saves the feature importance plot.
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_DIR)
    print(f"Feature importance plot saved to {config.FEATURE_IMPORTANCE_DIR}")
    plt.close()