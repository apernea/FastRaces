"""Model Training Module
This module trains the XGBoost model, evaluates it, and saves artifacts."""

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import config

def train_model(X_train, X_val, X_test, y_train, y_val, y_test, test_metadata):
    """
    Trains the XGBoost model, evaluates it, and saves artifacts.
    """

    # Model initialization with quantile regression for confidence intervals
    model = xgb.XGBRegressor(
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        learning_rate=config.XGB_PARAMS['learning_rate'],
        n_estimators=config.XGB_PARAMS['n_estimators'],
        max_depth=config.XGB_PARAMS['max_depth'],
        subsample=config.XGB_PARAMS['subsample'],
        colsample_bytree=config.XGB_PARAMS['colsample_bytree'],
        objective=config.XGB_PARAMS['objective']
    )
    lower_quant = xgb.XGBRegressor(
        learning_rate=config.XGB_PARAMS['learning_rate'],
        n_estimators=config.XGB_PARAMS['n_estimators'],
        max_depth=config.XGB_PARAMS['max_depth'],
        subsample=config.XGB_PARAMS['subsample'],
        colsample_bytree=config.XGB_PARAMS['colsample_bytree'],
        objective='reg:quantileerror',
        quantile_alpha=config.LOWER_QUANTILE)
    upper_quant = xgb.XGBRegressor(
        learning_rate=config.XGB_PARAMS['learning_rate'],
        n_estimators=config.XGB_PARAMS['n_estimators'],
        max_depth=config.XGB_PARAMS['max_depth'],
        subsample=config.XGB_PARAMS['subsample'],
        colsample_bytree=config.XGB_PARAMS['colsample_bytree'],
        objective='reg:quantileerror',
        quantile_alpha=config.UPPER_QUANTILE)

    # Training the models
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    lower_quant.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    upper_quant.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predictions and evaluations
    raw_predictions = model.predict(X_test)

    lower_preds = lower_quant.predict(X_test)
    lower_preds = np.clip(lower_preds, 0, 19)  # Ensure predictions are within valid range
    upper_preds = upper_quant.predict(X_test)
    upper_preds = np.clip(upper_preds, 0, 19)  # Ensure predictions are within valid range
    confidence = ((1 - np.clip(upper_preds - lower_preds, 0, 19) / 19) * 100).round().astype(int)                
    ranking_df = test_metadata[['Round']].copy()
    ranking_df['Raw_Pred'] = raw_predictions
    predictions = (
        ranking_df.groupby('Round')['Raw_Pred']
        .rank(method='first')
        .astype(int)
        .values - 1
    )
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
        'Predicted_Pos': predictions + 1,
        'Actual_Pos': y_test.values + 1,
        'Confidence in Prediction (%)': confidence
    }) 
    results_df = results_df.sort_values(by=['Round', 'Predicted_Pos']).reset_index(drop=True)
    results_df.to_csv(f"{config.PREDICTIONS_PATH}/predictions_{config.TEST_SEASON}.csv", index=False)
    print(f"Evaluation results saved to {config.PREDICTIONS_PATH}")

    #Feature importance plot
    save_feature_importance(model, X_train.columns)

    # --- Save Model ---
    model.save_model(f"{config.MODEL_PATH}/xgb_model.ubj")
    print(f"Model saved to {config.MODEL_PATH}")

    return model, results_df

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
    save_path = f"{config.FEATURE_IMPORTANCE_DIR}/feature_importance.png"
    plt.savefig(save_path)
    print(f"Feature importance plot saved to {save_path}")
    plt.close()