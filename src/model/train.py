"""Model Training Module
This module trains the XGBoost model, evaluates it, and saves artifacts."""

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import config

def train_model(X_train, X_test, y_train, y_test):
    """
    Trains the XGBoost model, evaluates it, and saves artifacts.
    """
    model = xgb.XGBClassifier(**config.XGB_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # --- Evaluation ---
    predictions = np.round(model.predict(X_test))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print("--------------------------------")

    #Feature importance plot
    save_feature_importance(model, X_train.columns)

    # --- Save Model ---
    model.save_model(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

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
    plt.savefig(config.FEATURE_IMPORTANCE_PLOT)
    print(f"Feature importance plot saved to {config.FEATURE_IMPORTANCE_PLOT}")
    plt.close()