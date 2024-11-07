# model/Kneighbors_regressor/predictor.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from model.Kneighbors.Kneighbors_regression import FormulaParser

# Load all models from the directory
def load_models(model_dir):
    model_paths = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname.endswith('.joblib')]
    models = [joblib.load(model_path) for model_path in model_paths]
    return models

# Ensemble Prediction Function
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    ensemble_prediction = np.mean(predictions, axis=0)
    return ensemble_prediction

# Evaluate Predictions
def evaluate_predictions(y_true, y_pred, output_path=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Display RMSE and R^2
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    # Optionally save predictions with true values for comparison
    if output_path:
        output = pd.DataFrame({
            'True Formation Energy': y_true,
            'Predicted Formation Energy': y_pred
        })
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output.to_csv(output_path, index=False)
        print(f"Predictions with true values saved to {output_path}")

    return rmse, r2
