import pandas as pd
import numpy as np
import joblib
import os
from random_forest import FormulaParser
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

# Load test data
test_data = pd.read_csv('split/test.csv')

# Shuffle the test data (optional, for randomized order if desired)
test_data = shuffle(test_data, random_state=42).reset_index(drop=True)

# Separate features and target
X_test = test_data[['formula', 'space_group']]
y_true = test_data['formation_energy_value']

# Directory where models are saved
model_dir = 'model/random_forest/'

# Load all models from the directory
model_paths = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname.endswith('.joblib')]
models = [joblib.load(model_path) for model_path in model_paths]

# Ensemble Prediction Function
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    ensemble_prediction = np.mean(predictions, axis=0)
    return ensemble_prediction

# Predict using ensemble on raw test data
ensemble_predictions = ensemble_predict(models, X_test)

# Calculate RMSE and R^2 score
rmse = np.sqrt(mean_squared_error(y_true, ensemble_predictions))
r2 = r2_score(y_true, ensemble_predictions)

# Display RMSE and R^2
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# Save predictions with true values for comparison
output = pd.DataFrame({
    'True Formation Energy': y_true,
    'Predicted Formation Energy': ensemble_predictions
})
output_path = 'model/random_forest/results/predictions.csv'
output.to_csv(output_path, index=False)

print(f"Predictions with true values saved to {output_path}")
