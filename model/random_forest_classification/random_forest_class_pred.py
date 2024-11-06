import pandas as pd
import numpy as np
import joblib
import os
from random_forest import FormulaParser
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

# Load test data
test_data = pd.read_csv('split/test.csv')

# Shuffle the test data (optional, for randomized order if desired)
test_data = shuffle(test_data, random_state=42).reset_index(drop=True)

# Separate features and binary target (0 for negative, 1 for positive formation energy)
X_test = test_data[['formula', 'space_group']]
y_true = (test_data['formation_energy_value'] > 0).astype(int)  # Convert to binary labels

# Directory where classification models are saved
model_dir = 'model/random_forest_classification/'

# Load all models from the directory
model_paths = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname.endswith('.joblib')]
models = [joblib.load(model_path) for model_path in model_paths]

# Ensemble Prediction Function
def ensemble_predict(models, X):
    predictions = []
    for model in models:
        predictions.append(model.predict(X))
    # Average the predictions and round to get binary classification results
    ensemble_prediction = np.round(np.mean(predictions, axis=0)).astype(int)
    return ensemble_prediction

# Predict using ensemble on raw test data
ensemble_predictions = ensemble_predict(models, X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_true, ensemble_predictions)
precision = precision_score(y_true, ensemble_predictions)
recall = recall_score(y_true, ensemble_predictions)
f1 = f1_score(y_true, ensemble_predictions)
conf_matrix = confusion_matrix(y_true, ensemble_predictions)

# Display metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Save predictions with true values for comparison
output = pd.DataFrame({
    'True Formation Energy Sign': y_true,
    'Predicted Formation Energy Sign': ensemble_predictions
})
output_path = 'model/random_forest_classification/results/classification_predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
output.to_csv(output_path, index=False)

print(f"Predictions with true values saved to {output_path}")
