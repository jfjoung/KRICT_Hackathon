# predict.py

import pandas as pd
# from model.random_forest.predictor import load_models, ensemble_predict, evaluate_predictions
# from model.random_forest.random_forest import FormulaParser
from sklearn.utils import shuffle

# # Load and shuffle test data
# test_data = pd.read_csv('split/test.csv')
# test_data = shuffle(test_data, random_state=42).reset_index(drop=True)

# # Separate features and target
# X_test = test_data[['formula', 'space_group']]
# y_true = test_data['formation_energy_value']

# # Load models and make predictions
# model_dir = 'model/random_forest/'
# models = load_models(model_dir)
# ensemble_predictions = ensemble_predict(models, X_test)

# # Evaluate predictions
# output_path = 'model/random_forest/results/predictions.csv'
# evaluate_predictions(y_true, ensemble_predictions, output_path=output_path)


from model.gradient_boosting_regressor.predictor import load_models, ensemble_predict, evaluate_predictions
from model.gradient_boosting_regressor.gradient_boosting import FormulaParser

# Load and shuffle test data
test_data = pd.read_csv('split/test.csv')
test_data = shuffle(test_data, random_state=42).reset_index(drop=True)

# Separate features and target
X_test = test_data[['formula', 'space_group']]
y_true = test_data['formation_energy_value_per_atom']

# Load models and make predictions
model_dir = 'model/gradient_boosting_regressor/'  
models = load_models(model_dir)
ensemble_predictions = ensemble_predict(models, X_test)

# Evaluate predictions
output_path = 'model/gradient_boosting_regressor/results/predictions.csv'  
evaluate_predictions(y_true, ensemble_predictions, output_path=output_path)
