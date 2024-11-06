import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
from data_utils.unique import unique_atom, unique_space_group

# Load and shuffle data
data = pd.read_csv('split/train.csv').sample(frac=1, random_state=42).reset_index(drop=True)

# Create binary target (0 = negative, 1 = positive)
data['formation_energy_sign'] = (data['formation_energy_value'] > 0).astype(int)

# Custom transformer to parse formula and extract atom types and stoichiometries
class FormulaParser(BaseEstimator, TransformerMixin):
    def __init__(self, unique_atoms):
        self.unique_atoms = unique_atoms
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        atom_data = []
        for formula in X['formula']:
            atom_counts = {atom: 0 for atom in self.unique_atoms}
            for element, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
                count = int(count) if count else 1
                if element in atom_counts:
                    atom_counts[element] = count
            atom_data.append(atom_counts)
        
        atom_df = pd.DataFrame(atom_data).fillna(0).astype(int)
        atom_df = atom_df.reindex(columns=self.unique_atoms, fill_value=0)
        
        return atom_df.values

# Prepare transformers for formula and space group, using unique lists for encoding
formula_transformer = FormulaParser(unique_atoms=unique_atom)
space_group_transformer = OneHotEncoder(categories=[unique_space_group], sparse_output=False)

# Combine transformations in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('formula', formula_transformer, ['formula']),
        ('space_group', space_group_transformer, ['space_group'])
    ]
)

# Set up data
X = data[['formula', 'space_group']]
y = data['formation_energy_sign']
cv = KFold(n_splits=9, shuffle=True, random_state=42)

# Directory to save models
model_dir = 'model/random_forest_classification/'
os.makedirs(model_dir, exist_ok=True)

# Track scores and confusion matrices
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_matrices = []

# Train and save models for each fold, and evaluate each on its validation set
model_paths = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Pipeline with preprocessor and random forest classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(model_dir, f'random_forest_fold_{fold}.joblib')
    joblib.dump(pipeline, model_path)
    model_paths.append(model_path)
    
    # Predict on validation set and evaluate
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    confusion_matrices.append(conf_matrix)

# Calculate and print mean and std of scores
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Average confusion matrix
avg_conf_matrix = np.mean(confusion_matrices, axis=0)
print("Average Confusion Matrix:")
print(avg_conf_matrix)
