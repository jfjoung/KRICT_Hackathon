# random_forest.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
import re
from data_utils.unique import unique_atom, unique_space_group

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

def train_random_forest():
    # Load and shuffle data
    data = pd.read_csv('split/train.csv').sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Prepare transformers for formula and space group
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
    y = data['formation_energy_value_per_atom']
    cv = KFold(n_splits=9, shuffle=True, random_state=42)

    # Directory to save models
    model_dir = 'model/random_forest/'
    os.makedirs(model_dir, exist_ok=True)

    # Track scores
    rmse_scores = []
    r2_scores = []

    # Train and save models for each fold
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Pipeline with preprocessor and random forest model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])

        # Train model
        pipeline.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(model_dir, f'random_forest_fold_{fold}.joblib')
        joblib.dump(pipeline, model_path)

        # Predict on validation set and evaluate
        y_pred = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

    # Print average RMSE and R^2 scores
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average R^2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
