import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import re

# Import unique atoms and space groups
from data_utils.unique import unique_atom, unique_space_group

# Load data
data = pd.read_csv('split/train.csv')

# Custom transformer to parse formula and extract atom types and stoichiometries
class FormulaParser(BaseEstimator, TransformerMixin):
    def __init__(self, unique_atoms):
        self.unique_atoms = unique_atoms  # List of unique atoms
    
    def fit(self, X, y=None):
        # Nothing to fit for atom data extraction
        return self
    
    def transform(self, X, y=None):
        # Display the original formula data
        # print("Original formula data:\n", X['formula'].head())
        
        # Extract atom presence and counts from each formula
        atom_data = []
        for formula in X['formula']:
            atom_counts = {atom: 0 for atom in self.unique_atoms}
            for element, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
                count = int(count) if count else 1
                if element in atom_counts:
                    atom_counts[element] = count
            atom_data.append(atom_counts)
        
        # Create DataFrame with atom counts
        atom_df = pd.DataFrame(atom_data).fillna(0).astype(int)
        atom_df = atom_df.reindex(columns=self.unique_atoms, fill_value=0)  # Ensure correct column order
        # print("Atom DataFrame before encoding:\n", atom_df.head())
        
        # Convert atom count data to a numpy array for model input
        combined_data = atom_df.values
        # print("Transformed matrix for formula:\n", combined_data[:5])  # Display first 5 rows
        
        return combined_data


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

# Pipeline with preprocessor and random forest model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Set up cross-validation
X = data[['formula', 'space_group']]
y = data['formation_energy_value']
cv = KFold(n_splits=9, shuffle=True, random_state=42)


# Perform cross-validation and observe the transformed input
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')

# Convert scores to positive MSE and display average and standard deviation
mse_scores = -scores
print(f"Mean MSE: {np.mean(mse_scores):.4f}, Std MSE: {np.std(mse_scores):.4f}")
