# KRICT_Hackathon
This is project was conducted during the November 2024 Korea Research Institute of Chemical Technology (KRICT) Hackathon.
 ### Authors
  - Joonyoung Francis Joung
  - Kwangsoo Kim

## Project 1 : Description-formation energy prediction of Inorganic Crystals
### Overview
This project aims to predict the formation energy of materials using machine learning models. By leveraging structured datasets and various regression models, we evaluate the feasibility of data-driven approaches in computational materials science.

### Methodology
1. Data Collection and Preprocessing

- Data was obtained via the MatDX API.
- Missing values were removed.
- Standardization was applied.
- Data points with a standard deviation above 1 were considered as outliers.
2. Model Training
- Various machine learning models were trained, including:

  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
  - Decision Tree
  - Gradient Boosting
  - Linear Regression
  - Random Forest
  - Support Vector Machines (SVM)

3. Results Analysis

- 9-fold cross-validation was used for evaluation.
- The best-performing model, MLP, achieved:
  - R² = 0.596
  - RMSE = 0.326




## Project 2 : Description-High-Throughput Raman Spectroscopy Analysis
This project aims to automate Raman spectrum analysis.
