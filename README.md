# KRICT ChemDX Hackathon 2024

## Overview
This repository contains the code and datasets used for the **2024 KRICT ChemDX Hackathon**, held from **November 6 to 8, 2024, at Hanbat National University and KRICT in Daejeon, South Korea**. Participants explored machine learning models for **formation energy prediction** using materials data from **MatDX**, a database within **ChemDX**. Additionally, high-throughput **Raman spectroscopy analysis** was conducted to process and analyze spectral data efficiently. The goal was to leverage machine learning techniques to predict the **formation energy** of inorganic materials and streamline **Raman spectral analysis** through automated workflows.


## Authors
- **Joonyoung F. Joung**  
  - Department of Chemical Engineering, MIT, Cambridge, Massachusetts, USA  
  - Department of Chemistry, Kookmin University, Seoul, Republic of Korea  
- **Kwangsoo Kim**  
  - Hydrogen Research Department, Korea Institute of Energy Research, Daejeon, Republic of Korea  
  - Department of Chemical and Biomolecular Engineering, Yonsei University, Seoul, Republic of Korea  

## Project 1 : Description-formation energy prediction of Inorganic Crystals

### Dataset
The **MatDX** dataset from **ChemDX** was used for training and evaluation. It contains material properties, including formation energy values, extracted from computational chemistry studies. Data preprocessing was performed using **practice.ipynb**, where outliers and missing values were handled, and the dataset was split into training and validation sets.

### Machine Learning Models
Various machine learning models were evaluated for formation energy prediction:

| Model                  | RMSE (eV) | R²     |
|------------------------|----------|--------|
| Gradient boosting     | 1.612    | -8.913 |
| Support vector machine | 0.461    | 0.189  |
| K-nearest neighbor    | 0.434    | 0.280  |
| Decision tree         | 0.414    | 0.345  |
| Random forest        | 0.390    | 0.419  |
| **Neural network**   | **0.326**  | **0.596**  |

#### Training and Evaluation
- **Training:** Implemented in `train.py`, using a **9-fold cross-validation** strategy.
- **Prediction:** Conducted through `model/{each_model}/predictor.py`.
- The **neural network** model outperformed traditional ML models with the lowest **RMSE (0.326 eV)** and the highest **R² (0.596)**.

### Getting Started
#### Installation
To set up the environment, use:
```bash
conda env create -f environment.yml
conda activate hackathon
```

#### Training
To train a model, run
```bash
python train.py
```

#### Making Predictions
To make predictions with a trained model, run:
```bash
python predict.py
```

## Project 2 : Description-High-Throughput Raman Spectroscopy Analysis

The `raman.ipynb` notebook in the `Raman` directory provides a workflow for processing and analyzing Raman spectroscopy data efficiently. The goal is to enable high-throughput Raman spectral analysis by automating key steps such as preprocessing, peak detection, and curve fitting. 

### Key Features:
- **Data Import and Preprocessing:** Loads Raman spectra, applies baseline correction, and normalizes intensity values.
- **Peak Detection and Fitting:** Uses Gaussian or Lorentzian curve fitting for accurate peak characterization.
- **Visualization and Batch Processing:** Generates plots for visual inspection and enables bulk spectral analysis.
- **Exporting Results:** Outputs processed data for further analysis.

This workflow accelerates Raman data analysis, allowing for rapid identification of spectral trends and key features in large datasets. Future improvements could include integrating machine learning for automated peak classification and expanding functionality for other spectroscopic techniques.


## Acknowledgments
This work was supported by Korea Research Institute of Chemical Technology (KRICT) as part of the 2024 KRICT ChemDX Hackathon, which provided a platform for collaborative research and machine learning applications in materials science.

