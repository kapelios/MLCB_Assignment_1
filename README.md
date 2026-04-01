# MLCB Assignment 1

## Report

[Assignment 1 Report](report/Assignment_1_Report.pdf)

## Project Structure

```
MLCB_Assignment_1/
├── data/
│   ├── development_data.csv
│   └── evaluation_data.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
├── report/
│   └── Assignment_1_Report.pdf
└── src/
    └── functions.py
```

### Data

Derived from **GSE40279** (Hannum et al., 2013) — 1000 CpG features selected by absolute Spearman correlation with age, with a random subset of values set to NaN.

- **`development_data.csv`** — 456 samples used for training and validation (80/20 stratified split).
- **`evaluation_data.csv`** — 100 samples held out entirely for final performance estimation.

### Models

- **`best_model.pkl`** — Serialized best-performing regression model selected after feature selection and hyperparameter tuning.

### Notebooks

- **`data_exploration.ipynb`** — Covers Task 1: data splitting, preprocessing (mean imputation, standard scaling, one-hot encoding), and exploratory analysis of age distribution and feature correlations.
- **`model_analysis.ipynb`** — Covers Tasks 2–4 and Bonus A/B: OLS baseline evaluation, feature selection (stability selection, mRMR), hyperparameter tuning (RandomizedSearchCV and Optuna), final evaluation on the held-out set, and sex classification from CpG methylation.

### Source

- **`src/functions.py`** — Shared utility functions used across both notebooks: data loading, preprocessing pipeline construction, bootstrap evaluation, stability selection, mRMR feature selection, model tuning, and all plotting helpers.