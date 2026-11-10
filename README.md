# Compressive Strength Prediction of Sustainable Concrete Using Ensemble and Non-ensemble Machine Learning Models

Code and data for **ICACE 2026 Paper 391** (8th International Conference on Advances in Civil Engineering, CUET, Chattogram, Bangladesh).
Authors: Abu Bakkar Rabbi and Tawhidul Islam Bhuiyan, University of Asia Pacific.

## Contents
| File | Reproduces in the paper |
|---|---|
| `data/concrete_data_3-56d.csv` | Filtered dataset: the 838 records of the Yeh (UCI) Concrete Compressive Strength dataset with curing ages of 3–56 days |
| `original_models.py` | Table 2 (LR, RF, GBR, XGBoost), Fig. 4 (measured vs predicted), Fig. 5 (SHAP), Fig. 6 (permutation importance) |
| `supplementary_analysis.py` | Table 2 rows for SVR and Decision Tree; Table 3 (repeated 5-fold CV, row-wise and mixture-grouped, mean ± SD); bootstrap 95% CI of RMSE differences; residual plots for all six models; dataset and partition counts |
| `requirements.txt` | Package versions used |

## How to run
```
pip install -r requirements.txt
python original_models.py data/concrete_data_3-56d.csv          # -> original_results/
python supplementary_analysis.py data/concrete_data_3-56d.csv   # -> supp_results/
```
Both scripts run in a few minutes on a normal laptop.

## Validation design
- **Hold-out (Table 2):** random row-wise 80/20 split with `train_test_split(test_size=0.2, random_state=42)` → 670 training / 168 test records. Because the split is row-wise, 113 of the 168 test records share their mixture composition with a training record.
- **Supplementary CV (Table 3):** 5 folds × 3 repeats (seeds 42, 43, 44). Mixture groups = records with identical values of the seven composition variables, irrespective of curing age (425 groups). Grouped folds keep each mixture within one fold; row-wise folds are shown for comparison.
- No automated hyperparameter search and no nested cross-validation were used; all models use the fixed settings below.

## Hyperparameters (fixed; all others at library defaults)
| Model | Settings |
|---|---|
| Linear Regression | ordinary least squares with intercept |
| Decision Tree | max_depth = 10, random_state = 42 |
| SVR | RBF kernel, C = 100, gamma = "scale", epsilon = 0.5, standardized inputs |
| Random Forest | n_estimators = 300, max_depth = 10, random_state = 42 |
| Gradient Boosting | n_estimators = 300, learning_rate = 0.05, max_depth = 4, random_state = 50 |
| XGBoost | n_estimators = 500, learning_rate = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, random_state = 42 |

## Software
Python 3.13.18, NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.4.1, Matplotlib 3.10.0, SHAP 0.54.1.

## Data source and licence
Data: I.-C. Yeh, *Concrete Compressive Strength* [Dataset], UCI Machine Learning Repository, 1998, doi:10.24432/C5PK67, licensed CC BY 4.0. The file here is a filtered subset (ages 3–56 days) and is redistributed under the same licence.
Code: MIT licence (see `LICENSE`).
