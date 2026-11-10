#!/usr/bin/env python3
"""
Reproduces Table 2 of ICACE 2026 Paper 391 exactly as originally executed:
random row-wise 80/20 split (random_state=42) and FIXED hyperparameters (no tuning).
Also produces Fig. 4 (measured vs predicted), Fig. 5 (SHAP summary) and Fig. 6 (permutation importance).

Usage:  python original_models.py data/concrete_data_3-56d.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

OUT = Path("original_results"); OUT.mkdir(exist_ok=True)
df = pd.read_csv(sys.argv[1])
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                                   random_state=50),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, random_state=42),
}
rows, preds = [], {}
for name, m in models.items():
    p = m.fit(X_train, y_train).predict(X_test)
    preds[name] = p
    mse = mean_squared_error(y_test, p)
    rows.append({"Model": name, "R2": r2_score(y_test, p), "MAE": mean_absolute_error(y_test, p),
                 "MSE": mse, "RMSE": np.sqrt(mse)})
t2 = pd.DataFrame(rows).set_index("Model")
t2.round(4).to_csv(OUT / "table2_holdout_metrics.csv")
print(f"Train = {len(X_train)}, Test = {len(X_test)}\n", t2.round(3))

# Fig. 4: measured vs predicted, common test set
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2), dpi=300)
lim = [0, max(y_test.max(), max(p.max() for p in preds.values())) * 1.05]
for a, (name, p) in zip(ax, preds.items()):
    a.scatter(y_test, p, s=8, alpha=0.6); a.plot(lim, lim, "k--", lw=0.8)
    a.set(xlim=lim, ylim=lim, xlabel="Measured (MPa)", title=f"{name}\nR² = {t2.loc[name, 'R2']:.3f}")
ax[0].set_ylabel("Predicted (MPa)")
fig.tight_layout(); fig.savefig(OUT / "fig4_measured_vs_predicted.png"); plt.close(fig)

xgb = models["XGBoost"]
pi = permutation_importance(xgb, X_test, y_test, n_repeats=30, random_state=42, scoring="r2")
imp = pd.DataFrame({"feature": X.columns, "mean_decrease_R2": pi.importances_mean,
                    "sd": pi.importances_std}).sort_values("mean_decrease_R2", ascending=False)
imp.round(4).to_csv(OUT / "permutation_importance.csv", index=False)
fig, a = plt.subplots(figsize=(4.5, 3), dpi=300)
i2 = imp.sort_values("mean_decrease_R2")
a.barh(i2.feature, i2.mean_decrease_R2, xerr=i2.sd); a.set_xlabel("Mean decrease in R² (30 repeats)")
fig.tight_layout(); fig.savefig(OUT / "fig6_permutation_importance.png"); plt.close(fig)
print("\nPermutation importance:\n", imp.round(3))
try:
    import shap
    sv = shap.TreeExplainer(xgb).shap_values(X_test)
    pd.DataFrame({"feature": X.columns, "mean_abs_SHAP": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_SHAP", ascending=False).round(4).to_csv(OUT / "shap_mean_abs.csv", index=False)
    shap.summary_plot(sv, X_test, show=False)
    plt.savefig(OUT / "fig5_shap_summary.png", dpi=300, bbox_inches="tight")
except ImportError:
    print("shap not installed - SHAP step skipped (pip install shap)")
