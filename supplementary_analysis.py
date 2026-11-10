#!/usr/bin/env python3
"""
Supplementary analysis for ICACE 2026 Paper 391 (reviewer response).

Keeps the ORIGINAL models and settings that produced Table 2 unchanged, and adds:
  - Check that Table 2 is reproduced (same train_test_split, same fixed settings)
  - Two extra non-ensemble nonlinear comparators (SVR, Decision Tree) on the same split  [Comment 5]
  - Group-aware 5-fold CV (mixture groups = identical composition across ages),
    repeated 3x, mean +/- SD of R2, MAE, MSE, RMSE                                 [Comments 2, 4]
  - Same CV without grouping (row-wise) to show the size of the leakage effect          [Comment 2]
  - Residual plots and paired-bootstrap 95% CI of RMSE differences vs XGBoost           [Comment 4]
  - Software versions, partition sizes, SCM-subset counts, filtered dataset            [Comments 1, 3, 6]

Usage:  python supplementary_analysis.py "concrete_data Clean-56 days.csv"
Runs in about 1-3 minutes. Outputs go to ./supp_results
"""
import json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("WARNING: xgboost not installed - XGBoost is skipped. Install with: pip install xgboost")

OUT = Path("supp_results"); OUT.mkdir(exist_ok=True)
df = pd.read_csv(sys.argv[1])
df.columns = [c.strip() for c in df.columns]
X, y = df.iloc[:, :-1], df.iloc[:, -1]
MIX = list(X.columns[:7])                      # composition only (age excluded)
groups = df.groupby([df[c].round(3) for c in MIX], sort=False).ngroup().values

# ---------------- models: ORIGINAL fixed settings (Table 2) + two comparators ----------------
models = {
    "LR":  LinearRegression(),
    "RF":  RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    "GBR": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=50),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.5)),
    "DT":  DecisionTreeRegressor(max_depth=10, random_state=42),
}
if HAVE_XGB:
    models["XGB"] = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)

def met(yt, p):
    mse = mean_squared_error(yt, p)
    return {"R2": r2_score(yt, p), "MAE": mean_absolute_error(yt, p), "MSE": mse, "RMSE": np.sqrt(mse)}

# ---------------- 1. original hold-out split (reproduces Table 2) ----------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
gtr, gte = groups[Xtr.index], groups[Xte.index]
preds, rows = {}, []
for n, m in models.items():
    preds[n] = clone(m).fit(Xtr, ytr).predict(Xte)
    rows.append({"Model": n, **met(yte, preds[n])})
hold = pd.DataFrame(rows).set_index("Model").round(4)
hold.to_csv(OUT / "holdout_metrics_original_split.csv")
print("\nHold-out (original split, n_test=%d):\n" % len(yte), hold.round(3))

# ---------------- 2. bootstrap CI of RMSE difference vs best model ----------------
ref = "XGB" if HAVE_XGB else hold["RMSE"].idxmin()
rng = np.random.default_rng(42); yv = yte.values
idx = [rng.integers(0, len(yv), len(yv)) for _ in range(2000)]
rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
boot = []
for n in preds:
    if n == ref: continue
    d = np.array([rmse(yv[i], preds[n][i]) - rmse(yv[i], preds[ref][i]) for i in idx])
    boot.append({"Comparison": f"{n} - {ref}", "dRMSE_mean": d.mean(),
                 "CI95_low": np.percentile(d, 2.5), "CI95_high": np.percentile(d, 97.5)})
pd.DataFrame(boot).round(3).to_csv(OUT / "holdout_bootstrap_dRMSE.csv", index=False)

# ---------------- 3. repeated 5-fold CV: grouped vs row-wise ----------------
def folds(n_rows, grp, seed, k=5):
    r = np.random.default_rng(seed)
    if grp is None:
        ids = r.permutation(n_rows); parts = np.array_split(ids, k)
        return [np.sort(p) for p in parts]
    ug = r.permutation(np.unique(grp)); parts = np.array_split(ug, k)
    return [np.where(np.isin(grp, p))[0] for p in parts]

cv_rows = []
for scheme, grp in [("grouped", groups), ("row-wise", None)]:
    for rep, seed in enumerate([42, 43, 44], 1):
        for k, te in enumerate(folds(len(df), grp, seed), 1):
            tr = np.setdiff1d(np.arange(len(df)), te)
            for n, m in models.items():
                p = clone(m).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
                cv_rows.append({"Scheme": scheme, "Repeat": rep, "Fold": k, "Model": n,
                                "n_train": len(tr), "n_test": len(te), **met(y.iloc[te], p)})
cv = pd.DataFrame(cv_rows); cv.round(4).to_csv(OUT / "cv_folds.csv", index=False)
summ = cv.groupby(["Scheme", "Model"])[["R2", "MAE", "MSE", "RMSE"]].agg(["mean", "std"]).round(3)
summ.to_csv(OUT / "cv_summary_mean_sd.csv")
print("\nRepeated 5-fold CV (mean, SD over 15 folds):\n", summ)
g = cv[cv.Scheme == "grouped"]
piv = g.pivot_table(index=["Repeat", "Fold"], columns="Model", values="RMSE")
wins = {n: f"{int((piv[n] > piv[ref]).sum())}/{len(piv)}" for n in piv if n != ref}
json.dump({"reference": ref, "folds_where_reference_has_lower_RMSE": wins},
          open(OUT / "grouped_cv_wins.json", "w"), indent=2)

# ---------------- 4. residual + measured-vs-predicted plots ----------------
names = list(preds)
fig, ax = plt.subplots(2, len(names), figsize=(2.4 * len(names), 4.8), dpi=300)
lim = [0, max(yte.max(), max(p.max() for p in preds.values())) * 1.05]
rmax = max(np.abs(yv - p).max() for p in preds.values()) * 1.05
for j, n in enumerate(names):
    a, b = ax[0, j], ax[1, j]
    a.scatter(yte, preds[n], s=6, alpha=.6); a.plot(lim, lim, "k--", lw=.8)
    a.set(xlim=lim, ylim=lim); a.set_title(f"{n}  R²={hold.loc[n,'R2']:.3f}", fontsize=8)
    b.scatter(preds[n], yv - preds[n], s=6, alpha=.6, color="tab:orange"); b.axhline(0, color="k", lw=.8)
    b.set_ylim(-rmax, rmax)
    a.set_xlabel("Measured (MPa)", fontsize=7); b.set_xlabel("Predicted (MPa)", fontsize=7)
    if j == 0: a.set_ylabel("Predicted (MPa)", fontsize=7); b.set_ylabel("Residual (MPa)", fontsize=7)
    for z in (a, b): z.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_measured_predicted_residuals.png"); plt.close(fig)

# ---------------- 5. partition sizes, SCM subset, versions, filtered data ----------------
scm = (df.iloc[:, 1] > 0) | (df.iloc[:, 2] > 0)
info = {
    "rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum()),
    "mixture_groups": int(len(np.unique(groups))),
    "holdout_train_rows": len(Xtr), "holdout_test_rows": len(Xte),
    "test_rows_whose_mixture_also_in_train": int(np.isin(gte, gtr).sum()),
    "grouped_cv_test_fold_sizes": cv[(cv.Scheme == "grouped") & (cv.Model == "LR")]["n_test"].tolist(),
    "rows_with_BFS_or_FA": int(scm.sum()),
    "rows_BFS_only": int(((df.iloc[:, 1] > 0) & (df.iloc[:, 2] == 0)).sum()),
    "rows_FA_only": int(((df.iloc[:, 1] == 0) & (df.iloc[:, 2] > 0)).sum()),
    "rows_BFS_and_FA": int(((df.iloc[:, 1] > 0) & (df.iloc[:, 2] > 0)).sum()),
    "rows_plain_cement": int((~scm).sum()),
}
json.dump(info, open(OUT / "dataset_and_partition_info.json", "w"), indent=2)
ver = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
       "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__ if HAVE_XGB else "not installed",
       "matplotlib": matplotlib.__version__}
json.dump(ver, open(OUT / "software_versions.json", "w"), indent=2)
df.to_csv(OUT / "filtered_dataset_used.csv", index=False)
print("\n", json.dumps(info, indent=1), "\n", ver, f"\nAll outputs in ./{OUT}")
