# XGBoost_FullMetrics.py
# -------------------------------------------------------------
# GPU XGBoost for Aurora Intensity Prediction (Full Metrics)
# Author: Group 10 (COMPSCI 760)
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -------------------------------------------------------------
# 0. Configs
# -------------------------------------------------------------
PIPELINE_MODE = True
REFIT_METRIC = "mae"
FAST_MODE = True
CPU_COUNT = os.cpu_count() or 2
N_JOBS = min(4, CPU_COUNT)
N_SPLITS = 2 if FAST_MODE else 3
N_ITER = 25 if FAST_MODE else 60
SAVE_DIR = "/kaggle/working/figs_xgb_fullmetrics"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Load dataset (Kaggle specific)
# -------------------------------------------------------------
CSV_PATH = "/kaggle/input/final-planb-24-preprocessed/final-planb-24_preprocessed.csv"

def _savefig(path, tight=True, dpi=150):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[Saved] {path}")

df = pd.read_csv(CSV_PATH, parse_dates=["time"])
print("Loaded:", CSV_PATH)
print("Shape:", df.shape)
print("Time range:", df["time"].min(), "→", df["time"].max())

# -------------------------------------------------------------
# 2. Define features + target
# -------------------------------------------------------------
TARGET_COL = "keogram_mean"
df = df.dropna(subset=[TARGET_COL]).copy()
drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]
features = [c for c in df.columns if c not in drop_cols]
X_all, y_all = df[features], df[TARGET_COL].values

# -------------------------------------------------------------
# 3. Time-based split
# -------------------------------------------------------------
train_idx = df[df["time"] < "2018-01-01"].index
val_idx   = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")].index
test_idx  = df[(df["time"] >= "2019-01-01") & (df["time"] < "2021-01-01")].index

X_train_df, X_val_df, X_test_df = X_all.loc[train_idx], X_all.loc[val_idx], X_all.loc[test_idx]
y_train = df.loc[train_idx, TARGET_COL].values
y_val   = df.loc[val_idx, TARGET_COL].values
y_test  = df.loc[test_idx, TARGET_COL].values
t_val  = df.loc[val_idx, "time"].values
t_test = df.loc[test_idx, "time"].values

if not PIPELINE_MODE:
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train_df)
    X_val   = imp.transform(X_val_df)
    X_test  = imp.transform(X_test_df)
else:
    X_train, X_val, X_test = X_train_df, X_val_df, X_test_df

# -------------------------------------------------------------
# 4. Model + Search (GPU)
# -------------------------------------------------------------
def _make_estimator():
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=42,
        n_jobs=N_JOBS,
        verbosity=0
    )
    if PIPELINE_MODE:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("xgb", base)])
    return base

def _prefix(name): return f"xgb__{name}" if PIPELINE_MODE else name

param_space = {
    _prefix("n_estimators"): [400, 800, 1200],
    _prefix("max_depth"): [4, 6, 8, 10],
    _prefix("learning_rate"): [0.03, 0.05, 0.1],
    _prefix("subsample"): [0.7, 0.9, 1.0],
    _prefix("colsample_bytree"): [0.7, 0.9, 1.0],
    _prefix("gamma"): [0, 0.1, 0.3],
    _prefix("reg_alpha"): [0, 0.1, 0.5],
    _prefix("reg_lambda"): [1, 2, 3],
    _prefix("min_child_weight"): [1, 3, 5],
}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

search = RandomizedSearchCV(
    estimator=_make_estimator(),
    param_distributions=param_space,
    n_iter=N_ITER,
    cv=tscv,
    scoring=scoring,
    refit=REFIT_METRIC,
    verbose=1,
    random_state=42,
)
search.fit(X_train, y_train)

best_est = search.best_estimator_
print("\nBest params:", search.best_params_)
print(f"Best CV {REFIT_METRIC.upper()}: {-search.best_score_:.4f}")

# -------------------------------------------------------------
# 5. Predictions
# -------------------------------------------------------------
val_pred = best_est.predict(X_val_df)
test_pred = best_est.predict(X_test_df)

# -------------------------------------------------------------
# 6. Extended Metrics (+ R² added)
# -------------------------------------------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def mase(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    return np.mean(np.abs(y_true - y_pred)) / d if d != 0 else np.nan

def nse(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - np.mean(y_true))**2)
    return 1 - num/den

def lag_corr(y_true, y_pred, max_lag=5):
    corrs = []
    for lag in range(1, max_lag + 1):
        if len(y_true) > lag:
            corrs.append(np.corrcoef(y_true[:-lag], y_pred[lag:])[0, 1])
    return np.nanmax(corrs)

def directional_accuracy(y_true, y_pred):
    dy_true = np.sign(np.diff(y_true))
    dy_pred = np.sign(np.diff(y_pred))
    return np.mean(dy_true == dy_pred)

def eval_all(name, y_true, y_pred, y_train):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    sm = smape(y_true, y_pred)
    ms = mase(y_true, y_pred, y_train)
    r, _ = pearsonr(y_true, y_pred)
    nse_val = nse(y_true, y_pred)
    lagc = lag_corr(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred)
    print(f"\n{name} Metrics:")
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} sMAPE={sm:.2f}% MASE={ms:.4f}")
    print(f"Pearson r={r:.4f} NSE={nse_val:.4f} LagCorr={lagc:.4f} DirAcc={da:.4f}")
    return [name, mae, rmse, r2, sm, ms, r, nse_val, lagc, da]

val_metrics = eval_all("VAL", y_val, val_pred, y_train)
test_metrics = eval_all("TEST", y_test, test_pred, y_train)

metrics_df = pd.DataFrame([val_metrics, test_metrics],
    columns=["Split","MAE","RMSE","R2","sMAPE","MASE","Pearson_r","NSE","LagCorr","DirAcc"])
metrics_path = os.path.join(SAVE_DIR, "extended_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\n[Saved] {metrics_path}")

# -------------------------------------------------------------
# 7. Visualizations
# -------------------------------------------------------------
def plot_pred_vs_actual(y_true, y_pred, split, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    minv, maxv = np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])
    plt.plot([minv, maxv], [minv, maxv], "r--")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"{split} — Predicted vs Actual")
    _savefig(save_path); plt.close()

def plot_time_series(t, y_true, y_pred, split, save_path):
    plt.figure(figsize=(10,4))
    plt.plot(t, y_true, label="Actual")
    plt.plot(t, y_pred, label="Predicted")
    plt.legend(); plt.title(f"{split} — Time Series")
    _savefig(save_path); plt.close()

def plot_residuals(y_true, y_pred, split, save_path):
    resid = y_pred - y_true
    plt.figure(figsize=(7,4))
    plt.hist(resid, bins=40, alpha=0.8)
    plt.xlabel("Residual"); plt.title(f"{split} — Residuals")
    _savefig(save_path); plt.close()

plot_pred_vs_actual(y_val, val_pred, "VAL", os.path.join(SAVE_DIR,"val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test, test_pred, "TEST", os.path.join(SAVE_DIR,"test_pred_vs_actual.png"))
plot_time_series(t_val, y_val, val_pred, "VAL", os.path.join(SAVE_DIR,"val_timeseries.png"))
plot_time_series(t_test, y_test, test_pred, "TEST", os.path.join(SAVE_DIR,"test_timeseries.png"))
plot_residuals(y_val, val_pred, "VAL", os.path.join(SAVE_DIR,"val_residuals.png"))
plot_residuals(y_test, test_pred, "TEST", os.path.join(SAVE_DIR,"test_residuals.png"))

# ---- Feature importance ----
model = best_est.named_steps["xgb"] if PIPELINE_MODE else best_est
imps = model.feature_importances_
top_idx = np.argsort(imps)[::-1][:15]
plt.figure(figsize=(8,5))
plt.barh(np.array(features)[top_idx][::-1], imps[top_idx][::-1])
plt.xlabel("Importance"); plt.title("Top 15 Feature Importances")
_savefig(os.path.join(SAVE_DIR,"feature_importance_top15.png")); plt.close()

print("\n✅ All metrics (including R²) computed and visualizations saved in:", SAVE_DIR)
