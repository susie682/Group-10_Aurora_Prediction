#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CatBoost.py — Fast & Safe HPO  (dataset switched to final-planb-24_preprocessed.csv)

import os, sys, subprocess, numpy as np, pandas as pd
try:
    from catboost import CatBoostRegressor
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "catboost>=1.2"])
    from catboost import CatBoostRegressor

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===================== Global Configs ========================
REFIT_METRIC = "mae"
CPU_COUNT    = os.cpu_count() or 2
N_JOBS       = min(4, CPU_COUNT)
N_SPLITS     = 2

SAVE_DIR     = "figs_catboost"
TARGET_COL   = "keogram_mean"

# HPO proxy & search space
MAX_HPO_ROWS        = 20000
HPO_DOWNSAMPLE_STEP = 2
N_ITER_HPO          = 20
ITERATIONS_RANGE    = [300, 600, 900]

# ===================== Paths & I/O ===========================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# >>> Switched dataset here <<<
CSV_PATH_OVERRIDE =  r"D:\760\final-planb-24_preprocessed.csv"   # fixed to your uploaded dataset
# CSV_PATH_OVERRIDE = None

CANDIDATE_PATHS = [
    CSV_PATH_OVERRIDE,
    os.path.join(BASE_DIR, "final-planb-24_preprocessed.csv"),
    os.path.join(BASE_DIR, "datasets", "final-planb-24_preprocessed.csv"),
    os.path.join(BASE_DIR, "..", "datasets", "final-planb-24_preprocessed.csv"),
    # optional fallbacks (in case you move files back to older name):
    os.path.join(BASE_DIR, "final-planb-24.csv"),
    os.path.join(BASE_DIR, "datasets", "final-planb-24.csv"),
    "final-planb-24_preprocessed.csv",
    "final-planb-24.csv",
]
CANDIDATE_PATHS = [p for p in CANDIDATE_PATHS if p]
CSV_PATH = next((p for p in CANDIDATE_PATHS if p and os.path.exists(p)), None)
if CSV_PATH is None:
    raise FileNotFoundError(
        "Could not find 'final-planb-24_preprocessed.csv'. "
        "Set CSV_PATH_OVERRIDE or place the file in ./ or ./datasets/."
    )

OUT_DIR = os.path.join(BASE_DIR, SAVE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

def _savefig(path, tight=True, dpi=150):
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=dpi); print("[Saved]", path)

# ======================= Load & Prep =========================
df = pd.read_csv(CSV_PATH, parse_dates=["time"])
print("Loaded:", CSV_PATH, "| shape:", df.shape, "| time:", df["time"].min(), "→", df["time"].max())

# drop rows with missing target
df = df.dropna(subset=[TARGET_COL]).copy()

# feature picking: keep numeric columns except explicitly dropped ones
drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]
num_cols  = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    raise ValueError("No numeric features after filtering. Please check dataset columns.")
features = num_cols

X_all = df[features].astype(np.float32)
y_all = df[TARGET_COL].astype(np.float32).values

# ===================== Time-aware Splits =====================
train_idx = df[df["time"] <  "2018-01-01"].index
val_idx   = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")].index
test_idx  = df[(df["time"] >= "2019-01-01") & (df["time"] < "2021-01-01")].index
print("Split sizes:", "train=",len(train_idx), "val=",len(val_idx), "test=",len(test_idx))

X_train_df, X_val_df, X_test_df = X_all.loc[train_idx], X_all.loc[val_idx], X_all.loc[test_idx]
y_train = y_all[df.index.get_indexer(train_idx)]
y_val   = y_all[df.index.get_indexer(val_idx)]
y_test  = y_all[df.index.get_indexer(test_idx)]
t_val   = df.loc[val_idx,  "time"].values
t_test  = df.loc[test_idx, "time"].values

# ==================== HPO Proxy Subset =======================
order = X_train_df.index
if HPO_DOWNSAMPLE_STEP > 1:
    order = order[::HPO_DOWNSAMPLE_STEP]
if len(order) > MAX_HPO_ROWS:
    order = order[:MAX_HPO_ROWS]

X_train_hpo = X_train_df.loc[order]
y_train_hpo = y_train[np.isin(X_train_df.index.values, order)]
print(f"[HPO proxy] rows={len(X_train_hpo)} from {len(X_train_df)} | step={HPO_DOWNSAMPLE_STEP} | cap={MAX_HPO_ROWS}")

# ================== CV, Estimator, Search ====================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
assert REFIT_METRIC in scoring

def make_estimator():
    base = CatBoostRegressor(
        loss_function="RMSE",
        boosting_type="Ordered",
        random_state=42, verbose=0,
        thread_count=max(1, (CPU_COUNT or 2)-1),
        allow_const_label=True
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("cat", base),
    ])

def p(k): return f"cat__{k}"

param_space = {
    p("iterations"):       ITERATIONS_RANGE,
    p("learning_rate"):    [0.03, 0.05, 0.07, 0.10],
    p("depth"):            [4, 6, 8],
    p("l2_leaf_reg"):      [3, 5, 7, 10],
    p("min_data_in_leaf"): [5, 10, 20],
    p("random_strength"):  [0.0, 0.5, 1.0],
}

search = RandomizedSearchCV(
    estimator=make_estimator(),
    param_distributions=param_space,
    n_iter=N_ITER_HPO,
    cv=tscv,
    scoring=scoring,
    refit=REFIT_METRIC,
    n_jobs=N_JOBS,
    verbose=1,
    random_state=2024,
    error_score=np.nan,
)

print("\n[HPO] RandomizedSearch (Ordered, safe space) ...")
search.fit(X_train_hpo, y_train_hpo)
if not np.isfinite(search.best_score_):
    raise RuntimeError("HPO failed to produce a valid configuration. Try increasing N_ITER_HPO or proxy size.")
print("Best params:", search.best_params_)
print(f"Best CV {REFIT_METRIC.upper()}: {-search.best_score_:.5f}")

# ================= Final train on FULL train =================
best_params = search.best_params_
final_model = make_estimator()
final_model.set_params(**best_params)

# early stopping settings (unchanged)
final_model.set_params(
    cat__use_best_model=True,
    cat__od_type="Iter",
    cat__od_wait=60
)

fit_params = {
    "cat__eval_set": (X_val_df, y_val),
    "cat__verbose": False
}
final_model.fit(X_train_df, y_train, **fit_params)

# ===================== Predict & Evaluate ====================
val_pred  = final_model.predict(X_val_df)
test_pred = final_model.predict(X_test_df)

def report(name, yt, yp):
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    r2   = r2_score(yt, yp)
    print(f"{name} -> RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

print("\n=== Evaluation (final with early stopping) ===")
report("VAL ", y_val,  val_pred)
report("TEST", y_test, test_pred)

# ================== Baselines & Table ========================
rows=[]
def metrics(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred)

rmse, mae = metrics(y_val, val_pred)
rows += [("VAL","CatBoost-Model",rmse,mae),
         ("VAL","Baseline-mean",
          np.sqrt(mean_squared_error(y_val, np.full_like(y_val, y_val.mean(), dtype=float))),
          mean_absolute_error(y_val, np.full_like(y_val, y_val.mean(), dtype=float))),
         ("VAL","Baseline-median",
          np.sqrt(mean_squared_error(y_val, np.full_like(y_val, np.median(y_val), dtype=float))),
          mean_absolute_error(y_val, np.full_like(y_val, np.median(y_val), dtype=float)))]

rmse, mae = metrics(y_test, test_pred)
rows += [("TEST","CatBoost-Model",rmse,mae),
         ("TEST","Baseline-mean",
          np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_test.mean(), dtype=float))),
          mean_absolute_error(y_test, np.full_like(y_test, y_test.mean(), dtype=float))),
         ("TEST","Baseline-median",
          np.sqrt(mean_squared_error(y_test, np.full_like(y_test, np.median(y_test), dtype=float))),
          mean_absolute_error(y_test, np.full_like(y_test, np.median(y_test), dtype=float)))]

comparison_df = pd.DataFrame(rows, columns=["Split","Method","RMSE","MAE"])
print("\n", comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(OUT_DIR, "comparison_metrics.csv"), index=False)
comparison_df.to_html(os.path.join(OUT_DIR, "comparison_metrics.html"), index=False)

# =================== Importance & Plots ======================
cat = final_model.named_steps["cat"]
imps = cat.get_feature_importance()
order = np.argsort(imps)[::-1][:15]
names = [features[i] for i in order]
vals  = [float(imps[i]) for i in order]

print("\nTop-15 feature importances:")
for n, s in zip(names, vals):
    print(f"{n:20s} {s:.4f}")

plt.figure(figsize=(8,5))
plt.barh(names[::-1], vals[::-1])
plt.xlabel("Importance"); plt.title("Top-15 Feature Importances (CatBoost)")
_savefig(os.path.join(OUT_DIR, "feature_importance_top15.png")); plt.close()

def plot_pred_vs_actual(y_true, y_pred, split, path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([mn,mx],[mn,mx],'--')
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"{split} — Pred vs Actual")
    _savefig(path); plt.close()

def plot_time_series(t, y_true, y_pred, split, path):
    plt.figure(figsize=(10,4))
    plt.plot(t, y_true, lw=1, label="Actual")
    plt.plot(t, y_pred, lw=1, label="Predicted")
    plt.xlabel("Time"); plt.ylabel(TARGET_COL); plt.title(f"{split} — Time Series"); plt.legend()
    _savefig(path); plt.close()

def plot_residuals_hist(y_true, y_pred, split, path):
    plt.figure(figsize=(7,4))
    plt.hist((y_pred - y_true), bins=40, alpha=0.8)
    plt.xlabel("Residual (Pred-Actual)"); plt.ylabel("Count"); plt.title(f"{split} — Residuals")
    _savefig(path); plt.close()

def plot_bar_metric_comparison(dfm, split, path):
    sub = dfm[dfm["Split"]==split].copy()
    labels=sub["Method"].tolist(); x=np.arange(len(labels)); w=0.38
    plt.figure(figsize=(9,4.5)); ax=plt.gca()
    ax.bar(x-w/2, sub["RMSE"].values, w, label="RMSE")
    ax.bar(x+w/2, sub["MAE"].values,  w, label="MAE")
    ax.set_xticks(x, labels, rotation=30, ha='right')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_title(f"{split} — Model vs Baselines"); ax.legend()
    _savefig(path); plt.close()

plot_pred_vs_actual(y_val,  val_pred,  "VAL",  os.path.join(OUT_DIR, "val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test, test_pred, "TEST", os.path.join(OUT_DIR, "test_pred_vs_actual.png"))
plot_time_series(t_val,  y_val,  val_pred,  "VAL",  os.path.join(OUT_DIR, "val_timeseries.png"))
plot_time_series(t_test, y_test, test_pred, "TEST", os.path.join(OUT_DIR, "test_timeseries.png"))
plot_residuals_hist(y_val,  val_pred,  "VAL",  os.path.join(OUT_DIR, "val_residuals_hist.png"))
plot_residuals_hist(y_test, test_pred, "TEST", os.path.join(OUT_DIR, "test_residuals_hist.png"))
plot_bar_metric_comparison(comparison_df, "VAL",  os.path.join(OUT_DIR, "val_model_vs_baselines.png"))
plot_bar_metric_comparison(comparison_df, "TEST", os.path.join(OUT_DIR, "test_model_vs_baselines.png"))

