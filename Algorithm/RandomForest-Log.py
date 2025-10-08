# RandomForest.py
# -------------------------------------------------------------
# Baseline + Tuned Random Forest for aurora intensity prediction (with visuals)
# Author: Susie + Group 10 (COMPSCI 760) — enhanced with dual-search & pipeline
# -------------------------------------------------------------
# This script:
#   1. Loads the dataset (final3.csv)
#   2. Selects features and defines the target column
#   3. (Optional) Uses a Pipeline to avoid CV leakage by fitting the imputer inside CV
#   4. Splits the dataset into train/validation/test by year (time-aware split)
#   5. Runs THREE randomized searches:
#        [A] RandomForest (bootstrap=True, optional row-subsampling)
#        [B] RandomForest (bootstrap=False)
#        [C] ExtraTrees (extremely randomized trees)
#      and picks the best by a chosen CV metric (RMSE or MAE, configurable)
#   6. Evaluates the selected best model on VAL and TEST
#   7. Prints baselines, (optional) simple mean-ensemble, and feature importances
#   8. Saves visualizations & tables
#   9. (Optional) Classification view: Precision/Recall/F1 + PR curve
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------
# 0. Global knobs (you can tweak these)
# -------------------------------------------------------------
PIPELINE_MODE = True
REFIT_METRIC  = "mae"

FAST_MODE   = True
CPU_COUNT   = os.cpu_count() or 2
N_JOBS      = min(4, CPU_COUNT)
N_SPLITS    = 2 if FAST_MODE else 3
N_ITER_A    = 20 if FAST_MODE else 40
N_ITER_B    = 8  if FAST_MODE else 40
N_ITER_C    = max(8, N_ITER_B)
RUN_SEARCH_B = True
RUN_SEARCH_C = True

SAVE_DIR_NAME = "figs_rf"
SAVE_TABLES   = True

CLASSIFICATION_VIEW = False
CLASS_THRESH = 0.3
CLASS_THRESH_QUANTILE = None

# -------------------- [A2 CHANGE] NEW knobs --------------------
# Approach 2: retain outliers + feature engineering
USE_WINSOR_Y     = True      # winsorize target upper tail (train-only)
Y_WINSOR_UP_Q    = 0.97      # compress top 1% down to 99th pct
USE_LOG1P_TARGET = True      # train on log1p(y); inverse with expm1 for metrics/plots
USE_REWEIGHT     = True      # sample reweighting for rare/extreme events (train-only)
REWEIGHT_REF_Q   = 0.90      # start boosting at the 90th percentile
REWEIGHT_ALPHA   = 3.0       # max additional weight (1 -> 4x when rank=1)
REWEIGHT_GAMMA   = 2.0       # curvature; higher = focus more on extreme tail
# ----------------------------------------------------------------

# -------------------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "final-planb-24.csv")

SAVE_DIR = os.path.join(BASE_DIR, SAVE_DIR_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

def _savefig(path, tight=True, dpi=150):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[Saved] {path}")

df = pd.read_csv(CSV_PATH, parse_dates=["time"])
print("Loaded:", CSV_PATH)
print("Shape before drop:", df.shape)
print("Time range:", df["time"].min(), "->", df["time"].max(), flush=True)

# -------------------------------------------------------------
# 2. Define target and features
# -------------------------------------------------------------
TARGET_COL = "keogram_mean"

before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
after = len(df)
print(f"Dropped rows with NaN target ({TARGET_COL}): {before - after}", flush=True)

drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]
features = [c for c in df.columns if c not in drop_cols]
assert TARGET_COL not in features, "Leakage: target column is in features!"

X_all = df[features]
y_all_raw = df[TARGET_COL].values  # [A2 CHANGE] keep raw y for transforms

# -------------------------------------------------------------
# 3. Time-based train/val/test split
# -------------------------------------------------------------
# For dataset covering 2012–2024:
#   Train = 2012–2020
#   Validation = 2021-2022
#   Test = 2023–2024
train_idx = df[(df["time"] < "2021-01-01")].index
val_idx   = df[(df["time"] >= "2021-01-01") & (df["time"] < "2023-01-01")].index
test_idx  = df[(df["time"] >= "2023-01-01") & (df["time"] < "2025-01-01")].index

print("Split sizes:",
      "train =", len(train_idx),
      "val =", len(val_idx),
      "test =", len(test_idx), flush=True)

X_train_df = X_all.loc[train_idx]
X_val_df   = X_all.loc[val_idx]
X_test_df  = X_all.loc[test_idx]

# [A2 CHANGE] Slice raw target first (no transform yet)
y_train_raw = y_all_raw[df.index.get_indexer(train_idx)]
y_val_raw   = y_all_raw[df.index.get_indexer(val_idx)]
y_test_raw  = y_all_raw[df.index.get_indexer(test_idx)]

t_val  = df.loc[val_idx,  "time"].values
t_test = df.loc[test_idx, "time"].values

assert not np.isnan(y_train_raw).any()
assert not np.isnan(y_val_raw).any()
assert not np.isnan(y_test_raw).any()

if not PIPELINE_MODE:
    print("PIPELINE_MODE=False: imputing outside the model (slight CV leakage).")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df)
    X_val   = imputer.transform(X_val_df)
    X_test  = imputer.transform(X_test_df)
else:
    print("PIPELINE_MODE=True: imputer will be fitted inside CV folds via Pipeline.")
    X_train, X_val, X_test = X_train_df, X_val_df, X_test_df

# -------------------- [A2 CHANGE] Target engineering --------------------
# Fit transformation ONLY on TRAIN to avoid leakage, then apply to VAL/TEST.
y_cap = y_train_raw.copy()

# 1) Winsorize upper tail on train, then apply SAME cap to val/test (no fitting on them)
if USE_WINSOR_Y:
    up = np.quantile(y_train_raw, Y_WINSOR_UP_Q)
    y_cap = np.minimum(y_cap, up)
    y_val_cap  = np.minimum(y_val_raw,  up)
    y_test_cap = np.minimum(y_test_raw, up)
else:
    y_val_cap, y_test_cap = y_val_raw.copy(), y_test_raw.copy()

# 2) Log1p transform (train-fitted, deterministic for val/test)
def _fwd(y):  # forward transform
    return np.log1p(y) if USE_LOG1P_TARGET else y
def _inv(y):  # inverse transform for reporting/plots
    return np.expm1(y) if USE_LOG1P_TARGET else y

y_train = _fwd(y_cap)
y_val   = _fwd(y_val_cap)
y_test  = _fwd(y_test_cap)

# 3) Sample reweighting (train-only), smooth by percentile rank
if USE_REWEIGHT:
    ranks = pd.Series(y_train_raw).rank(method="average") / len(y_train_raw)
    # boost only above reference quantile
    boost_zone = np.clip((ranks - REWEIGHT_REF_Q) / (1 - REWEIGHT_REF_Q), 0, 1)
    sample_weight_train = 1.0 + REWEIGHT_ALPHA * (boost_zone ** REWEIGHT_GAMMA)
else:
    sample_weight_train = None
# ------------------------------------------------------------------------

# -------------------------------------------------------------
# 4. Helpers: sklearn version capability checks
# -------------------------------------------------------------
def _supports_params_rf(**kwargs) -> bool:
    try:
        RandomForestRegressor(random_state=0, n_jobs=1, **kwargs)
        return True
    except TypeError:
        return False

def _supports_params_et(**kwargs) -> bool:
    try:
        ExtraTreesRegressor(random_state=0, n_jobs=1, **kwargs)
        return True
    except TypeError:
        return False

HAS_ABS_CRITERION_RF = _supports_params_rf(criterion="absolute_error")
HAS_MAX_SAMPLES_RF   = _supports_params_rf(max_samples=0.8, bootstrap=True)
HAS_ABS_CRITERION_ET = _supports_params_et(criterion="absolute_error")

# -------------------------------------------------------------
# 5. THREE Randomized Searches (RF True/False + ExtraTrees)
# -------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
}
assert REFIT_METRIC in scoring

common_space = {
    "n_estimators":      [300, 600, 900] if FAST_MODE else [400, 800, 1200, 1600],
    "max_depth":         [None, 12, 24]  if FAST_MODE else [None, 8, 12, 16, 24, 32],
    "min_samples_split": [2, 10, 20]     if FAST_MODE else [2, 5, 10, 20, 40],
    "min_samples_leaf":  [2, 4, 8]       if FAST_MODE else [1, 2, 4, 8, 12, 16],
    "max_features":      ["sqrt", 0.5]   if FAST_MODE else ["sqrt", 0.3, 0.5, 0.7, 1.0],
}
CRITERIA_LIST = ["squared_error", "absolute_error"] if (HAS_ABS_CRITERION_RF or HAS_ABS_CRITERION_ET) else ["squared_error"]

def _make_estimator_rf():
    base = RandomForestRegressor(random_state=42, n_jobs=N_JOBS)
    if PIPELINE_MODE:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", base)])
    return base

def _make_estimator_et():
    base = ExtraTreesRegressor(random_state=42, n_jobs=N_JOBS)
    if PIPELINE_MODE:
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("et", base)])
    return base

def _prefix_rf(p): return f"rf__{p}" if PIPELINE_MODE else p
def _prefix_et(p): return f"et__{p}" if PIPELINE_MODE else p

# Fit-kwargs to pass sample weights through Pipeline into CV fits
# [A2 CHANGE] only provided when USE_REWEIGHT=True
fitkw_rf = {"rf__sample_weight": sample_weight_train} if (PIPELINE_MODE and USE_REWEIGHT) else ({ "sample_weight": sample_weight_train } if USE_REWEIGHT else {})
fitkw_et = {"et__sample_weight": sample_weight_train} if (PIPELINE_MODE and USE_REWEIGHT) else ({ "sample_weight": sample_weight_train } if USE_REWEIGHT else {})

# -------- [A] RF Search bootstrap=True --------
param_boot = {_prefix_rf(k): v for k, v in common_space.items()}
param_boot[_prefix_rf("criterion")] = CRITERIA_LIST
param_boot[_prefix_rf("bootstrap")] = [True]
if HAS_MAX_SAMPLES_RF:
    param_boot[_prefix_rf("max_samples")] = [0.7, 0.9, 1.0] if FAST_MODE else [0.5, 0.7, 0.9, 1.0]

search_boot = RandomizedSearchCV(
    estimator=_make_estimator_rf(),
    param_distributions=param_boot,
    n_iter=N_ITER_A, cv=tscv, scoring=scoring, refit=REFIT_METRIC,
    n_jobs=N_JOBS, verbose=1, random_state=42,
)
print("\n[Search A] RandomForest (bootstrap=True) ...")
search_boot.fit(X_train, y_train, **fitkw_rf)  # [A2 CHANGE] pass sample weights
print("  A: best params:", search_boot.best_params_)
print(f"  A: best CV {REFIT_METRIC.upper()}: {-search_boot.best_score_:.4f}")

# -------- [B] RF Search bootstrap=False --------
search_noboot = None
if RUN_SEARCH_B:
    param_noboot = {_prefix_rf(k): v for k, v in common_space.items()}
    param_noboot[_prefix_rf("criterion")] = CRITERIA_LIST
    param_noboot[_prefix_rf("bootstrap")] = [False]

    search_noboot = RandomizedSearchCV(
        estimator=_make_estimator_rf(),
        param_distributions=param_noboot,
        n_iter=N_ITER_B, cv=tscv, scoring=scoring, refit=REFIT_METRIC,
        n_jobs=N_JOBS, verbose=1, random_state=43,
    )
    print("\n[Search B] RandomForest (bootstrap=False) ...")
    try:
        search_noboot.fit(X_train, y_train, **fitkw_rf)  # [A2 CHANGE]
        print("  B: best params:", search_noboot.best_params_)
        print(f"  B: best CV {REFIT_METRIC.upper()}: {-search_noboot.best_score_:.4f}")
    except KeyboardInterrupt:
        print("  B: interrupted by user; falling back to Search A.")
        search_noboot = None

# -------- [C] ExtraTrees Search --------
search_et = None
if RUN_SEARCH_C:
    try:
        param_et = {_prefix_et(k): v for k, v in common_space.items()}
        et_criteria = ["squared_error", "absolute_error"] if HAS_ABS_CRITERION_ET else ["squared_error"]
        param_et[_prefix_et("criterion")] = et_criteria

        search_et = RandomizedSearchCV(
            estimator=_make_estimator_et(),
            param_distributions=param_et,
            n_iter=N_ITER_C, cv=tscv, scoring=scoring, refit=REFIT_METRIC,
            n_jobs=N_JOBS, verbose=1, random_state=44,
        )
        print("\n[Search C] ExtraTrees ...")
        search_et.fit(X_train, y_train, **fitkw_et)  # [A2 CHANGE]
        print("  C: best params:", search_et.best_params_)
        print(f"  C: best CV {REFIT_METRIC.upper()}: {-search_et.best_score_:.4f}")
    except KeyboardInterrupt:
        print("  C: interrupted by user; skipping ExtraTrees.")
        search_et = None
    except Exception as e:
        print(f"  C: skipped due to error: {e}")
        search_et = None

# -------- Select overall best --------
candidates = [s for s in (search_boot, search_noboot, search_et) if s is not None]
best_search = max(candidates, key=lambda s: s.best_score_)
print("\n[Selection] Choosing the overall better search by CV metric ...")
print("  Selected params:", best_search.best_params_)
print(f"  Selected CV {REFIT_METRIC.upper()}: {-best_search.best_score_:.4f}")
best_estimator = best_search.best_estimator_

# -------------------------------------------------------------
# 6. Evaluate performance (inverse-transform predictions)
# -------------------------------------------------------------
def eval_and_print(split_name, y_true_raw, y_pred_transformed):
    """y_true_raw is on raw scale; y_pred_transformed is on transformed scale."""
    y_pred_raw = _inv(y_pred_transformed)  # [A2 CHANGE] bring back to raw scale
    mse = mean_squared_error(y_true_raw, y_pred_raw)
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2  = r2_score(y_true_raw, y_pred_raw)
    print(f"{split_name} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}", flush=True)
    return y_pred_raw

print("\n=== Evaluation (Selected Best Model) ===", flush=True)
val_pred_t  = best_estimator.predict(X_val_df)
test_pred_t = best_estimator.predict(X_test_df)

# Back to raw for all downstream plots/metrics
val_pred_raw  = eval_and_print("VAL ",  y_val_raw,  val_pred_t)
test_pred_raw = eval_and_print("TEST",  y_test_raw, test_pred_t)

# -------------------------------------------------------------
# Baselines: mean and median (on RAW scale)
# -------------------------------------------------------------
def baseline_report(y_true_raw, name="mean"):
    if name == "mean":
        yhat = np.full_like(y_true_raw, fill_value=np.mean(y_true_raw), dtype=float)
    elif name == "median":
        yhat = np.full_like(y_true_raw, fill_value=np.median(y_true_raw), dtype=float)
    else:
        raise ValueError("name must be 'mean' or 'median'")
    mse = mean_squared_error(y_true_raw, yhat)
    mae = mean_absolute_error(y_true_raw, yhat)
    rmse = np.sqrt(mse)
    print(f"Baseline ({name}) -> RMSE: {rmse:.4f}  MAE: {mae:.4f}")

print("\n=== Baselines ===")
baseline_report(y_val_raw,  "mean")
baseline_report(y_val_raw,  "median")
baseline_report(y_test_raw, "mean")
baseline_report(y_test_raw, "median")

# -------------------------------------------------------------
# Optional: Simple mean-ensemble across A/B/C
# -------------------------------------------------------------
def _predict_auto(est, X_df, X_arr):
    return est.predict(X_df) if PIPELINE_MODE else est.predict(X_arr)

avail_ests = [s.best_estimator_ for s in (search_boot, search_noboot, search_et) if s is not None]

if len(avail_ests) >= 2:
    print("\n[Ensemble] Evaluating simple mean-ensemble of available best estimators ...")
    val_preds_t  = np.column_stack([_predict_auto(e, X_val_df,  X_val)  for e in avail_ests]).mean(axis=1)
    test_preds_t = np.column_stack([_predict_auto(e, X_test_df, X_test) for e in avail_ests]).mean(axis=1)

    def _eval(y_true_raw, y_pred_t, name):
        y_pred_raw = _inv(y_pred_t)  # [A2 CHANGE]
        mse = mean_squared_error(y_true_raw, y_pred_raw)
        mae = mean_absolute_error(y_true_raw, y_pred_raw)
        r2  = r2_score(y_true_raw, y_pred_raw)
        print(f"{name} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")
        return y_pred_raw

    val_ens_raw  = _eval(y_val_raw,  val_preds_t,  "VAL  Ensemble")
    test_ens_raw = _eval(y_test_raw, test_preds_t, "TEST Ensemble")
    ENSEMBLE_RESULTS = (_inv(val_preds_t), _inv(test_preds_t))  # raw scale
else:
    ENSEMBLE_RESULTS = None

# -------------------------------------------------------------
# Pretty comparison table (RAW scale)
# -------------------------------------------------------------
def _metrics(y_true_raw, y_pred_raw):
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    return rmse, mae

rows = []
rmse_m, mae_m = _metrics(y_val_raw,  val_pred_raw)
rows.append(("VAL",  "Model", rmse_m, mae_m))
rows.append(("VAL",  "Baseline-mean",
             np.sqrt(mean_squared_error(y_val_raw,  np.full_like(y_val_raw, y_val_raw.mean(), dtype=float))),
             mean_absolute_error(y_val_raw, np.full_like(y_val_raw, y_val_raw.mean(), dtype=float))))
rows.append(("VAL",  "Baseline-median",
             np.sqrt(mean_squared_error(y_val_raw,  np.full_like(y_val_raw, np.median(y_val_raw), dtype=float))),
             mean_absolute_error(y_val_raw, np.full_like(y_val_raw, np.median(y_val_raw), dtype=float))))

rmse_m, mae_m = _metrics(y_test_raw, test_pred_raw)
rows.append(("TEST", "Model", rmse_m, mae_m))
rows.append(("TEST", "Baseline-mean",
             np.sqrt(mean_squared_error(y_test_raw, np.full_like(y_test_raw, y_test_raw.mean(), dtype=float))),
             mean_absolute_error(y_test_raw, np.full_like(y_test_raw, y_test_raw.mean(), dtype=float))))
rows.append(("TEST", "Baseline-median",
             np.sqrt(mean_squared_error(y_test_raw, np.full_like(y_test_raw, np.median(y_test_raw), dtype=float))),
             mean_absolute_error(y_test_raw, np.full_like(y_test_raw, np.median(y_test_raw), dtype=float))))

if ENSEMBLE_RESULTS is not None:
    val_ens_raw, test_ens_raw = ENSEMBLE_RESULTS
    rmse_v, mae_v = _metrics(y_val_raw,  val_ens_raw)
    rmse_t, mae_t = _metrics(y_test_raw, test_ens_raw)
    rows.append(("VAL",  "Ensemble", rmse_v, mae_v))
    rows.append(("TEST", "Ensemble", rmse_t, mae_t))

comparison_df = pd.DataFrame(rows, columns=["Split", "Method", "RMSE", "MAE"])
print(comparison_df.to_string(index=False))

if SAVE_TABLES:
    comp_csv  = os.path.join(SAVE_DIR, "comparison_metrics.csv")
    comp_html = os.path.join(SAVE_DIR, "comparison_metrics.html")
    comparison_df.to_csv(comp_csv, index=False)
    comparison_df.to_html(comp_html, index=False)
    print(f"[Saved] {comp_csv}")
    print(f"[Saved] {comp_html}")

# -------------------------------------------------------------
# 7. Feature importance analysis (+ bar plot)
# -------------------------------------------------------------
def _extract_model_and_importances(estimator):
    model = estimator
    if hasattr(estimator, "named_steps"):
        if "rf" in estimator.named_steps:
            model = estimator.named_steps["rf"]
        elif "et" in estimator.named_steps:
            model = estimator.named_steps["et"]
        else:
            model = None
    if model is not None and hasattr(model, "feature_importances_"):
        return model, model.feature_importances_
    return model, None

underlying_model, importances = _extract_model_and_importances(best_estimator)

if importances is None:
    print("\n[Info] Feature importances not available (estimator doesn't expose them).")
else:
    order = np.argsort(importances)[::-1][:15]
    top_feats = [(features[i], float(importances[i])) for i in order]
    print("\nTop-15 feature importances:")
    for name, score in top_feats:
        print(f"{name:20s}  {score:.4f}")

    plt.figure(figsize=(8, 5))
    names = [x[0] for x in top_feats]
    vals  = [x[1] for x in top_feats]
    ax = plt.gca()
    ax.barh(names[::-1], vals[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top-15 Feature Importances")
    _savefig(os.path.join(SAVE_DIR, "feature_importance_top15.png"))
    plt.close()

# -------------------------------------------------------------
# 8. Visualizations (RAW scale)
# -------------------------------------------------------------
def plot_pred_vs_actual(y_true_raw, y_pred_raw, split, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_raw, y_pred_raw, s=10, alpha=0.6)
    minv = float(np.nanmin([y_true_raw.min(), y_pred_raw.min()]))
    maxv = float(np.nanmax([y_true_raw.max(), y_pred_raw.max()]))
    plt.plot([minv, maxv], [minv, maxv], linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{split} — Predicted vs. Actual")
    _savefig(save_path)
    plt.close()

def plot_time_series(t, y_true_raw, y_pred_raw, split, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_true_raw, linewidth=1, label="Actual")
    plt.plot(t, y_pred_raw, linewidth=1, alpha=0.9, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COL)
    plt.title(f"{split} — Time Series: Actual vs. Predicted")
    plt.legend()
    _savefig(save_path)
    plt.close()

def plot_residuals_hist(y_true_raw, y_pred_raw, split, save_path):
    resid = y_pred_raw - y_true_raw
    plt.figure(figsize=(7, 4))
    plt.hist(resid, bins=40, alpha=0.8)
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Count")
    plt.title(f"{split} — Residuals Histogram")
    _savefig(save_path)
    plt.close()

def plot_bar_metric_comparison(df_metrics, split, save_path):
    sub = df_metrics[df_metrics["Split"] == split].copy()
    labels = sub["Method"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig = plt.figure(figsize=(9, 4.5))
    ax = plt.gca()
    ax.bar(x - width/2, sub["RMSE"].values, width, label="RMSE")
    ax.bar(x + width/2, sub["MAE"].values,  width, label="MAE")
    ax.set_xticks(x, labels, rotation=30, ha='right')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.set_title(f"{split} — Model vs Baselines")
    ax.legend()
    _savefig(save_path)
    plt.close()

plot_pred_vs_actual(y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_pred_vs_actual.png"))

plot_time_series(t_val,  y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_timeseries.png"))
plot_time_series(t_test, y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_timeseries.png"))

plot_residuals_hist(y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_residuals_hist.png"))
plot_residuals_hist(y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_residuals_hist.png"))

plot_bar_metric_comparison(comparison_df, "VAL",  os.path.join(SAVE_DIR, "val_model_vs_baselines.png"))
plot_bar_metric_comparison(comparison_df, "TEST", os.path.join(SAVE_DIR, "test_model_vs_baselines.png"))

# -------------------------------------------------------------
# 9. Optional classification view (unchanged, but uses RAW preds)
# -------------------------------------------------------------
def _maybe_get_threshold(y_true):
    if CLASS_THRESH_QUANTILE is not None:
        return float(np.quantile(y_true, CLASS_THRESH_QUANTILE))
    return CLASS_THRESH

if CLASSIFICATION_VIEW:
    thr_val  = _maybe_get_threshold(y_val_raw)   # [A2 CHANGE] use raw
    thr_test = _maybe_get_threshold(y_test_raw)

    y_val_bin  = (y_val_raw  >= thr_val).astype(int)
    y_test_bin = (y_test_raw >= thr_test).astype(int)

    val_pred_bin  = (val_pred_raw  >= thr_val).astype(int)
    test_pred_bin = (test_pred_raw >= thr_test).astype(int)

    def _cls_report(split, y_true_bin, y_pred_bin, scores_raw, save_prefix):
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        print(f"[{split} Classification] Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        p, r, _ = precision_recall_curve(y_true_bin, scores_raw)
        plt.figure(figsize=(6, 5))
        plt.plot(r, p, linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{split} — Precision-Recall Curve")
        _savefig(os.path.join(SAVE_DIR, f"{save_prefix}_pr_curve.png"))
        plt.close()

    _cls_report("VAL",  y_val_bin,  val_pred_bin,  val_pred_raw,  "val")
    _cls_report("TEST", y_test_bin, test_pred_bin, test_pred_raw, "test")
else:
    print("\n[Info] Classification view disabled (set CLASSIFICATION_VIEW=True to compute Recall/F1 & PR curves).")
