# RandomForest.py
# -------------------------------------------------------------
# Baseline + Tuned Random Forest for aurora intensity prediction (with visuals)
# Author: Susie + Group 10 (COMPSCI 760)  — enhanced with dual-search & pipeline
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
#   8. NEW: Saves visualizations (pred vs. actual, residuals, feature importance bar),
#           and exports comparison tables (CSV/HTML). Optional classification view
#           computes Precision/Recall/F1 + PR curve if you enable it.
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor  # RF + ET (NEW)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.pipeline import Pipeline  # pipeline support to avoid CV leakage

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------
# 0. Global knobs (you can tweak these)
# -------------------------------------------------------------
# Use a Pipeline (imputer inside) to avoid CV leakage. Recommended=True.
PIPELINE_MODE = True

# Choose which metric to "refit"/select the best model on: "rmse" or "mae"
# Tip: under heavy-tailed noise, "mae" is often more robust.
REFIT_METRIC = "mae"

# Runtime knobs — keep searches light on laptops
FAST_MODE   = True                            # True = smaller grids and fewer iterations
CPU_COUNT   = os.cpu_count() or 2
N_JOBS      = min(4, CPU_COUNT)               # limit parallelism to avoid OS thrashing
N_SPLITS    = 2 if FAST_MODE else 3           # fewer CV splits in fast mode
N_ITER_A    = 20 if FAST_MODE else 40         # iterations for RF bootstrap=True
N_ITER_B    = 8  if FAST_MODE else 40         # iterations for RF bootstrap=False
N_ITER_C    = max(8, N_ITER_B)                # iterations for ExtraTrees (NEW)
RUN_SEARCH_B = True                            # set False to skip Search B entirely
RUN_SEARCH_C = True                            # set False to skip ExtraTrees search

# ---- NEW: Visualization & output knobs ----
SAVE_DIR_NAME = "figs_rf"
SAVE_TABLES   = True

# ---- NEW: Optional "classification view" (for Recall/F1) ----
# Set to True to evaluate a binary event like "aurora intensity >= threshold".
# This does NOT change training (still regression); it only evaluates classification metrics post-hoc.
CLASSIFICATION_VIEW = False
# Threshold for positive class (you can tune; e.g., choose a physical intensity, or a quantile).
CLASS_THRESH = 0.3      # example value; adjust to your scale
# Or you can set as quantile if you prefer (set to None to disable quantile override)
CLASS_THRESH_QUANTILE = None  # e.g., 0.8 means top 20% treated as positive

# -------------------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path of this script
CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "final-planb-24.csv")

# Prepare output dir
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
# Final target = "keogram_mean" (ground-based aurora intensity)
TARGET_COL = "keogram_mean"

# 1) Drop rows where the target is NaN (models cannot train on NaN y)
before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
after = len(df)
print(f"Dropped rows with NaN target ({TARGET_COL}): {before - after}", flush=True)

# 2) Columns that should NOT be used as input features
#    - time: timestamp (index-like)
#    - Kp/ap: you decided to exclude geomagnetic indices
#    - keogram_*: siblings of the target (never include to avoid leakage)
drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]

# 3) Candidate features = everything else
features = [c for c in df.columns if c not in drop_cols]

# Final safety check: target must not appear in features
assert TARGET_COL not in features, "Leakage: target column is in features!"

# Keep X as a DataFrame; whether we impute here or inside Pipeline depends on PIPELINE_MODE
X_all = df[features]
y_all = df[TARGET_COL].values

# -------------------------------------------------------------
# 3. Time-based train/val/test split
# -------------------------------------------------------------
# For dataset covering 2012–2020:
#   Train = 2012–2017
#   Validation = 2018
#   Test = 2019–2020
train_idx = df[(df["time"] < "2018-01-01")].index
val_idx   = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")].index
test_idx  = df[(df["time"] >= "2019-01-01") & (df["time"] < "2021-01-01")].index

print("Split sizes:",
      "train =", len(train_idx),
      "val =", len(val_idx),
      "test =", len(test_idx), flush=True)

# Slice the raw (non-imputed) frames. If PIPELINE_MODE=True, the imputer will run inside the Pipeline.
X_train_df = X_all.loc[train_idx]
X_val_df   = X_all.loc[val_idx]
X_test_df  = X_all.loc[test_idx]

y_train = y_all[df.index.get_indexer(train_idx)]
y_val   = y_all[df.index.get_indexer(val_idx)]
y_test  = y_all[df.index.get_indexer(test_idx)]

t_val  = df.loc[val_idx,  "time"].values
t_test = df.loc[test_idx, "time"].values

# Safety check: make sure targets have no NaN
assert not np.isnan(y_train).any(), "y_train still has NaN!"
assert not np.isnan(y_val).any(),   "y_val still has NaN!"
assert not np.isnan(y_test).any(),  "y_test still has NaN!"

# If PIPELINE_MODE=False, we impute outside (older way). If True, skip here.
if not PIPELINE_MODE:
    print("PIPELINE_MODE=False: imputing outside the model (slight CV leakage).")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df)
    X_val   = imputer.transform(X_val_df)
    X_test  = imputer.transform(X_test_df)
else:
    print("PIPELINE_MODE=True: imputer will be fitted inside CV folds via Pipeline.")
    X_train, X_val, X_test = X_train_df, X_val_df, X_test_df

# -------------------------------------------------------------
# 4. Helpers: sklearn version capability checks
# -------------------------------------------------------------
def _supports_params_rf(**kwargs) -> bool:
    """Return True if RandomForestRegressor accepts the given kwargs in this sklearn version."""
    try:
        RandomForestRegressor(random_state=0, n_jobs=1, **kwargs)
        return True
    except TypeError:
        return False

def _supports_params_et(**kwargs) -> bool:
    """Return True if ExtraTreesRegressor accepts the given kwargs in this sklearn version."""
    try:
        ExtraTreesRegressor(random_state=0, n_jobs=1, **kwargs)
        return True
    except TypeError:
        return False

HAS_ABS_CRITERION_RF = _supports_params_rf(criterion="absolute_error")
HAS_MAX_SAMPLES_RF   = _supports_params_rf(max_samples=0.8, bootstrap=True)
HAS_ABS_CRITERION_ET = _supports_params_et(criterion="absolute_error")

# -------------------------------------------------------------
# 5. THREE Randomized Searches (RF True/False + ExtraTrees) — Compute-friendly
# -------------------------------------------------------------
# 5.1 Time-series cross-validation (preserves temporal order)
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# 5.2 Scoring dict (built-in strings avoid parameter-routing issues)
#     NOTE: these are negated scores by sklearn convention (higher is better).
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
}
assert REFIT_METRIC in scoring, "REFIT_METRIC must be 'rmse' or 'mae'"

# 5.3 Common hyperparameter search space (true tree hyperparameters)
#     Kept compact in FAST_MODE to save compute.
common_space = {
    "n_estimators":      [300, 600, 900] if FAST_MODE else [400, 800, 1200, 1600],
    "max_depth":         [None, 12, 24]  if FAST_MODE else [None, 8, 12, 16, 24, 32],
    "min_samples_split": [2, 10, 20]     if FAST_MODE else [2, 5, 10, 20, 40],
    "min_samples_leaf":  [2, 4, 8]       if FAST_MODE else [1, 2, 4, 8, 12, 16],
    "max_features":      ["sqrt", 0.5]   if FAST_MODE else ["sqrt", 0.3, 0.5, 0.7, 1.0],
}
# Try absolute-error splits when supported (robust for heavy-tailed noise)
CRITERIA_LIST = ["squared_error", "absolute_error"] if (HAS_ABS_CRITERION_RF or HAS_ABS_CRITERION_ET) else ["squared_error"]

# ---- Utilities to build estimators & param prefixes (RF / ET), Pipeline-aware ----
def _make_estimator_rf():
    """Build an RF estimator (Pipeline if PIPELINE_MODE)."""
    base = RandomForestRegressor(random_state=42, n_jobs=N_JOBS)
    if PIPELINE_MODE:
        return Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("rf",  base),
        ])
    return base

def _make_estimator_et():
    """Build an ExtraTrees estimator (Pipeline if PIPELINE_MODE)."""
    base = ExtraTreesRegressor(random_state=42, n_jobs=N_JOBS)
    if PIPELINE_MODE:
        return Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("et",  base),
        ])
    return base

def _prefix_rf(param_name: str) -> str:
    return f"rf__{param_name}" if PIPELINE_MODE else param_name

def _prefix_et(param_name: str) -> str:
    return f"et__{param_name}" if PIPELINE_MODE else param_name

# -------- [A] RF Search with bootstrap=True (optionally with row subsampling) --------
param_boot = {}
for k, v in common_space.items():
    # include criteria only if RF supports it; otherwise stay with squared_error
    if k == "criterion":
        continue
    param_boot[_prefix_rf(k)] = v
param_boot[_prefix_rf("criterion")] = CRITERIA_LIST  # both if any supported
param_boot[_prefix_rf("bootstrap")] = [True]
if HAS_MAX_SAMPLES_RF:
    param_boot[_prefix_rf("max_samples")] = [0.7, 0.9, 1.0] if FAST_MODE else [0.5, 0.7, 0.9, 1.0]

search_boot = RandomizedSearchCV(
    estimator=_make_estimator_rf(),
    param_distributions=param_boot,
    n_iter=N_ITER_A,
    cv=tscv,
    scoring=scoring,
    refit=REFIT_METRIC,              # choose best by RMSE or MAE
    n_jobs=N_JOBS,
    verbose=1,
    random_state=42,
)
print("\n[Search A] RandomForest (bootstrap=True) ...")
search_boot.fit(X_train, y_train)
print("  A: best params:", search_boot.best_params_)
print(f"  A: best CV {REFIT_METRIC.upper()}: {-search_boot.best_score_:.4f}")

# -------- [B] RF Search with bootstrap=False --------
search_noboot = None
if RUN_SEARCH_B:
    param_noboot = {}
    for k, v in common_space.items():
        if k == "criterion":
            continue
        param_noboot[_prefix_rf(k)] = v
    param_noboot[_prefix_rf("criterion")] = CRITERIA_LIST
    param_noboot[_prefix_rf("bootstrap")] = [False]

    search_noboot = RandomizedSearchCV(
        estimator=_make_estimator_rf(),
        param_distributions=param_noboot,
        n_iter=N_ITER_B,
        cv=tscv,
        scoring=scoring,
        refit=REFIT_METRIC,
        n_jobs=N_JOBS,
        verbose=1,
        random_state=43,  # different seed for diversity
    )
    print("\n[Search B] RandomForest (bootstrap=False) ...")
    try:
        search_noboot.fit(X_train, y_train)
        print("  B: best params:", search_noboot.best_params_)
        print(f"  B: best CV {REFIT_METRIC.upper()}: {-search_noboot.best_score_:.4f}")
    except KeyboardInterrupt:
        print("  B: interrupted by user; falling back to Search A.")
        search_noboot = None

# -------- [C] ExtraTrees Search (no bootstrap/max_samples used here) --------
search_et = None
if RUN_SEARCH_C:
    try:
        param_et = {}
        for k, v in common_space.items():
            # ExtraTrees shares most params: n_estimators/max_depth/min_samples*/max_features/criterion
            if k == "criterion":
                continue
            param_et[_prefix_et(k)] = v
        # criteria for ET: use both if ET supports absolute_error, else squared_error only
        et_criteria = ["squared_error", "absolute_error"] if HAS_ABS_CRITERION_ET else ["squared_error"]
        param_et[_prefix_et("criterion")] = et_criteria

        search_et = RandomizedSearchCV(
            estimator=_make_estimator_et(),
            param_distributions=param_et,
            n_iter=N_ITER_C,
            cv=tscv,
            scoring=scoring,
            refit=REFIT_METRIC,
            n_jobs=N_JOBS,
            verbose=1,
            random_state=44,
        )
        print("\n[Search C] ExtraTrees ...")
        search_et.fit(X_train, y_train)
        print("  C: best params:", search_et.best_params_)
        print(f"  C: best CV {REFIT_METRIC.upper()}: {-search_et.best_score_:.4f}")
    except KeyboardInterrupt:
        print("  C: interrupted by user; skipping ExtraTrees.")
        search_et = None
    except Exception as e:
        print(f"  C: skipped due to error: {e}")
        search_et = None

# -------- Pick the overall best by chosen CV metric --------
candidates = [s for s in (search_boot, search_noboot, search_et) if s is not None]
best_search = max(candidates, key=lambda s: s.best_score_)  # higher neg score = better (lower error)

print("\n[Selection] Choosing the overall better search by CV metric ...")
print("  Selected params:", best_search.best_params_)
print(f"  Selected CV {REFIT_METRIC.upper()}: {-best_search.best_score_:.4f}")

# Use the tuned best estimator downstream (Pipeline or bare model depending on PIPELINE_MODE)
best_estimator = best_search.best_estimator_

# -------------------------------------------------------------
# 6. Evaluate performance on validation and test sets
# -------------------------------------------------------------
def eval_and_print(split_name, y_true, y_pred):
    """Compute and print MSE, MAE, and R² for a given dataset split."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{split_name} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}", flush=True)

print("\n=== Evaluation (Selected Best Model) ===", flush=True)
if PIPELINE_MODE:
    val_pred  = best_estimator.predict(X_val_df)
    test_pred = best_estimator.predict(X_test_df)
else:
    val_pred  = best_estimator.predict(X_val)
    test_pred = best_estimator.predict(X_test)

eval_and_print("VAL ", y_val,  val_pred)
eval_and_print("TEST", y_test, test_pred)

# -------------------------------------------------------------
# Baseline comparisons: mean and median predictors
# -------------------------------------------------------------
def baseline_report(y_true, name="mean"):
    if name == "mean":
        yhat = np.full_like(y_true, fill_value=np.mean(y_true), dtype=float)
    elif name == "median":
        yhat = np.full_like(y_true, fill_value=np.median(y_true), dtype=float)
    else:
        raise ValueError("name must be 'mean' or 'median'")

    mse = mean_squared_error(y_true, yhat)
    mae = mean_absolute_error(y_true, yhat)
    rmse = np.sqrt(mse)
    print(f"Baseline ({name}) -> RMSE: {rmse:.4f}  MAE: {mae:.4f}")

print("\n=== Baselines ===")
baseline_report(y_val,  "mean")
baseline_report(y_val,  "median")
baseline_report(y_test, "mean")
baseline_report(y_test, "median")

# -------------------------------------------------------------
# Optional: Simple mean-ensemble across available best estimators (A/B/C)
# -------------------------------------------------------------
def _predict_auto(est, X_df, X_arr):
    """Helper: predict with or without Pipeline using the same call-site."""
    if PIPELINE_MODE:
        return est.predict(X_df)
    return est.predict(X_arr)

avail_ests = []
for s in (search_boot, search_noboot, search_et):
    if s is not None:
        avail_ests.append(s.best_estimator_)

if len(avail_ests) >= 2:
    print("\n[Ensemble] Evaluating simple mean-ensemble of available best estimators ...")
    if PIPELINE_MODE:
        val_preds  = np.column_stack([_predict_auto(e, X_val_df,  X_val)  for e in avail_ests]).mean(axis=1)
        test_preds = np.column_stack([_predict_auto(e, X_test_df, X_test) for e in avail_ests]).mean(axis=1)
    else:
        val_preds  = np.column_stack([_predict_auto(e, X_val_df,  X_val)  for e in avail_ests]).mean(axis=1)
        test_preds = np.column_stack([_predict_auto(e, X_test_df, X_test) for e in avail_ests]).mean(axis=1)

    def _eval(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        print(f"{name} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

    _eval(y_val,  val_preds,  "VAL  Ensemble")
    _eval(y_test, test_preds, "TEST Ensemble")

    ENSEMBLE_RESULTS = (val_preds, test_preds)
else:
    ENSEMBLE_RESULTS = None

# -------------------------------------------------------------
# Pretty comparison table (Model vs Baselines on RMSE/MAE)
# -------------------------------------------------------------
def _metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return rmse, mae

rows = []
rmse_m, mae_m = _metrics(y_val,  val_pred)
rows.append(("VAL",  "Model", rmse_m, mae_m))
rows.append(("VAL",  "Baseline-mean",
             np.sqrt(mean_squared_error(y_val,  np.full_like(y_val, y_val.mean(), dtype=float))),
             mean_absolute_error(y_val, np.full_like(y_val, y_val.mean(), dtype=float))))
rows.append(("VAL",  "Baseline-median",
             np.sqrt(mean_squared_error(y_val,  np.full_like(y_val, np.median(y_val), dtype=float))),
             mean_absolute_error(y_val, np.full_like(y_val, np.median(y_val), dtype=float))))

rmse_m, mae_m = _metrics(y_test, test_pred)
rows.append(("TEST", "Model", rmse_m, mae_m))
rows.append(("TEST", "Baseline-mean",
             np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_test.mean(), dtype=float))),
             mean_absolute_error(y_test, np.full_like(y_test, y_test.mean(), dtype=float))))
rows.append(("TEST", "Baseline-median",
             np.sqrt(mean_squared_error(y_test, np.full_like(y_test, np.median(y_test), dtype=float))),
             mean_absolute_error(y_test, np.full_like(y_test, np.median(y_test), dtype=float))))

# Append ensemble rows if available
if ENSEMBLE_RESULTS is not None:
    val_preds, test_preds = ENSEMBLE_RESULTS
    rmse_v, mae_v = _metrics(y_val,  val_preds)
    rmse_t, mae_t = _metrics(y_test, test_preds)
    rows.append(("VAL",  "Ensemble", rmse_v, mae_v))
    rows.append(("TEST", "Ensemble", rmse_t, mae_t))

comparison_df = pd.DataFrame(rows, columns=["Split", "Method", "RMSE", "MAE"])
print(comparison_df.to_string(index=False))

# Save tables
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
    """
    Return (underlying_tree_model, importances_array or None).
    Works for Pipeline(estimator) and bare RF/ET.
    """
    model = estimator
    if hasattr(estimator, "named_steps"):  # Pipeline case
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

    # ---- NEW: bar chart of top importances ----
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
# 8. NEW — Visualizations for predictions and residuals
# -------------------------------------------------------------
def plot_pred_vs_actual(y_true, y_pred, split, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    minv = float(np.nanmin([y_true.min(), y_pred.min()]))
    maxv = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([minv, maxv], [minv, maxv], linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{split} — Predicted vs. Actual")
    _savefig(save_path)
    plt.close()

def plot_time_series(t, y_true, y_pred, split, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_true, linewidth=1, label="Actual")
    plt.plot(t, y_pred, linewidth=1, alpha=0.9, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COL)
    plt.title(f"{split} — Time Series: Actual vs. Predicted")
    plt.legend()
    _savefig(save_path)
    plt.close()

def plot_residuals_hist(y_true, y_pred, split, save_path):
    resid = y_pred - y_true
    plt.figure(figsize=(7, 4))
    plt.hist(resid, bins=40, alpha=0.8)
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Count")
    plt.title(f"{split} — Residuals Histogram")
    _savefig(save_path)
    plt.close()

def plot_bar_metric_comparison(df_metrics, split, save_path):
    sub = df_metrics[df_metrics["Split"] == split].copy()
    # bar chart: RMSE / MAE side-by-side
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

# Produce plots for VAL / TEST
plot_pred_vs_actual(y_val,  val_pred,  "VAL",  os.path.join(SAVE_DIR, "val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test, test_pred, "TEST", os.path.join(SAVE_DIR, "test_pred_vs_actual.png"))

plot_time_series(t_val,  y_val,  val_pred,  "VAL",  os.path.join(SAVE_DIR, "val_timeseries.png"))
plot_time_series(t_test, y_test, test_pred, "TEST", os.path.join(SAVE_DIR, "test_timeseries.png"))

plot_residuals_hist(y_val,  val_pred,  "VAL",  os.path.join(SAVE_DIR, "val_residuals_hist.png"))
plot_residuals_hist(y_test, test_pred, "TEST", os.path.join(SAVE_DIR, "test_residuals_hist.png"))

plot_bar_metric_comparison(comparison_df, "VAL",  os.path.join(SAVE_DIR, "val_model_vs_baselines.png"))
plot_bar_metric_comparison(comparison_df, "TEST", os.path.join(SAVE_DIR, "test_model_vs_baselines.png"))

# -------------------------------------------------------------
# 9. NEW — Optional classification view: Precision / Recall / F1 + PR curve
# -------------------------------------------------------------
def _maybe_get_threshold(y_true):
    if CLASS_THRESH_QUANTILE is not None:
        return float(np.quantile(y_true, CLASS_THRESH_QUANTILE))
    return CLASS_THRESH

if CLASSIFICATION_VIEW:
    thr_val  = _maybe_get_threshold(y_val)
    thr_test = _maybe_get_threshold(y_test)

    y_val_bin  = (y_val  >= thr_val).astype(int)
    y_test_bin = (y_test >= thr_test).astype(int)

    # Use regression outputs as scores; binarize by same threshold for label
    val_pred_bin  = (val_pred  >= thr_val).astype(int)
    test_pred_bin = (test_pred >= thr_test).astype(int)

    def _cls_report(split, y_true_bin, y_pred_bin, scores, save_prefix):
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        print(f"[{split} Classification] Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

        # PR curve using continuous score (the regressed intensity)
        p, r, _ = precision_recall_curve(y_true_bin, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(r, p, linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{split} — Precision-Recall Curve")
        _savefig(os.path.join(SAVE_DIR, f"{save_prefix}_pr_curve.png"))
        plt.close()

    _cls_report("VAL",  y_val_bin,  val_pred_bin,  val_pred,  "val")
    _cls_report("TEST", y_test_bin, test_pred_bin, test_pred, "test")
else:
    print("\n[Info] Classification view disabled (set CLASSIFICATION_VIEW=True to compute Recall/F1 & PR curves).")
