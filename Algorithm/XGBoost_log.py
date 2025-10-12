# =============================================================
# XGBoost (GPU-Optimized) — LOG-target + Winsor + Reweight + Combo Search
# Raw-scale scoring + timings + visuals + advanced metrics
# Dataset: Aurora Intensity (final-planb-24)
# Environment: Kaggle (GPU T4 ×2)
# Author: Group 10 (COMPSCI 760)
# =============================================================

import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, make_scorer
)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# =============================================================
# 0) CONFIGURATION
# =============================================================
PIPELINE_MODE = True
REFIT_METRIC = "combo_mae"
LAM_COMBO = 0.7

FAST_MODE = True
CPU_COUNT = os.cpu_count() or 2
N_JOBS = min(4, CPU_COUNT)
N_SPLITS = 2 if FAST_MODE else 3
N_ITER = 25 if FAST_MODE else 60
SAVE_DIR = "/kaggle/working/figs_xgb_combo"
os.makedirs(SAVE_DIR, exist_ok=True)

# Target transforms and weighting
USE_WINSOR_Y = True
Y_WINSOR_UP_Q = 0.995
USE_LOG1P_TARGET = True
USE_REWEIGHT = True
REWEIGHT_REF_Q = 0.96
REWEIGHT_ALPHA = 12.0
REWEIGHT_GAMMA = 5.0

# Smearing correction
USE_SMEARING = True
SMEARING_MODE = "bin"
SMEARING_BINS = 12

# DTW comparison settings
DTW_MODE = "per_step"

# Timing helper
timings = {}
def tic(): return time.perf_counter()
def toc(t0): return time.perf_counter() - t0
def timed(key, fn, *a, **kw):
    t0 = tic(); out = fn(*a, **kw); timings[key] = toc(t0)
    print(f"[Timing] {key}: {timings[key]:.2f}s")
    return out

# =============================================================
# 1) LOAD DATA (KAGGLE PATH)
# =============================================================
CSV_PATH = "/kaggle/input/final-planb-24-final/final-planb-24 (1).csv"
print(f"Loading dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, parse_dates=["time"])
print(f"Loaded shape = {df.shape}")
print(f"Time range = {df['time'].min()} → {df['time'].max()}")

TARGET_COL = "keogram_mean"
df = df.dropna(subset=[TARGET_COL]).sort_values("time")

# Autoregressive features (no leakage)
df["y_lag1"] = df[TARGET_COL].shift(1)
df["y_rollmean_6"] = df[TARGET_COL].rolling(6, min_periods=1).mean().shift(1)
df["y_rollmax_6"] = df[TARGET_COL].rolling(6, min_periods=1).max().shift(1)

drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]
features = [c for c in df.columns if c not in drop_cols]
X_all = df[features]
y_all_raw = df[TARGET_COL].values

# Time-based split
train_idx = df[df["time"] < "2018-01-01"].index
val_idx   = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")].index
test_idx  = df[(df["time"] >= "2019-01-01") & (df["time"] < "2021-01-01")].index

print("Split sizes:",
      f"Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

X_train_df, X_val_df, X_test_df = (
    X_all.loc[train_idx], X_all.loc[val_idx], X_all.loc[test_idx]
)
y_train_raw = df.loc[train_idx, TARGET_COL].values
y_val_raw   = df.loc[val_idx, TARGET_COL].values
y_test_raw  = df.loc[test_idx, TARGET_COL].values
t_val  = df.loc[val_idx, "time"].to_numpy()
t_test = df.loc[test_idx, "time"].to_numpy()

# =============================================================
# 2) TARGET PROCESSING
# =============================================================
def _fwd(y): return np.log1p(y) if USE_LOG1P_TARGET else y
def _inv(y): return np.expm1(y) if USE_LOG1P_TARGET else y

if USE_WINSOR_Y:
    cap = np.quantile(y_train_raw, Y_WINSOR_UP_Q)
    y_train_cap = np.minimum(y_train_raw, cap)
    y_val_cap   = np.minimum(y_val_raw, cap)
    y_test_cap  = np.minimum(y_test_raw, cap)
else:
    y_train_cap, y_val_cap, y_test_cap = y_train_raw, y_val_raw, y_test_raw

y_train_t = _fwd(y_train_cap)
y_val_t   = _fwd(y_val_cap)
y_test_t  = _fwd(y_test_cap)

# Tail reweighting
if USE_REWEIGHT:
    ranks = pd.Series(y_train_raw).rank(pct=True)
    boost = np.clip((ranks - REWEIGHT_REF_Q) / (1 - REWEIGHT_REF_Q), 0, 1)
    sample_weight_train = 1 + REWEIGHT_ALPHA * (boost ** REWEIGHT_GAMMA)
else:
    sample_weight_train = None

# =============================================================
# 3) SCORING FUNCTIONS (raw-scale)
# =============================================================
def _to_raw(y): return _inv(y) if USE_LOG1P_TARGET else y
def _raw_mae(y_true, y_pred): return -mean_absolute_error(_to_raw(y_true), _to_raw(y_pred))
def _tail_mae(y_true, y_pred, q=0.9):
    ytr, ypr = _to_raw(y_true), _to_raw(y_pred)
    thr = np.quantile(ytr, q)
    mask = ytr >= thr
    return -mean_absolute_error(ytr[mask], ypr[mask])
def _combo_mae(y_true, y_pred, lam=LAM_COMBO):
    return -(lam * -_raw_mae(y_true, y_pred) + (1 - lam) * -_tail_mae(y_true, y_pred))

RAW_MAE = make_scorer(_raw_mae, greater_is_better=True)
TAIL_MAE = make_scorer(lambda yt, yp: _tail_mae(yt, yp), greater_is_better=True)
COMBO_MAE = make_scorer(lambda yt, yp: _combo_mae(yt, yp), greater_is_better=True)
scoring = {"raw_mae": RAW_MAE, "tail_mae": TAIL_MAE, "combo_mae": COMBO_MAE}

# =============================================================
# 4) MODEL + GPU-ENABLED CV SEARCH
# =============================================================
def _make_estimator_xgb():
    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",      # use hist, device=cuda replaces gpu_hist
        predictor="gpu_predictor",
        device="cuda",
        random_state=42,
        n_jobs=N_JOBS
    )
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("xgb", base)]) if PIPELINE_MODE else base

_prefix = lambda p: f"xgb__{p}" if PIPELINE_MODE else p
param_grid = {
    _prefix("n_estimators"): [300, 600, 900] if FAST_MODE else [400, 800, 1200],
    _prefix("max_depth"): [4, 6, 8],
    _prefix("learning_rate"): [0.03, 0.05, 0.1],
    _prefix("subsample"): [0.8, 1.0],
    _prefix("colsample_bytree"): [0.8, 1.0],
    _prefix("gamma"): [0, 0.5, 1],
    _prefix("reg_lambda"): [1, 5, 10],
    _prefix("reg_alpha"): [0, 1, 5]
}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
fitkw = {"xgb__sample_weight": sample_weight_train} if (PIPELINE_MODE and USE_REWEIGHT) else (
    {"sample_weight": sample_weight_train} if USE_REWEIGHT else {}
)

print("\n[Search] GPU XGBoost combo search (T4 ×2) ...")
search = RandomizedSearchCV(
    estimator=_make_estimator_xgb(),
    param_distributions=param_grid,
    n_iter=N_ITER,
    cv=tscv,
    scoring=scoring,
    refit=REFIT_METRIC,
    verbose=1,
    n_jobs=N_JOBS,
    random_state=42
)
timed("search_fit", search.fit, X_train_df, y_train_t, **fitkw)
best_est = search.best_estimator_
print("Best params:", search.best_params_)
print(f"Best CV combo_mae: {-search.best_score_:.4f}")

# =============================================================
# 5) SMEARING CORRECTION
# =============================================================
smearing = {"mode": "off"}
if USE_SMEARING and USE_LOG1P_TARGET:
    train_pred_t = best_est.predict(X_train_df)
    eps = y_train_t - train_pred_t
    if SMEARING_MODE == "global":
        smearing = {"mode": "global", "global": float(np.mean(np.exp(eps)))}
    else:
        q = np.quantile(train_pred_t, np.linspace(0, 1, SMEARING_BINS + 1))
        bins = np.unique(q)
        ids = np.digitize(train_pred_t, bins[1:-1])
        facs = []
        for b in range(len(bins)-1):
            mask = ids == b
            facs.append(float(np.mean(np.exp(eps[mask]))) if mask.any() else 1.0)
        smearing = {"mode": "bin", "bins": bins.tolist(), "factors": facs}

def _apply_smear(pred_t, sm):
    y = np.expm1(pred_t)
    if sm.get("mode") == "global":
        return y * sm["global"]
    if sm.get("mode") == "bin":
        bins = np.array(sm["bins"]); facs = np.array(sm["factors"])
        ids = np.digitize(pred_t, bins[1:-1])
        ids = np.clip(ids, 0, len(facs)-1)
        return y * facs[ids]
    return y

# =============================================================
# 6) EVALUATION + METRICS
# =============================================================
try:
    from fastdtw import fastdtw
    _use_fdtw = True
except:
    _use_fdtw = False

def _pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 1 else np.nan

def _dtw(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if _use_fdtw:
        d, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
    else:
        n, m = len(a), len(b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i - 1] - b[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        d = D[n, m]
    return d / len(a) if DTW_MODE == "per_step" else d

def _eval(split, y_true_raw, y_pred_t):
    y_pred_raw = _apply_smear(y_pred_t, smearing) if (USE_SMEARING and USE_LOG1P_TARGET) else _inv(y_pred_t)
    mse = mean_squared_error(y_true_raw, y_pred_raw)
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)
    pear = _pearson(y_true_raw, y_pred_raw)
    dtw = _dtw(y_true_raw, y_pred_raw)
    thr = np.quantile(y_true_raw, 0.9)
    yb, yp = (y_true_raw >= thr).astype(int), (y_pred_raw >= thr).astype(int)
    f1 = f1_score(yb, yp, zero_division=0)
    print(f"{split}: RMSE={np.sqrt(mse):.3f}  MAE={mae:.3f}  R2={r2:.3f}  Pearson={pear:.3f}  DTW={dtw:.3f}  F1@q90={f1:.3f}")
    return dict(RMSE=np.sqrt(mse), MAE=mae, R2=r2, Pearson=pear, DTW=dtw, F1_q90=f1)

val_pred_t  = best_est.predict(X_val_df)
test_pred_t = best_est.predict(X_test_df)
val_metrics  = _eval("VAL ", y_val_raw, val_pred_t)
test_metrics = _eval("TEST", y_test_raw, test_pred_t)

# =============================================================
# 7) VISUALS + SAVE RESULTS
# =============================================================
def _savefig(path): plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); print("[Saved]", path)

val_pred_raw  = _apply_smear(val_pred_t, smearing)
test_pred_raw = _apply_smear(test_pred_t, smearing)

# --- VAL ---
if len(t_val) != len(y_val_raw):
    print("[WARN] VAL time axis mismatch — using indices.")
    t_val = np.arange(len(y_val_raw))
plt.figure(figsize=(10,4))
plt.plot(t_val, y_val_raw, label="Actual")
plt.plot(t_val, val_pred_raw, label="Predicted")
plt.legend(); plt.title("VAL — Time Series")
_savefig(os.path.join(SAVE_DIR, "val_timeseries.png"))

# --- TEST ---
if len(t_test) != len(y_test_raw):
    print("[WARN] TEST time axis mismatch — using indices.")
    t_test = np.arange(len(y_test_raw))
plt.figure(figsize=(10,4))
plt.plot(t_test, y_test_raw, label="Actual")
plt.plot(t_test, test_pred_raw, label="Predicted")
plt.legend(); plt.title("TEST — Time Series")
_savefig(os.path.join(SAVE_DIR, "test_timeseries.png"))

# --- Scatter plots ---
plt.figure(figsize=(6,6))
plt.scatter(y_val_raw, val_pred_raw, s=10, alpha=0.6)
plt.plot([y_val_raw.min(), y_val_raw.max()], [y_val_raw.min(), y_val_raw.max()], "--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("VAL — Pred vs Actual")
_savefig(os.path.join(SAVE_DIR, "val_pred_vs_actual.png"))

plt.figure(figsize=(6,6))
plt.scatter(y_test_raw, test_pred_raw, s=10, alpha=0.6)
plt.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], "--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("TEST — Pred vs Actual")
_savefig(os.path.join(SAVE_DIR, "test_pred_vs_actual.png"))

# --- Save metrics ---
pd.DataFrame([val_metrics, test_metrics], index=["VAL","TEST"]).to_csv(
    os.path.join(SAVE_DIR, "metrics_summary.csv")
)
print("\n✅ Done. GPU XGBoost run completed successfully.")
