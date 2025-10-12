# -------------------------------------------------------------
# RandomForest with LOG-target + winsor + reweight + 3-way search
# Raw-scale scoring + timings + visuals + advanced metrics
# + AR features + combo refit (overall + tail)
# Author: Susie + Group 10 (COMPSCI 760)
# -------------------------------------------------------------

import os, json, time
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    precision_score, recall_score, f1_score, make_scorer
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================
# 0) GLOBAL KNOBS
# =========================
PIPELINE_MODE = True

# --- 搜索/选优指标：在“原始尺度”优化（组合目标） ---
REFIT_METRIC   = "combo_mae"   # 兼顾整体与上尾
LAM_COMBO      = 0.7           # combo 权重: lam*MAE_all + (1-lam)*MAE_tail

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
CLASS_THRESH         = 0.3
CLASS_THRESH_QUANTILE = None

# ===== 可选图：新指标（Pearson / F1@q=0.90）的小图 =====
GENERATE_EXTRA_FIGS = True

# ----- Target engineering knobs -----
USE_WINSOR_Y      = True     # 保留极值 + 轻压尾
Y_WINSOR_UP_Q     = 0.995
USE_LOG1P_TARGET  = True     # y -> log1p(y)

# 上尾样本重加权
USE_REWEIGHT      = True
REWEIGHT_REF_Q    = 0.96
REWEIGHT_ALPHA    = 12.0
REWEIGHT_GAMMA    = 5.0

# ----- Smearing 纠偏（log→原尺度系统性低估） -----
USE_SMEARING      = True
SMEARING_MODE     = "bin"    # "global" or "bin"
SMEARING_BINS     = 12

# ----- DTW 记分风格（与 XGB 对齐）-----
# SCALE: "raw"（原始尺度）或 "z"（z-score 标准化后仅比较形状）
# MODE : "per_step"（按长度归一化，推荐）或 "sum"（总和）
DTW_SCALE = "raw"       # <- 若 XGB 用 z-score，请改为 "z"
DTW_MODE  = "per_step"  # <- 若 XGB 报的是总和，请改为 "sum"

# ---------- Timing helpers ----------
timings = {}
def tic(): return time.perf_counter()
def toc(t0): return time.perf_counter() - t0
def timed(key, fn, *args, **kwargs):
    t0 = tic()
    out = fn(*args, **kwargs)
    dt = toc(t0)
    timings[key] = float(dt)
    print(f"[Timing] {key}: {dt:.3f}s")
    return out

# =========================
# 1) Load data
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "final-planb-24.csv")

SAVE_DIR = os.path.join(BASE_DIR, SAVE_DIR_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

def _savefig(path, tight=True, dpi=150):
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[Saved] {path}")

t0 = tic()
df = pd.read_csv(CSV_PATH, parse_dates=["time"])
timings["load_data"] = toc(t0)
print("Loaded:", CSV_PATH)
print("Shape before drop:", df.shape)
print("Time range:", df["time"].min(), "->", df["time"].max(), flush=True)

# =========================
# 2) Target & AR features
# =========================
TARGET_COL = "keogram_mean"
before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
after = len(df)
print(f"Dropped rows with NaN target ({TARGET_COL}): {before - after}", flush=True)

df = df.sort_values("time")

# —— 自回归/短窗统计，全部 shift(1)（无泄漏） ——
df["y_lag1"]       = df[TARGET_COL].shift(1)
df["y_rollmean_6"] = df[TARGET_COL].rolling(6, min_periods=1).mean().shift(1)
df["y_rollmax_6"]  = df[TARGET_COL].rolling(6, min_periods=1).max().shift(1)

drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]
features = [c for c in df.columns if c not in drop_cols]
assert TARGET_COL not in features

X_all = df[features]
y_all_raw = df[TARGET_COL].values

# =========================
# 3) Time split
# =========================
t0 = tic()
train_idx = df[(df["time"] < "2018-01-01")].index
val_idx   = df[(df["time"] >= "2018-01-01") & (df["time"] < "2019-01-01")].index
test_idx  = df[(df["time"] >= "2019-01-01") & (df["time"] < "2021-01-01")].index
timings["split_time"] = toc(t0)

print("Split sizes:",
      "train =", len(train_idx),
      "val =",   len(val_idx),
      "test =",  len(test_idx), flush=True)

X_train_df = X_all.loc[train_idx]
X_val_df   = X_all.loc[val_idx]
X_test_df  = X_all.loc[test_idx]

y_train_raw = y_all_raw[df.index.get_indexer(train_idx)]
y_val_raw   = y_all_raw[df.index.get_indexer(val_idx)]
y_test_raw  = y_all_raw[df.index.get_indexer(test_idx)]

t_val  = df.loc[val_idx,  "time"].values
t_test = df.loc[test_idx, "time"].values

assert not np.isnan(y_train_raw).any()
assert not np.isnan(y_val_raw).any()
assert not np.isnan(y_test_raw).any()

if not PIPELINE_MODE:
    print("PIPELINE_MODE=False: imputing outside (may cause slight CV leakage).")
    t0 = tic()
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train_df)
    X_val   = imp.transform(X_val_df)
    X_test  = imp.transform(X_test_df)
    timings["impute_time"] = toc(t0)
else:
    print("PIPELINE_MODE=True: imputer inside CV via Pipeline.")
    X_train, X_val, X_test = X_train_df, X_val_df, X_test_df

# =========================
# 4) Target engineering + (optional) reweight
# =========================
def _fwd(y):  return np.log1p(y) if USE_LOG1P_TARGET else y
def _inv(y):  return np.expm1(y) if USE_LOG1P_TARGET else y

t0 = tic()
if USE_WINSOR_Y:
    cap_up = float(np.quantile(y_train_raw, Y_WINSOR_UP_Q))
    y_train_cap = np.minimum(y_train_raw, cap_up)
    y_val_cap   = np.minimum(y_val_raw,   cap_up)
    y_test_cap  = np.minimum(y_test_raw,  cap_up)
else:
    y_train_cap, y_val_cap, y_test_cap = y_train_raw, y_val_raw, y_test_raw

y_train_t = _fwd(y_train_cap)
y_val_t   = _fwd(y_val_cap)
y_test_t  = _fwd(y_test_cap)

if USE_REWEIGHT:
    ranks = pd.Series(y_train_raw).rank(method="average") / len(y_train_raw)
    boost_zone = np.clip((ranks - REWEIGHT_REF_Q) / (1 - REWEIGHT_REF_Q), 0, 1)
    sample_weight_train = 1.0 + REWEIGHT_ALPHA * (boost_zone ** REWEIGHT_GAMMA)
else:
    sample_weight_train = None
timings["target_engineering"] = toc(t0)

def _summ(name, arr):
    arr = np.asarray(arr, float)
    qs = np.quantile(arr, [0.5, 0.9, 0.95, 0.99])
    print(f"[Dist] {name}  mean={arr.mean():.4f}  p50={qs[0]:.4f}  p90={qs[1]:.4f}  p95={qs[2]:.4f}  p99={qs[3]:.4f}")
_summ("y_train_raw", y_train_raw)
if USE_WINSOR_Y: _summ("y_train_cap", y_train_cap)
if USE_LOG1P_TARGET: _summ("y_train_t(log1p)", y_train_t)

# =========================
# 5) Searches on transformed y (CV scoring 在“原始尺度”)
# =========================
def _supports_params_rf(**kwargs):
    try:
        RandomForestRegressor(random_state=0, n_jobs=1, **kwargs); return True
    except TypeError:
        return False

def _supports_params_et(**kwargs):
    try:
        ExtraTreesRegressor(random_state=0, n_jobs=1, **kwargs); return True
    except TypeError:
        return False

HAS_ABS_CRIT_RF = _supports_params_rf(criterion="absolute_error")
HAS_MAX_SAMPLES_RF = _supports_params_rf(max_samples=0.8, bootstrap=True)
HAS_ABS_CRIT_ET = _supports_params_et(criterion="absolute_error")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# ---- 自定义“原始尺度”打分器 ----
def _to_raw(arr):
    arr = np.asarray(arr, dtype=float)
    return _inv(arr) if USE_LOG1P_TARGET else arr

def _raw_rmse_scorer(y_true_t, y_pred_t):
    y_true_raw = _to_raw(y_true_t); y_pred_raw = _to_raw(y_pred_t)
    return -float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))

def _raw_mae_scorer(y_true_t, y_pred_t):
    y_true_raw = _to_raw(y_true_t); y_pred_raw = _to_raw(y_pred_t)
    return -float(mean_absolute_error(y_true_raw, y_pred_raw))

def _tail_mae_raw_scorer(y_true_t, y_pred_t, q=0.90):
    y_true_raw = _to_raw(y_true_t); y_pred_raw = _to_raw(y_pred_t)
    thr = np.quantile(y_true_raw, q)
    m = y_true_raw >= thr
    if not np.any(m): m = np.ones_like(y_true_raw, dtype=bool)
    return -float(mean_absolute_error(y_true_raw[m], y_pred_raw[m]))

def _combo_mae(y_true_t, y_pred_t, lam=LAM_COMBO):
    mae_all  = -_raw_mae_scorer(y_true_t, y_pred_t)
    mae_tail = -_tail_mae_raw_scorer(y_true_t, y_pred_t, 0.90)
    return -(lam * mae_all + (1 - lam) * mae_tail)

RAW_RMSE = make_scorer(_raw_rmse_scorer, greater_is_better=True)
RAW_MAE  = make_scorer(_raw_mae_scorer,  greater_is_better=True)
TAIL_MAE = make_scorer(lambda yt, yp: _tail_mae_raw_scorer(yt, yp, q=0.90), greater_is_better=True)
COMBO_MAE = make_scorer(lambda yt, yp: _combo_mae(yt, yp, lam=LAM_COMBO), greater_is_better=True)

scoring = {"raw_rmse": RAW_RMSE, "raw_mae": RAW_MAE, "tail_mae": TAIL_MAE, "combo_mae": COMBO_MAE}
assert REFIT_METRIC in scoring

common_space = {
    "n_estimators":      [300, 600, 900] if FAST_MODE else [400, 800, 1200, 1600],
    "max_depth":         [None, 12, 24]  if FAST_MODE else [None, 8, 12, 16, 24, 32],
    "min_samples_split": [2, 10, 20]     if FAST_MODE else [2, 5, 10, 20, 40],
    "min_samples_leaf":  [2, 4, 8]       if FAST_MODE else [1, 2, 4, 8, 12, 16],
    "max_features":      ["sqrt", 0.5]   if FAST_MODE else ["sqrt", 0.3, 0.5, 0.7, 1.0],
}
CRITERIA_LIST = ["squared_error", "absolute_error"] if (HAS_ABS_CRIT_RF or HAS_ABS_CRIT_ET) else ["squared_error"]

def _make_estimator_rf():
    base = RandomForestRegressor(random_state=42, n_jobs=N_JOBS)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", base)]) if PIPELINE_MODE else base

def _make_estimator_et():
    base = ExtraTreesRegressor(random_state=42, n_jobs=N_JOBS)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("et", base)]) if PIPELINE_MODE else base

def _prefix_rf(p): return f"rf__{p}" if PIPELINE_MODE else p
def _prefix_et(p): return f"et__{p}" if PIPELINE_MODE else p

fitkw_rf = {"rf__sample_weight": sample_weight_train} if (PIPELINE_MODE and USE_REWEIGHT) else (
           {"sample_weight": sample_weight_train} if USE_REWEIGHT else {})
fitkw_et = {"et__sample_weight": sample_weight_train} if (PIPELINE_MODE and USE_REWEIGHT) else (
           {"sample_weight": sample_weight_train} if USE_REWEIGHT else {})

# [A] RF bootstrap=True
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
timed("search_A_fit", search_boot.fit, X_train, y_train_t, **fitkw_rf)
print("  A: best params:", search_boot.best_params_)
print(f"  A: best CV {REFIT_METRIC.upper()}: {-search_boot.best_score_:.4f}")

# [B] RF bootstrap=False
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
    timed("search_B_fit", search_noboot.fit, X_train, y_train_t, **fitkw_rf)
    print("  B: best params:", search_noboot.best_params_)
    print(f"  B: best CV {REFIT_METRIC.upper()}: {-search_noboot.best_score_:.4f}")

# [C] ExtraTrees
search_et = None
if RUN_SEARCH_C:
    param_et = {_prefix_et(k): v for k, v in common_space.items()}
    param_et[_prefix_et("criterion")] = ["squared_error", "absolute_error"] if HAS_ABS_CRIT_ET else ["squared_error"]

    search_et = RandomizedSearchCV(
        estimator=_make_estimator_et(),
        param_distributions=param_et,
        n_iter=N_ITER_C, cv=tscv, scoring=scoring, refit=REFIT_METRIC,
        n_jobs=N_JOBS, verbose=1, random_state=44,
    )
    print("\n[Search C] ExtraTrees ...")
    timed("search_C_fit", search_et.fit, X_train, y_train_t, **fitkw_et)
    print("  C: best params:", search_et.best_params_)
    print(f"  C: best CV {REFIT_METRIC.upper()}: {-search_et.best_score_:.4f}")

# Select best
cands = [s for s in (search_boot, search_noboot, search_et) if s is not None]
best_search = max(cands, key=lambda s: s.best_score_)
print("\n[Selection] overall best by CV:", best_search.best_params_)
print(f"  Selected CV {REFIT_METRIC.upper()}: {-best_search.best_score_:.4f}")
best_estimator = best_search.best_estimator_

# =========================
# （可选）Smearing 纠偏
# =========================
smearing = {"mode": "off", "global": 1.0}
if USE_SMEARING:
    if PIPELINE_MODE:
        train_pred_t = best_estimator.predict(X_train_df)
    else:
        train_pred_t = best_estimator.predict(X_train)
    y_train_t_for_smear = _fwd(y_train_cap)
    eps = y_train_t_for_smear - train_pred_t
    smearing["mode"] = SMEARING_MODE
    if SMEARING_MODE == "global":
        smearing["global"] = float(np.mean(np.exp(eps)))
    else:
        q = np.quantile(train_pred_t, np.linspace(0, 1, SMEARING_BINS + 1))
        q = np.unique(q)
        bin_ids = np.digitize(train_pred_t, q[1:-1], right=False)
        factors = []
        fallback = float(np.mean(np.exp(eps)))
        for b in range(len(q)-1):
            mask = (bin_ids == b)
            factors.append(float(np.mean(np.exp(eps[mask]))) if mask.any() else fallback)
        smearing["bins"] = q.tolist()
        smearing["factors"] = factors

def _apply_smearing(pred_t, sm):
    if not USE_SMEARING or not USE_LOG1P_TARGET:
        return _inv(pred_t)
    y = np.expm1(pred_t)
    if sm.get("mode") == "global":
        return y * sm.get("global", 1.0)
    bins = np.array(sm.get("bins", []))
    facs = np.array(sm.get("factors", []))
    if bins.size == 0 or facs.size == 0:
        return y
    bin_ids = np.digitize(pred_t, bins[1:-1], right=False)
    idx = np.clip(bin_ids, 0, len(facs)-1)
    scale = facs[idx]
    return y * scale

# =========================
# ==== Metric helpers (Pearson / DTW / F1@q=0.90)
# =========================
def _pearson_corr(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2: return np.nan
    return float(np.corrcoef(a[m], b[m])[0,1])

# ---- 统一的 DTW（与 XGB 对齐）----
def _prepare_for_dtw(y_true_raw, y_pred_raw):
    a = np.asarray(y_true_raw, dtype=float)
    b = np.asarray(y_pred_raw, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if DTW_SCALE == "z":
        # z-normalization（形状相似度）
        a_std = np.std(a)
        b_std = np.std(b)
        if a_std > 0: a = (a - np.mean(a)) / a_std
        else: a = a - np.mean(a)
        if b_std > 0: b = (b - np.mean(b)) / b_std
        else: b = b - np.mean(b)
    return a, b

try:
    from fastdtw import fastdtw
    _use_fdtw = True
except Exception:
    _use_fdtw = False

def _dtw_distance(y_true_raw, y_pred_raw):
    a, b = _prepare_for_dtw(y_true_raw, y_pred_raw)
    if len(a) == 0 or len(b) == 0: return np.nan
    if _use_fdtw:
        dist, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))  # L1
    else:
        # 朴素 DTW
        n, m = len(a), len(b)
        D = np.full((n+1, m+1), np.inf, dtype=float)
        D[0,0] = 0.0
        for i in range(1, n+1):
            ai = a[i-1]
            for j in range(1, m+1):
                cost = abs(ai - b[j-1])
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        dist = float(D[n, m])
    if DTW_MODE == "per_step":
        dist = dist / len(a)
    return float(dist)

def _f1_extreme(y_true_raw, y_pred_raw, q=0.90):
    thr = float(np.quantile(y_true_raw, q))
    y_true_bin = (np.asarray(y_true_raw) >= thr).astype(int)
    y_pred_bin = (np.asarray(y_pred_raw) >= thr).astype(int)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    return prec, rec, f1, thr

# =========================
# 6) Evaluate on RAW scale (with timing)
# =========================
def _eval_raw(split, y_true_raw, y_pred_t, prefix):
    t0 = tic()
    if USE_SMEARING and USE_LOG1P_TARGET:
        y_pred_raw = _apply_smearing(y_pred_t, smearing)
    else:
        y_pred_raw = _inv(y_pred_t)
    timings[f"{prefix}_invert"] = toc(t0)

    t1 = tic()
    mse = mean_squared_error(y_true_raw, y_pred_raw)
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2  = r2_score(y_true_raw, y_pred_raw)
    pear = _pearson_corr(y_true_raw, y_pred_raw)
    dtw  = _dtw_distance(y_true_raw, y_pred_raw)  # <- 统一 DTW
    p90, r90, f190, thr = _f1_extreme(y_true_raw, y_pred_raw, q=0.90)
    timings[f"{prefix}_metrics"] = toc(t1)

    print(f"{split} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}  |  "
          f"Pearson: {pear:.4f}  DTW: {dtw:.4f}  F1@q=0.90: {f190:.4f} "
          f"(P={p90:.4f}, R={r90:.4f}, thr={thr:.4f})", flush=True)

    return y_pred_raw, {
        "RMSE": float(np.sqrt(mse)),
        "MAE":  float(mae),
        "R2":   float(r2),
        "Pearson": float(pear) if np.isfinite(pear) else np.nan,
        "DTW":     float(dtw) if np.isfinite(dtw) else np.nan,
        "F1_q90":  float(f190),
        "Prec_q90": float(p90),
        "Rec_q90":  float(r90),
        "Thr_q90":  float(thr),
    }

print("\n=== Evaluation (Selected Best Model on RAW scale) ===", flush=True)
if PIPELINE_MODE:
    val_pred_t  = timed("val_predict",  best_estimator.predict, X_val_df)
    test_pred_t = timed("test_predict", best_estimator.predict, X_test_df)
else:
    val_pred_t  = timed("val_predict",  best_estimator.predict, X_val)
    test_pred_t = timed("test_predict", best_estimator.predict, X_test)

val_pred_raw,  val_metrics_dict  = _eval_raw("VAL ",  y_val_raw,  val_pred_t,  "val")
test_pred_raw, test_metrics_dict = _eval_raw("TEST",  y_test_raw, test_pred_t, "test")

# =========================
# 7) Baselines & tables（RAW）
# =========================
def _metrics_basic(y_true_raw, y_pred_raw):
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    return rmse, mae

def baseline_mean(y_true):   return np.full_like(y_true, y_true.mean(), dtype=float)
def baseline_median(y_true): return np.full_like(y_true, np.median(y_true), dtype=float)
def baseline_persist(y_true):
    yb = np.empty_like(y_true, dtype=float); yb[0] = y_true[0]; yb[1:] = y_true[:-1]; return yb

yhat_val_mean   = baseline_mean(y_val_raw)
yhat_val_median = baseline_median(y_val_raw)
yhat_val_pers   = baseline_persist(y_val_raw)

yhat_tst_mean   = baseline_mean(y_test_raw)
yhat_tst_median = baseline_median(y_test_raw)
yhat_tst_pers   = baseline_persist(y_test_raw)

rows_basic = []
rows_basic.append(("VAL",  "Model(log-target)", *_metrics_basic(y_val_raw,  val_pred_raw)))
rows_basic.append(("VAL",  "Baseline-mean",     *_metrics_basic(y_val_raw,  yhat_val_mean)))
rows_basic.append(("VAL",  "Baseline-median",   *_metrics_basic(y_val_raw,  yhat_val_median)))
rows_basic.append(("VAL",  "Baseline-persist",  *_metrics_basic(y_val_raw,  yhat_val_pers)))

rows_basic.append(("TEST", "Model(log-target)", *_metrics_basic(y_test_raw, test_pred_raw)))
rows_basic.append(("TEST", "Baseline-mean",     *_metrics_basic(y_test_raw, yhat_tst_mean)))
rows_basic.append(("TEST", "Baseline-median",   *_metrics_basic(y_test_raw, yhat_tst_median)))
rows_basic.append(("TEST", "Baseline-persist",  *_metrics_basic(y_test_raw, yhat_tst_pers)))

comparison_df = pd.DataFrame(rows_basic, columns=["Split", "Method", "RMSE", "MAE"])

def _all_metrics_row(split, method, y_true, y_pred, override=None):
    if override is not None:
        d = {"Split":split, "Method":method}
        d.update(override)
        return d
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    pear = _pearson_corr(y_true, y_pred)
    dtw  = _dtw_distance(y_true, y_pred)  # <- 统一 DTW
    p90, r90, f190, thr = _f1_extreme(y_true, y_pred, q=0.90)
    return {
        "Split": split, "Method": method,
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "Pearson": float(pear) if np.isfinite(pear) else np.nan,
        "DTW": float(dtw) if np.isfinite(dtw) else np.nan,
        "F1_q90": float(f190), "Prec_q90": float(p90), "Rec_q90": float(r90),
        "Thr_q90": float(thr)
    }

rows_adv = []
rows_adv.append(_all_metrics_row("VAL",  "Model(log-target)", y_val_raw,  val_pred_raw,  val_metrics_dict))
rows_adv.append(_all_metrics_row("VAL",  "Baseline-mean",     y_val_raw,  yhat_val_mean))
rows_adv.append(_all_metrics_row("VAL",  "Baseline-median",   y_val_raw,  yhat_val_median))
rows_adv.append(_all_metrics_row("VAL",  "Baseline-persist",  y_val_raw,  yhat_val_pers))

rows_adv.append(_all_metrics_row("TEST", "Model(log-target)", y_test_raw, test_pred_raw, test_metrics_dict))
rows_adv.append(_all_metrics_row("TEST", "Baseline-mean",     y_test_raw, yhat_tst_mean))
rows_adv.append(_all_metrics_row("TEST", "Baseline-median",   y_test_raw, yhat_tst_median))
rows_adv.append(_all_metrics_row("TEST", "Baseline-persist",  y_test_raw, yhat_tst_pers))

advanced_df = pd.DataFrame(rows_adv, columns=[
    "Split","Method","RMSE","MAE","R2","Pearson","DTW","F1_q90","Prec_q90","Rec_q90","Thr_q90"
])

print("\n[Advanced Metrics]")
print(advanced_df.to_string(index=False))

# ---- 保存表格 ----
if SAVE_TABLES:
    comp_csv  = os.path.join(SAVE_DIR, "comparison_metrics.csv")
    comp_html = os.path.join(SAVE_DIR, "comparison_metrics.html")
    advanced_csv  = os.path.join(SAVE_DIR, "advanced_metrics.csv")
    advanced_html = os.path.join(SAVE_DIR, "advanced_metrics.html")

    advanced_df.to_csv(comp_csv, index=False)
    advanced_df.to_html(comp_html, index=False)
    advanced_df.to_csv(advanced_csv, index=False)
    advanced_df.to_html(advanced_html, index=False)
    print(f"[Saved] {comp_csv}")
    print(f"[Saved] {comp_html}")
    print(f"[Saved] {advanced_csv}")
    print(f"[Saved] {advanced_html}")

# =========================
# 8) Feature importances
# =========================
def _extract_model_and_importances(estimator):
    model = estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("rf", estimator.named_steps.get("et", model))
    if model is not None and hasattr(model, "feature_importances_"):
        return model, model.feature_importances_
    return model, None

mdl, importances = _extract_model_and_importances(best_estimator)
if importances is not None:
    order = np.argsort(importances)[::-1][:15]
    top_feats = [(features[i], float(importances[i])) for i in order]
    print("\nTop-15 feature importances:")
    for n, s in top_feats: print(f"{n:20s}  {s:.4f}")
    plt.figure(figsize=(10,6))
    names = [x[0] for x in top_feats][::-1]
    vals  = [x[1] for x in top_feats][::-1]
    ax = plt.gca(); ax.barh(names, vals)
    ax.set_xlabel("Importance"); ax.set_title("Top-15 Feature Importances")
    _savefig(os.path.join(SAVE_DIR, "feature_importance_top15.png")); plt.close()

# =========================
# 9) Visuals（RAW）
# =========================
def plot_pred_vs_actual(y_true_raw, y_pred_raw, split, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_raw, y_pred_raw, s=10, alpha=0.6)
    minv = float(np.nanmin([y_true_raw.min(), y_pred_raw.min()]))
    maxv = float(np.nanmax([y_true_raw.max(), y_pred_raw.max()]))
    plt.plot([minv, maxv], [minv, maxv], linestyle='--')
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"{split} — Predicted vs. Actual")
    _savefig(save_path); plt.close()

def plot_time_series(t, y_true_raw, y_pred_raw, split, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_true_raw, linewidth=1, label="Actual")
    plt.plot(t, y_pred_raw, linewidth=1, alpha=0.9, label="Predicted")
    plt.xlabel("Time"); plt.ylabel(TARGET_COL)
    plt.title(f"{split} — Time Series: Actual vs. Predicted")
    plt.legend(); _savefig(save_path); plt.close()

def plot_residuals_hist(y_true_raw, y_pred_raw, split, save_path):
    resid = y_pred_raw - y_true_raw
    plt.figure(figsize=(7, 4))
    plt.hist(resid, bins=40, alpha=0.8)
    plt.xlabel("Residual (Pred - Actual)"); plt.ylabel("Count")
    plt.title(f"{split} — Residuals Histogram")
    _savefig(save_path); plt.close()

def plot_bar_metric_comparison(df_metrics, split, save_path):
    sub = df_metrics[df_metrics["Split"] == split].copy()
    labels = sub["Method"].tolist(); x = np.arange(len(labels)); width = 0.38
    fig = plt.figure(figsize=(10, 5)); ax = plt.gca()
    ax.bar(x - width/2, sub["RMSE"].values, width, label="RMSE")
    ax.bar(x + width/2, sub["MAE"].values,  width, label="MAE")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.set_title(f"{split} — Model vs Baselines"); ax.legend()
    _savefig(save_path); plt.close()

# ---- 原有图 ----
plot_pred_vs_actual(y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_pred_vs_actual.png"))
plot_time_series(t_val,  y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_timeseries.png"))
plot_time_series(t_test, y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_timeseries.png"))
plot_residuals_hist(y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_residuals_hist.png"))
plot_residuals_hist(y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_residuals_hist.png"))
plot_bar_metric_comparison(advanced_df, "VAL",  os.path.join(SAVE_DIR, "val_model_vs_baselines.png"))
plot_bar_metric_comparison(advanced_df, "TEST", os.path.join(SAVE_DIR, "test_model_vs_baselines.png"))

# ---- 可选：小图 ----
if GENERATE_EXTRA_FIGS:
    def _bar_simple(df, split, metric_name, save_path):
        sub = df[df["Split"] == split]
        plt.figure(figsize=(8,4))
        plt.bar(sub["Method"], sub[metric_name])
        plt.xticks(rotation=25, ha='right')
        plt.title(f"{split} — {metric_name}")
        _savefig(save_path); plt.close()
    _bar_simple(advanced_df, "VAL",  "Pearson", os.path.join(SAVE_DIR, "val_metric_pearson.png"))
    _bar_simple(advanced_df, "TEST", "Pearson", os.path.join(SAVE_DIR, "test_metric_pearson.png"))
    _bar_simple(advanced_df, "VAL",  "F1_q90",  os.path.join(SAVE_DIR, "val_metric_f1q90.png"))
    _bar_simple(advanced_df, "TEST", "F1_q90",  os.path.join(SAVE_DIR, "test_metric_f1q90.png"))

# =========================
# 10) Optional classification view
# =========================
def _maybe_get_threshold(y_true):
    if CLASS_THRESH_QUANTILE is not None:
        return float(np.quantile(y_true, CLASS_THRESH_QUANTILE))
    return CLASS_THRESH

if CLASSIFICATION_VIEW:
    thr_val  = _maybe_get_threshold(y_val_raw)
    thr_test = _maybe_get_threshold(y_test_raw)

    y_val_bin  = (y_val_raw  >= thr_val).astype(int)
    y_test_bin = (y_test_raw >= thr_test).astype(int)
    val_pred_bin  = (val_pred_raw  >= thr_val).astype(int)
    test_pred_bin = (test_pred_raw >= thr_test).astype(int)

    from sklearn.metrics import precision_recall_curve
    def _cls_report(split, y_true_bin, y_pred_bin, scores, save_prefix):
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        print(f"[{split} Classification] Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        p, r, _ = precision_recall_curve(y_true_bin, scores)
        plt.figure(figsize=(6, 5)); plt.plot(r, p, linewidth=2)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{split} — Precision-Recall Curve")
        _savefig(os.path.join(SAVE_DIR, f"{save_prefix}_pr_curve.png")); plt.close()

    _cls_report("VAL",  y_val_bin,  val_pred_bin,  val_pred_raw,  "val")
    _cls_report("TEST", y_test_bin, test_pred_bin, test_pred_raw, "test")
else:
    print("\n[Info] Classification view disabled.")

# =========================
# 11) Save timing & data stats
# =========================
stats = {
    "n_features": len(features),
    "n_rows_total": int(len(df)),
    "n_rows_train": int(len(train_idx)),
    "n_rows_val":   int(len(val_idx)),
    "n_rows_test":  int(len(test_idx)),
}
timings.update(stats)

timings["search_total_fit"] = float(
    timings.get("search_A_fit", 0.0) +
    timings.get("search_B_fit", 0.0) +
    timings.get("search_C_fit", 0.0)
)
timings["eval_total"] = float(
    timings.get("val_predict", 0.0) + timings.get("val_metrics", 0.0) +
    timings.get("test_predict", 0.0) + timings.get("test_metrics", 0.0)
)

tim_csv = os.path.join(SAVE_DIR, "timings.csv")
tim_json = os.path.join(SAVE_DIR, "timings.json")
pd.DataFrame([timings]).to_csv(tim_csv, index=False)
with open(tim_json, "w") as f:
    json.dump(timings, f, indent=2)

print(f"[Saved] {tim_csv}")
print(f"[Saved] {tim_json}")
print("\nDone.")
