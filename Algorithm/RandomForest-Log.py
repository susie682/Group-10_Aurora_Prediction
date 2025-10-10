# RandomForest-Log.py
# -------------------------------------------------------------
# Fast ExtraTrees + A2 target engineering + Advanced Metrics
# + Runtime & Data Size Reporting
# Author: Susie + Group 10 (COMPSCI 760) — streamlined for speed
# -------------------------------------------------------------
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)

# =========================
# 0) GLOBAL KNOBS (SPEED)
# =========================
PIPELINE_MODE = True
REFIT_METRIC  = "mae"

FAST_MODE   = True
CPU_COUNT   = os.cpu_count() or 2
N_JOBS      = min(4, CPU_COUNT)     # 如遇到 macOS resource_tracker 警告，可暂改为 1
N_SPLITS    = 2
N_ITER_C    = 6
RUN_SEARCH_C = True

SAVE_DIR_NAME = "figs_rf"
SAVE_TABLES   = True

# ----- A2 knobs -----
USE_WINSOR_Y     = True
Y_WINSOR_UP_Q    = 0.98       # 压缩顶部 2%（可按需要改为 0.90 压 10%）
USE_LOG1P_TARGET = False
USE_REWEIGHT     = True
REWEIGHT_REF_Q   = 0.90
REWEIGHT_ALPHA   = 6.0
REWEIGHT_GAMMA   = 3.0

# =========================
# Runtime helpers
# =========================
TIMINGS = []
def _tic():
    return time.perf_counter()

def _tock(t0, name):
    dt = time.perf_counter() - t0
    TIMINGS.append((name, dt))
    print(f"[Time] {name}: {dt:.2f} s")
    return dt

def _save_runtime_report(save_dir, extra_info):
    df = pd.DataFrame(TIMINGS, columns=["Stage", "Seconds"])
    for k, v in extra_info.items():
        df.loc[df.shape[0]] = [k, v]
    path = os.path.join(save_dir, "runtime_report.csv")
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")

def _savefig(path, tight=True, dpi=150):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[Saved] {path}")

# =========================
# 1) IO & SETUP
# =========================
t0_all = _tic()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")
# 你的当前数据路径（按你的发帖保持不变）
CSV_PATH = os.path.join(DATA_DIR, "final_planb_notWeighted_10_filtered.csv")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

SAVE_DIR = os.path.join(BASE_DIR, SAVE_DIR_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loaded:", CSV_PATH)
t0 = _tic()
df = pd.read_csv(CSV_PATH, parse_dates=["time"])
_tock(t0, "Load CSV")

print("Shape before drop:", df.shape)
print("Time range:", df["time"].min(), "->", df["time"].max(), flush=True)

# =========================
# 2) TARGET / FEATURES
# =========================
t0 = _tic()
TARGET_COL = "keogram_mean"
before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
after = len(df)
print(f"Dropped rows with NaN target ({TARGET_COL}): {before - after}", flush=True)

drop_cols = ["time", "keogram_mean", "keogram_median", "keogram_max"]
features = [c for c in df.columns if c not in drop_cols]
assert TARGET_COL not in features

X_all = df[features]
y_all_raw = df[TARGET_COL].values
n_rows, n_features = X_all.shape
print(f"[Data] rows={n_rows}, features={n_features}")
_tock(t0, "Drop NaN & select features")

# =========================
# 3) TIME SPLIT（固定年段）
# =========================
t0 = _tic()
df = df.sort_values("time")
# 你给定的 2024 年内切分
train_idx = df[df["time"] <  "2024-01-01"].index
val_idx   = df[(df["time"] >= "2024-01-01") & (df["time"] < "2024-07-01")].index
test_idx  = df[(df["time"] >= "2024-07-01") & (df["time"] < "2025-01-01")].index

print("Split sizes:", "train =", len(train_idx), "val =", len(val_idx), "test =", len(test_idx), flush=True)

X_train_df = X_all.loc[train_idx]
X_val_df   = X_all.loc[val_idx]
X_test_df  = X_all.loc[test_idx]

y_train_raw = y_all_raw[df.index.get_indexer(train_idx)]
y_val_raw   = y_all_raw[df.index.get_indexer(val_idx)]
y_test_raw  = y_all_raw[df.index.get_indexer(test_idx)]

t_val  = df.loc[val_idx,  "time"].values
t_test = df.loc[test_idx, "time"].values

# 分别打印每个 split 的 (rows, features)
print(f"[Data|Train] rows={X_train_df.shape[0]}, features={X_train_df.shape[1]}")
print(f"[Data|Val]   rows={X_val_df.shape[0]}, features={X_val_df.shape[1]}")
print(f"[Data|Test]  rows={X_test_df.shape[0]}, features={X_test_df.shape[1]}")

assert not np.isnan(y_train_raw).any()
assert not np.isnan(y_val_raw).any()
assert not np.isnan(y_test_raw).any()

if not PIPELINE_MODE:
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df)
    X_val   = imputer.transform(X_val_df)
    X_test  = imputer.transform(X_test_df)
else:
    print("PIPELINE_MODE=True: imputer will be fitted inside CV folds via Pipeline.")
    X_train, X_val, X_test = X_train_df, X_val_df, X_test_df
_tock(t0, "Time split & (optional) impute")

# =========================
# 4) TARGET ENGINEERING
# =========================
t0 = _tic()
y_cap = y_train_raw.copy()
if USE_WINSOR_Y:
    cap_up = float(np.quantile(y_train_raw, Y_WINSOR_UP_Q))
    y_cap = np.minimum(y_cap, cap_up)
    y_val_cap  = np.minimum(y_val_raw,  cap_up)
    y_test_cap = np.minimum(y_test_raw, cap_up)
else:
    y_val_cap, y_test_cap = y_val_raw.copy(), y_test_raw.copy()

def _fwd(y):  return np.log1p(y) if USE_LOG1P_TARGET else y
def _inv(y):  return np.expm1(y) if USE_LOG1P_TARGET else y

y_train = _fwd(y_cap)
y_val_t = _fwd(y_val_cap)
y_test_t = _fwd(y_test_cap)

if USE_REWEIGHT:
    ranks = pd.Series(y_train_raw).rank(method="average") / len(y_train_raw)
    boost_zone = np.clip((ranks - REWEIGHT_REF_Q) / (1 - REWEIGHT_REF_Q), 0, 1)
    sample_weight_train = 1.0 + REWEIGHT_ALPHA * (boost_zone ** REWEIGHT_GAMMA)
else:
    sample_weight_train = None
_tock(t0, "Target engineering (winsor/log/reweight)")

# =========================
# 5) MODEL: ExtraTrees + RandomizedSearchCV
# =========================
def _supports_params_et(**kwargs) -> bool:
    try:
        ExtraTreesRegressor(random_state=0, n_jobs=1, **kwargs)
        return True
    except TypeError:
        return False

HAS_ABS_CRITERION_ET = _supports_params_et(criterion="absolute_error")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
assert REFIT_METRIC in scoring

common_space = {
    "n_estimators":      [200, 400],
    "max_depth":         [10, 12],
    "min_samples_split": [10, 20],
    "min_samples_leaf":  [4, 8],
    "max_features":      ["sqrt"],
}

def _make_estimator_et():
    base = ExtraTreesRegressor(random_state=42, n_jobs=N_JOBS)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("et", base)]) if PIPELINE_MODE else base

def _prefix_et(p): return f"et__{p}" if PIPELINE_MODE else p

fitkw_et = {"et__sample_weight": sample_weight_train} if (PIPELINE_MODE and USE_REWEIGHT) else (
           {"sample_weight": sample_weight_train} if USE_REWEIGHT else {})

param_et = {_prefix_et(k): v for k, v in common_space.items()}
param_et[_prefix_et("criterion")] = ["squared_error", "absolute_error"] if HAS_ABS_CRITERION_ET else ["squared_error"]

t0 = _tic()
search_et = RandomizedSearchCV(
    estimator=_make_estimator_et(),
    param_distributions=param_et,
    n_iter=N_ITER_C, cv=tscv, scoring=scoring, refit=REFIT_METRIC,
    n_jobs=N_JOBS, verbose=1, random_state=44,
)
print("\n[Search C] ExtraTrees ...")
search_et.fit(X_train, y_train, **fitkw_et)
_tock(t0, "Model search+fit (ExtraTrees)")

print("  C: best params:", search_et.best_params_)
print(f"  C: best CV {REFIT_METRIC.upper()}: {-search_et.best_score_:.4f}")

best_estimator = search_et.best_estimator_
print("\n[Selection] Using ExtraTrees best estimator.")

# =========================
# 6) EVALUATION (to RAW)
# =========================
def eval_and_print(split_name, y_true_raw, y_pred_transformed):
    y_pred_raw = _inv(y_pred_transformed)
    mse = mean_squared_error(y_true_raw, y_pred_raw)
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2  = r2_score(y_true_raw, y_pred_raw)
    print(f"{split_name} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}", flush=True)
    return y_pred_raw

t0 = _tic()
print("\n=== Evaluation (Selected Best Model) ===", flush=True)
val_pred_t  = best_estimator.predict(X_val_df)
test_pred_t = best_estimator.predict(X_test_df)
val_pred_raw  = eval_and_print("VAL ",  y_val_raw,  val_pred_t)
test_pred_raw = eval_and_print("TEST",  y_test_raw, test_pred_t)
_tock(t0, "Evaluation (predict+metrics)")

# =========================
# 7) BASELINES + 表格
# =========================
def baseline_mean(y_true_raw):   return np.full_like(y_true_raw, y_true_raw.mean(), dtype=float)
def baseline_median(y_true_raw): return np.full_like(y_true_raw, np.median(y_true_raw), dtype=float)
def baseline_persist(y_true_raw):
    yb = np.empty_like(y_true_raw, dtype=float)
    yb[0] = y_true_raw[0]
    yb[1:] = y_true_raw[:-1]
    return yb

def _metrics(y_true_raw, y_pred_raw):
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    return rmse, mae

yhat_val_mean, yhat_val_median, yhat_val_pers = baseline_mean(y_val_raw), baseline_median(y_val_raw), baseline_persist(y_val_raw)
yhat_tst_mean, yhat_tst_median, yhat_tst_pers = baseline_mean(y_test_raw), baseline_median(y_test_raw), baseline_persist(y_test_raw)

rows = []
rows.append(("VAL",  "Model",             *_metrics(y_val_raw,  val_pred_raw)))
rows.append(("VAL",  "Baseline-mean",     *_metrics(y_val_raw,  yhat_val_mean)))
rows.append(("VAL",  "Baseline-median",   *_metrics(y_val_raw,  yhat_val_median)))
rows.append(("VAL",  "Baseline-persist",  *_metrics(y_val_raw,  yhat_val_pers)))

rows.append(("TEST", "Model",             *_metrics(y_test_raw, test_pred_raw)))
rows.append(("TEST", "Baseline-mean",     *_metrics(y_test_raw, yhat_tst_mean)))
rows.append(("TEST", "Baseline-median",   *_metrics(y_test_raw, yhat_tst_median)))
rows.append(("TEST", "Baseline-persist",  *_metrics(y_test_raw, yhat_tst_pers)))

comparison_df = pd.DataFrame(rows, columns=["Split", "Method", "RMSE", "MAE"])
print(comparison_df.to_string(index=False))
if SAVE_TABLES:
    comp_csv  = os.path.join(SAVE_DIR, "comparison_metrics.csv")
    comp_html = os.path.join(SAVE_DIR, "comparison_metrics.html")
    comparison_df.to_csv(comp_csv, index=False)
    comparison_df.to_html(comp_html, index=False)
    print(f"[Saved] {comp_csv}")
    print(f"[Saved] {comp_html}")

# =========================
# 8) FEATURE IMPORTANCE
# =========================
def _extract_model_and_importances(estimator):
    model = estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("et", model)
    if model is not None and hasattr(model, "feature_importances_"):
        return model, model.feature_importances_
    return model, None

t0 = _tic()
model_, importances = _extract_model_and_importances(best_estimator)
if importances is not None:
    order = np.argsort(importances)[::-1][:15]
    top_feats = [(features[i], float(importances[i])) for i in order]
    print("\nTop-15 feature importances:")
    for n, s in top_feats:
        print(f"{n:20s}  {s:.4f}")
    plt.figure(figsize=(10,6))
    names = [x[0] for x in top_feats][::-1]
    vals  = [x[1] for x in top_feats][::-1]
    ax = plt.gca()
    ax.barh(names, vals)
    ax.set_xlabel("Importance")
    ax.set_title("Top-15 Feature Importances (ExtraTrees)")
    _savefig(os.path.join(SAVE_DIR, "feature_importance_top15.png"))
    plt.close()
_tock(t0, "Feature importance & plot")

# =========================
# 9) PLOTS (RAW scale)
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
    labels = sub["Method"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig = plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    ax.bar(x - width/2, sub["RMSE"].values, width, label="RMSE")
    ax.bar(x + width/2, sub["MAE"].values,  width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')  # 旋转防重叠
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.set_title(f"{split} — Model vs Baselines")
    ax.legend()
    _savefig(save_path); plt.close()

t0 = _tic()
plot_pred_vs_actual(y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_pred_vs_actual.png"))
plot_time_series(t_val,  y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_timeseries.png"))
plot_time_series(t_test, y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_timeseries.png"))
plot_residuals_hist(y_val_raw,  val_pred_raw,  "VAL",  os.path.join(SAVE_DIR, "val_residuals_hist.png"))
plot_residuals_hist(y_test_raw, test_pred_raw, "TEST", os.path.join(SAVE_DIR, "test_residuals_hist.png"))
plot_bar_metric_comparison(comparison_df, "VAL",  os.path.join(SAVE_DIR, "val_model_vs_baselines.png"))
plot_bar_metric_comparison(comparison_df, "TEST", os.path.join(SAVE_DIR, "test_model_vs_baselines.png"))
_tock(t0, "Plotting (all figures)")

# =========================
# 10) ADVANCED METRICS: Pearson, DTW, F1(extreme) + plots
# =========================
def _pearson_corr(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2: return np.nan
    return float(np.corrcoef(a[m], b[m])[0,1])

try:
    from fastdtw import fastdtw
    def _dtw_distance(y_true, y_pred):
        a = np.asarray(y_true, dtype=float); b = np.asarray(y_pred, dtype=float)
        m = np.isfinite(a) & np.isfinite(b); a=a[m]; b=b[m]
        if len(a)==0 or len(b)==0: return np.nan
        dist, _ = fastdtw(a, b); return float(dist)
except Exception:
    def _dtw_distance(y_true, y_pred, every=5):
        a = np.asarray(y_true, dtype=float)[::every]
        b = np.asarray(y_pred, dtype=float)[::every]
        m = np.isfinite(a) & np.isfinite(b); a=a[m]; b=b[m]
        if len(a)==0 or len(b)==0: return np.nan
        n, mlen = len(a), len(b)
        D = np.full((n+1, mlen+1), np.inf); D[0,0]=0.0
        for i in range(1, n+1):
            ai = a[i-1]
            for j in range(1, mlen+1):
                cost = abs(ai - b[j-1])
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(D[n, mlen])

def _f1_extreme(y_true_raw, y_pred_raw, q=0.90):
    thr = float(np.quantile(y_true_raw, q))
    y_true_bin = (np.asarray(y_true_raw) >= thr).astype(int)
    y_pred_bin = (np.asarray(y_pred_raw) >= thr).astype(int)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    return prec, rec, f1, thr

def _advanced_metrics_block(split_name, y_true_raw, y_pred_raw, baselines_dict, save_prefix):
    pear = {"Model": _pearson_corr(y_true_raw, y_pred_raw)}
    dtw  = {"Model": _dtw_distance(y_true_raw, y_pred_raw)}
    prf  = {"Model": _f1_extreme(y_true_raw, y_pred_raw, q=0.90)}
    for k, yhat in baselines_dict.items():
        pear[k] = _pearson_corr(y_true_raw, yhat)
        dtw[k]  = _dtw_distance(y_true_raw, yhat)
        prf[k]  = _f1_extreme(y_true_raw, yhat, q=0.90)

    print(f"\n[{split_name}] Pearson (higher is better):")
    for k,v in pear.items(): print(f"{k:>16s}: {v:.4f}")
    print(f"[{split_name}] DTW distance (lower is better):")
    for k,v in dtw.items(): print(f"{k:>16s}: {v:.2f}")
    print(f"[{split_name}] F1@q=0.90 (higher is better):")
    for k,(p,r,f1,thr) in prf.items():
        print(f"{k:>16s}: F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

    methods = list(pear.keys())
    df_adv = pd.DataFrame({
        "Method": methods,
        "Pearson": [pear[m] for m in methods],
        "DTW (lower is better)": [dtw[m] for m in methods],
        "F1@q=0.90": [prf[m][2] for m in methods],
        "Precision@q=0.90": [prf[m][0] for m in methods],
        "Recall@q=0.90": [prf[m][1] for m in methods],
    })
    adv_csv  = os.path.join(SAVE_DIR, f"{save_prefix}_advanced_metrics.csv")
    adv_html = os.path.join(SAVE_DIR, f"{save_prefix}_advanced_metrics.html")
    df_adv.to_csv(adv_csv, index=False)
    df_adv.to_html(adv_html, index=False)
    print(f"[Saved] {adv_csv}")
    print(f"[Saved] {adv_html}")

    # 小柱状图（修复横坐标重叠）
    plt.figure(figsize=(6.2,3.4)); plt.bar(df_adv["Method"], df_adv["Pearson"]); plt.xticks(rotation=30, ha='right')
    plt.title(f"{split_name} — Pearson (higher better)"); plt.ylabel("Pearson r")
    _savefig(os.path.join(SAVE_DIR, f"{save_prefix}_pearson_bar.png")); plt.close()

    plt.figure(figsize=(6.2,3.4)); plt.bar(df_adv["Method"], df_adv["DTW (lower is better)"]); plt.xticks(rotation=30, ha='right')
    plt.title(f"{split_name} — DTW (lower better)"); plt.ylabel("DTW distance")
    _savefig(os.path.join(SAVE_DIR, f"{save_prefix}_dtw_bar.png")); plt.close()

    plt.figure(figsize=(6.2,3.4)); plt.bar(df_adv["Method"], df_adv["F1@q=0.90"]); plt.xticks(rotation=30, ha='right')
    plt.title(f"{split_name} — F1 (higher better)"); plt.ylabel("F1 score")
    _savefig(os.path.join(SAVE_DIR, f"{save_prefix}_f1_bar.png")); plt.close()

t0 = _tic()
baselines_val = {"Baseline-mean": yhat_val_mean, "Baseline-median": yhat_val_median, "Baseline-persist": yhat_val_pers}
baselines_tst = {"Baseline-mean": yhat_tst_mean, "Baseline-median": yhat_tst_median, "Baseline-persist": yhat_tst_pers}
_advanced_metrics_block("VAL",  y_val_raw,  val_pred_raw,  baselines_val,  save_prefix="val")
_advanced_metrics_block("TEST", y_test_raw, test_pred_raw, baselines_tst, save_prefix="test")
_tock(t0, "Advanced metrics (Pearson/DTW/F1)")

# =========================
# 11) OPTIONAL PR CURVE (off)
# =========================
print("\n[Info] Classification view disabled (set CLASSIFICATION_VIEW=True to compute PR curves).")

# =========================
# 12) Save runtime report & total time
# =========================
total_sec = _tock(t0_all, "TOTAL (end-to-end)")
extra = {
    "Rows(after_drop)": float(n_rows),
    "Features": float(n_features),
    "Train_rows": float(X_train_df.shape[0]),
    "Val_rows": float(X_val_df.shape[0]),
    "Test_rows": float(X_test_df.shape[0]),
    "n_jobs": float(N_JOBS),
    "CV_splits": float(N_SPLITS),
    "Search_iter_ET": float(N_ITER_C),
}
_save_runtime_report(SAVE_DIR, extra)
