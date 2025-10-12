#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================================
# Aurora dataset preprocessing:
# 1) Log-transform ONLY targets
# 2) Winsorize feature extremes (cap top 3% at 97th percentile)
# 3) Upweight extreme target samples
# ============================================================

import os
import numpy as np
import pandas as pdi

# ---------- CONFIG ------------------------------------------------------------
CSV_PATH         = r"D:/760/final-planb-24.csv"  
SAVE_DIR         = os.path.dirname(CSV_PATH)
OUT_CSV          = os.path.join(SAVE_DIR, "aurorafinal-planb-24.csv")


TARGET_COLS      = ["keogram_mean", "keogram_median", "keogram_max"]
PRIMARY_TARGET   = "keogram_mean"        

# Winsorization：只截断上尾（示例为 top 3% -> cap 到 97% 分位）
WINSORIZE_UPPER_Q = 0.97                 # 上截断分位
APPLY_WINSORIZE_TO = "features"          # "features" | "all_numeric_except_targets"

# Sample weights：把 PRIMARY_TARGET ≥ 97% 分位的样本加更高权重
EXTREME_Q        = 0.97
BASE_WEIGHT      = 1.0
EXTREME_WEIGHT   = 5.0                   # 你可以改成 3、10 等

# -----------------------------------------------------------------------------


def safe_log1p(s: pd.Series) -> pd.Series:
    """log1p for target; clip negatives to 0 just in case."""
    s = s.copy()
    s = s.fillna(np.nan)
    # 若可能有负数，先把小于0的值抬到0（强约束：亮度/强度本应非负）
    s = s.clip(lower=0)
    return np.log1p(s)


def winsorize_upper(df: pd.DataFrame, cols, upper_q=0.97, suffix="_wz") -> pd.DataFrame:
    """
    Cap values above upper quantile to that quantile (upper-only winsorization).
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            uq = df[c].quantile(upper_q)
            df[c + suffix] = df[c].where(df[c] <= uq, uq)
    return df


# ===================== 1) LOAD ===============================================
df = pd.read_csv(CSV_PATH)

# 检查目标列是否存在
missing = [c for c in TARGET_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing target columns in CSV: {missing}")

# 只对主目标列做非空筛选（日志里提到 Dropped rows with NaN target）
before_n = len(df)
df = df.loc[~df[PRIMARY_TARGET].isna()].reset_index(drop=True)
dropped = before_n - len(df)
print(f"Dropped rows with NaN {PRIMARY_TARGET}: {dropped}")

# ===================== 2) LOG-TRANSFORM (targets only) =======================
for t in TARGET_COLS:
    df[t + "_log1p"] = safe_log1p(df[t])

# ===================== 3) WINSORIZATION (features) ===========================
# 数值列
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

# 从 Winsorization 排除：目标原列与它们的 log1p 列（通常不动目标；截图思路是“保留极端事件”）
exclude_for_wz = set(TARGET_COLS + [t + "_log1p" for t in TARGET_COLS])

if APPLY_WINSORIZE_TO == "features":
    # 只对“特征”做截断：即数值列里除去目标与目标的 log1p 列
    wz_cols = [c for c in numeric_cols if c not in exclude_for_wz]
else:
    # 或者对“除目标之外的所有数值列”做截断（包含 log1p 以外的数值）
    wz_cols = [c for c in numeric_cols if c not in TARGET_COLS]

df = winsorize_upper(df, cols=wz_cols, upper_q=WINSORIZE_UPPER_Q, suffix="_wz")

# 说明：
# - 新生成的列名为  原列名 + "_wz"（只对超过 97% 分位的值进行上截断）
# - 原列保留不变（方便你对比和做特征选择）

# ===================== 4) SAMPLE WEIGHTS for extreme targets ==================
# 以 PRIMARY_TARGET 的 97% 分位为阈值，定义极端样本；这些行加更高权重
thr_extreme = df[PRIMARY_TARGET].quantile(EXTREME_Q)
is_extreme  = df[PRIMARY_TARGET] >= thr_extreme
df["sample_weight"] = np.where(is_extreme, EXTREME_WEIGHT, BASE_WEIGHT)

print(f"{is_extreme.sum()} rows marked as EXTREME (>= {EXTREME_Q:.0%} quantile, value >= {thr_extreme:.4g}).")
print(f"Non-extreme rows: {(~is_extreme).sum()}")

# ===================== 5) OPTIONAL: quick summary ============================
summary = {
    "rows_after_drop": len(df),
    "extreme_threshold_value": float(thr_extreme),
    "extreme_rows": int(is_extreme.sum()),
    "non_extreme_rows": int((~is_extreme).sum()),
    "winsorize_upper_q": WINSORIZE_UPPER_Q,
    "extreme_quantile_for_weights": EXTREME_Q,
    "extreme_weight": EXTREME_WEIGHT,
    "base_weight": BASE_WEIGHT,
}
print("Summary:", summary)

# ===================== 6) SAVE ==============================================
df.to_csv(OUT_CSV, index=False)
print(f"Saved preprocessed file with weights => {OUT_CSV}")



# In[7]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- CatBoost check ----------------------------------------------------------
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "catboost>=1.2"])
    from catboost import CatBoostRegressor, Pool

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score

# ----------------------------- CONFIG ----------------------------------------
CSV_PATH       = r"D:/760/aurorafinal-planb-24.csv"
SAVE_DIR       = os.path.dirname(CSV_PATH) or "."
MODEL_PATH     = os.path.join(SAVE_DIR, "catboost_timeaware_leaksafe.cbm")
VAL_PRED_PATH  = os.path.join(SAVE_DIR, "catboost_val_timeaware.csv")
TEST_PRED_PATH = os.path.join(SAVE_DIR, "catboost_test_timeaware.csv")
FI_CSV_PATH    = os.path.join(SAVE_DIR, "catboost_fi_timeaware.csv")

TARGET_COL     = "keogram_mean"
TARGET_FAMILY  = ["keogram_mean", "keogram_median", "keogram_max"]

# 只对目标做 log1p（作用在 y 上）
USE_LOG1P_TARGET = True
TARGET_WINSOR_UPPER_Q = 0.97

# 极端事件阈值（用于F1/Precision/Recall，基于训练集原始尺度的该分位数）
EXTREME_Q = 0.97

# DTW 的带宽比例
DTW_WINDOW_RATIO = 0.1

# 自回归短窗参数
ROLL_WINDOW = 6   # 6 点窗口（例如 6 小时/6 个样本）
SEED = 760

CAT_PARAMS = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5.0,
    random_strength=1.5,
    n_estimators=2000,
    od_type="Iter",
    od_wait=100,
    random_seed=SEED,
    verbose=100
    # 有 GPU 可加: task_type="GPU"
)

# ----------------------------- UTILS -----------------------------------------
def pearson_corr(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):  # 常数序列时相关系数未定义
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def dtw_distance(s, t, window_ratio=None):
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return np.nan

    if window_ratio is None:
        w = max(n, m)  # 无窗口
    else:
        w = int(max(n, m) * float(window_ratio))
        w = max(w, abs(n - m))  # 至少覆盖长度差

    INF = 1e18
    prev = np.full(m + 1, INF)
    curr = np.full(m + 1, INF)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[0] = INF
        j_start = max(1, i - w)
        j_end   = min(m, i + w)
        if j_start > 1:
            curr[1:j_start] = INF
        if j_end < m:
            curr[j_end+1:] = INF
        for j in range(j_start, j_end + 1):
            cost = (s[i - 1] - t[j - 1]) ** 2
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[m]))

def to_event_labels(y, threshold):
    y = np.asarray(y, dtype=float)
    return (y >= threshold).astype(int)

# ----------------------------- LOAD ------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}  rows={len(df)}, cols={df.shape[1]}")

# ========== 时间解析 ==========
if "time" not in df.columns:
    raise ValueError("缺少 'time' 列。")

dt = pd.to_datetime(df["time"], errors="coerce", utc=False)
if dt.isna().all():
    ser = pd.to_numeric(df["time"], errors="coerce")
    if ser.isna().all():
        raise ValueError("无法解析 'time'。")
    dt = pd.to_datetime(ser, unit="D", origin="1899-12-30", errors="coerce")

df["__time__"] = dt
df["__year__"] = df["__time__"].dt.year.astype(int)

# ========== 自回归 + 短窗特征（严格滞后，避免泄漏） ==========
# 在按时间排序的序列上计算，再对齐回原索引
df_sorted = df.sort_values("__time__").copy()
y_sorted  = df_sorted[TARGET_COL].astype(float)

lag1      = y_sorted.shift(1)
rollmean6 = y_sorted.rolling(ROLL_WINDOW, min_periods=1).mean().shift(1)
rollmax6  = y_sorted.rolling(ROLL_WINDOW, min_periods=1).max().shift(1)

df["y_lag1"]        = np.nan
df["y_rollmean_6"]  = np.nan
df["y_rollmax_6"]   = np.nan
df.loc[df_sorted.index, "y_lag1"]       = lag1.values
df.loc[df_sorted.index, "y_rollmean_6"] = rollmean6.values
df.loc[df_sorted.index, "y_rollmax_6"]  = rollmax6.values

print("Added AR features: y_lag1, y_rollmean_6, y_rollmax_6 (all shifted by 1).")

# ========== 防泄漏：构造特征 ==========
if TARGET_COL not in df.columns:
    raise ValueError(f"缺少目标列：{TARGET_COL}")

has_weight = "sample_weight" in df.columns
print("Using sample_weight:", has_weight)

# 1) 从特征中排除：时间列 + 目标家族（原列/ *_log1p / *_wz）+ 所有 *_log1p、*_wz 列
exclude = {"time", "__time__", "__year__"}
for t in TARGET_FAMILY:
    for suf in ["", "_log1p", "_wz"]:
        col = f"{t}{suf}"
        if col in df.columns:
            exclude.add(col)

for c in list(df.columns):
    if str(c).endswith("_log1p") or str(c).endswith("_wz"):
        exclude.add(c)

if has_weight:
    exclude.add("sample_weight")  # 仅作权重，不进特征

feature_candidates = [c for c in df.columns if c not in exclude]

def make_X_and_cat_idx(frame: pd.DataFrame, cols):
    X = frame[cols].copy()
    X.columns = [str(col) for col in X.columns]
    # 去重列名
    if len(set(X.columns)) != X.shape[1]:
        counts, new_cols = {}, []
        for col in X.columns:
            if col not in counts:
                counts[col] = 0; new_cols.append(col)
            else:
                counts[col] += 1; new_cols.append(f"{col}_dup{counts[col]}")
        X.columns = new_cols
    # object -> string
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        X[c] = X[c].astype("string").fillna(pd.NA)
    cat_cols = X.select_dtypes(include=["string","object","category"]).columns.tolist()
    cat_idx  = [X.columns.get_loc(c) for c in cat_cols]
    return X, cat_cols, cat_idx

X_all, cat_cols, cat_idx = make_X_and_cat_idx(df, feature_candidates)
print(f"Num features: {X_all.shape[1]} (categorical: {len(cat_cols)})")

# ========== 目标只在 y 上做 winsorize + log1p ==========
y_all_raw = df[TARGET_COL].astype(float).values
upper_q = np.nanquantile(y_all_raw, TARGET_WINSOR_UPPER_Q)
y_all_raw_wz = np.minimum(y_all_raw, upper_q)

def _fwd(y):   # forward transform for training label
    return np.log1p(y) if USE_LOG1P_TARGET else y

def _inv(y):   # inverse transform for evaluation
    return np.expm1(y) if USE_LOG1P_TARGET else y

y_all = _fwd(y_all_raw_wz)
w_all = df["sample_weight"].values if has_weight else np.ones(len(df), dtype=float)

# ----------------------- 时间感知切分 -----------------------
# Train: 2012-01-01 ~ 2017-12-31
# Val  : 2018-01-01 ~ 2018-12-31
# Test : 2019-01-01 ~ 2020-12-31
train_idx = df.loc[(df["__time__"] >= pd.Timestamp("2012-01-01")) &
                   (df["__time__"] <  pd.Timestamp("2018-01-01"))].index
val_idx   = df.loc[(df["__time__"] >= pd.Timestamp("2018-01-01")) &
                   (df["__time__"] <  pd.Timestamp("2019-01-01"))].index
test_idx  = df.loc[(df["__time__"] >= pd.Timestamp("2019-01-01")) &
                   (df["__time__"] <  pd.Timestamp("2021-01-01"))].index
assert len(train_idx)>0 and len(val_idx)>0 and len(test_idx)>0, "时间段为空，请检查数据年份覆盖。"

X_tr, y_tr, w_tr = X_all.loc[train_idx], y_all[train_idx], w_all[train_idx]
X_va, y_va, w_va = X_all.loc[val_idx],   y_all[val_idx],   w_all[val_idx]
X_te, y_te, w_te = X_all.loc[test_idx],  y_all[test_idx],  w_all[test_idx]

train_pool = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_idx)
val_pool   = Pool(X_va, y_va, weight=w_va, cat_features=cat_idx)
test_pool  = Pool(X_te, y_te, weight=w_te, cat_features=cat_idx)

# ----------------------------- 训练 ------------------------------------------
model = CatBoostRegressor(**CAT_PARAMS)
model.fit(train_pool, eval_set=val_pool)

# ----------------------------- 评估（log & raw） -----------------------------
def metrics_report(y_true_logspace, y_pred_logspace, times, tag):
    rmse_log = mean_squared_error(y_true_logspace, y_pred_logspace, squared=False)
    y_true   = _inv(y_true_logspace)
    y_pred   = _inv(y_pred_logspace)
    rmse     = mean_squared_error(y_true, y_pred, squared=False)
    mae      = mean_absolute_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred)
    pearson  = pearson_corr(y_true, y_pred)

    order = np.argsort(times)
    y_true_ord = y_true[order]
    y_pred_ord = y_pred[order]
    dtw_dist = dtw_distance(y_true_ord, y_pred_ord, window_ratio=DTW_WINDOW_RATIO)

    print(f"\n[{tag}] Metrics")
    print(f"RMSE (log1p): {rmse_log:.6f}")
    print(f"RMSE (raw)  : {rmse:.6f}")
    print(f"MAE  (raw)  : {mae:.6f}")
    print(f"R2   (raw)  : {r2:.6f}")
    print(f"Pearson r   : {pearson:.6f}")
    print(f"DTW distance: {dtw_dist:.6f}")

    return y_true, y_pred, pearson, dtw_dist, order

val_pred_log  = model.predict(val_pool)
test_pred_log = model.predict(test_pool)

val_times  = df.loc[val_idx, "__time__"].values
test_times = df.loc[test_idx, "__time__"].values

y_va_raw, va_pred_raw, va_r, va_dtw, val_order = metrics_report(y_va,  val_pred_log,  val_times,  "VAL (2018)")
y_te_raw, te_pred_raw, te_r, te_dtw, test_order= metrics_report(y_te, test_pred_log, test_times, "TEST (2019–2020)")

# ----------------------------- Baselines（全局均值/中位数） -------------------
y_train_raw_wz = _inv(y_tr) 
global_mean    = float(np.mean(y_train_raw_wz))
global_median  = float(np.median(y_train_raw_wz))

def eval_const_baseline(y_true_raw, const_value, name, tag):
    yhat = np.full_like(y_true_raw, fill_value=const_value, dtype=float)
    rmse = mean_squared_error(y_true_raw, yhat, squared=False)
    mae  = mean_absolute_error(y_true_raw, yhat)
    r2   = r2_score(y_true_raw, yhat)
    r    = pearson_corr(y_true_raw, yhat)
    dtw  = dtw_distance(y_true_raw, yhat, window_ratio=DTW_WINDOW_RATIO)
    print(f"[{tag}] Baseline ({name}) -> RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.6f} | Pearson: {r:.6f} | DTW: {dtw:.6f}")

print("\n--- Baselines (raw, constant lines) ---")
print("VAL (2018):")
eval_const_baseline(y_va_raw, global_mean,   "GlobalMean",   "VAL")
eval_const_baseline(y_va_raw, global_median, "GlobalMedian", "VAL")
print("TEST (2019-2020):")
eval_const_baseline(y_te_raw, global_mean,   "GlobalMean",   "TEST")
eval_const_baseline(y_te_raw, global_median, "GlobalMedian", "TEST")

# ----------------------------- 极端事件 F1/Precision/Recall -------------------
from sklearn.metrics import precision_recall_curve

extreme_thr = float(np.quantile(y_train_raw_wz, EXTREME_Q))
print(f"\nExtreme-event threshold (train {EXTREME_Q:.0%} quantile, raw): {extreme_thr:.6f}")

# 1) 主阈值：在 VAL 上，按预测值扫描阈值，选择使 F1 最大的 t_pred
y_true_evt_val = (y_va_raw >= extreme_thr).astype(int)
prec_v, rec_v, thr_v = precision_recall_curve(y_true_evt_val, va_pred_raw)
f1s_v   = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)
best_i_v = int(np.nanargmax(f1s_v))
t_pred = float(thr_v[best_i_v]) if best_i_v < len(thr_v) else float(np.median(va_pred_raw))
print(f"Chosen prediction threshold on VAL (max-F1): t_pred = {t_pred:.6f}")

# 2) 率匹配阈值：用训练集极端比例 p，令验证集预测的正例比例也为 p，得 t_rate
pos_rate_train = float((y_train_raw_wz >= extreme_thr).mean())  # p = 1 - EXTREME_Q（但更稳：用实际比例）
t_rate = float(np.quantile(va_pred_raw, 1.0 - pos_rate_train))
print(f"Rate-matched threshold from VAL (match train positive rate={pos_rate_train:.4f}): t_rate = {t_rate:.6f}")

def event_metrics_at_threshold(y_true_raw, y_pred_raw, tag, t):
    y_true_evt = (y_true_raw >= extreme_thr).astype(int)
    y_pred_evt = (y_pred_raw >= t).astype(int)
    n_true = int(y_true_evt.sum()); n_pred = int(y_pred_evt.sum())
    tp = int(((y_true_evt==1) & (y_pred_evt==1)).sum())
    if n_true == 0:
        print(f"[{tag}] Positives -> true:{n_true} | pred@t:{n_pred}")
        print(f"[{tag}] No positive events; P/R/F1 = N/A")
        return np.nan, np.nan, np.nan
    p = precision_score(y_true_evt, y_pred_evt, zero_division=0)
    r = recall_score(y_true_evt, y_pred_evt, zero_division=0)
    f = f1_score(y_true_evt, y_pred_evt, zero_division=0)
    print(f"[{tag}] Positives -> true:{n_true} | pred@t:{n_pred} | TP:{tp}")
    print(f"[{tag}] Extreme events -> Precision: {p:.6f} | Recall: {r:.6f} | F1: {f:.6f}")
    return p, r, f

# 3) 报告两套阈值的结果（不影响其它任何输出）
event_metrics_at_threshold(y_va_raw, va_pred_raw, "VAL (2018, t_pred)", t_pred)
event_metrics_at_threshold(y_te_raw, te_pred_raw, "TEST (2019–2020, t_pred)", t_pred)

event_metrics_at_threshold(y_va_raw, va_pred_raw, "VAL (2018, rate-matched)", t_rate)
event_metrics_at_threshold(y_te_raw, te_pred_raw, "TEST (2019–2020, rate-matched)", t_rate)

# ----------------------------- 保存预测明细 -----------------------------------
val_out = X_va.copy()
val_out["time"]        = val_times
val_out["y_true_raw"]  = y_va_raw
val_out["y_pred_raw"]  = va_pred_raw
val_out["residual"]    = y_va_raw - va_pred_raw
val_out["baseline_mean"]   = global_mean
val_out["baseline_median"] = global_median
val_out.to_csv(VAL_PRED_PATH, index=False)
print(f"\nSaved: {VAL_PRED_PATH}")

test_out = X_te.copy()
test_out["time"]        = test_times
test_out["y_true_raw"]  = y_te_raw
test_out["y_pred_raw"]  = te_pred_raw
test_out["residual"]    = y_te_raw - te_pred_raw
test_out["baseline_mean"]   = global_mean
test_out["baseline_median"] = global_median
test_out.to_csv(TEST_PRED_PATH, index=False)
print(f"Saved: {TEST_PRED_PATH}")

# ----------------------------- 特征重要度 ------------------------------------
importances = model.get_feature_importance(Pool(X_all, y_all, cat_features=cat_idx))
fi = pd.DataFrame({"feature": list(X_all.columns), "importance": importances}).sort_values("importance", ascending=False)
fi.to_csv(FI_CSV_PATH, index=False)
print(f"Feature importance saved to: {FI_CSV_PATH}")

# ----------------------------- 绘图 ------------------------------------------
order = test_order
plt.figure(figsize=(12, 4.7))
plt.plot(test_out["time"].values[order], test_out["y_true_raw"].values[order], label="True")
plt.plot(test_out["time"].values[order], test_out["y_pred_raw"].values[order], label="CatBoost")
plt.plot(test_out["time"].values[order], np.full(len(order), global_mean),   label="Baseline (GlobalMean)")
plt.plot(test_out["time"].values[order], np.full(len(order), global_median), label="Baseline (GlobalMedian)")
plt.title("Test (2019–2020) — Time Series with Constant Baselines")
plt.xlabel("Time"); plt.ylabel(TARGET_COL); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_test_timeseries_const_baselines.png"), dpi=160)

plt.figure(figsize=(12, 4.7))
plt.plot(test_out["time"].values[order], test_out["residual"].values[order])
plt.title("Test (2019–2020) — Residuals over time (True - Pred)")
plt.xlabel("Time"); plt.ylabel("Residual"); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_test_residuals.png"), dpi=160)

plt.figure(figsize=(6.5, 4.7))
plt.hist(test_out["residual"].values, bins=50)
plt.title("Test (2019–2020) — Residuals histogram")
plt.xlabel("Residual"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_test_residual_hist.png"), dpi=160)

print("\nSaved plots: plot_test_timeseries_const_baselines.png / plot_test_residuals.png / plot_test_residual_hist.png")
model.save_model(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# ========================= VAL (2018) 可视化 =========================
# 1) 时间序列：模型 vs 全局均值/中位数 baseline
order = val_order
plt.figure(figsize=(12, 4.7))
plt.plot(val_out["time"].values[order], val_out["y_true_raw"].values[order], label="True")
plt.plot(val_out["time"].values[order], val_out["y_pred_raw"].values[order], label="CatBoost")
plt.plot(val_out["time"].values[order], np.full(len(order), global_mean),   label="Baseline (GlobalMean)")
plt.plot(val_out["time"].values[order], np.full(len(order), global_median), label="Baseline (GlobalMedian)")
plt.title("Val (2018) — Time Series with Constant Baselines")
plt.xlabel("Time"); plt.ylabel(TARGET_COL); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_val_timeseries_const_baselines.png"), dpi=160)


# 2) 残差随时间
plt.figure(figsize=(12, 4.7))
plt.plot(val_out["time"].values[order], val_out["residual"].values[order])
plt.title("Val (2018) — Residuals over time (True - Pred)")
plt.xlabel("Time"); plt.ylabel("Residual"); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_val_residuals.png"), dpi=160)

# 3) 残差直方图
plt.figure(figsize=(6.5, 4.7))
plt.hist(val_out["residual"].values, bins=50)
plt.title("Val (2018) — Residuals histogram")
plt.xlabel("Residual"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_val_residual_hist.png"), dpi=160)

# 4) 预测 vs 实际（散点图）
plt.figure(figsize=(5.8, 5.8))
x = val_out["y_true_raw"].values
y = val_out["y_pred_raw"].values
plt.scatter(x, y, s=10, alpha=0.6)
mn, mx = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
plt.plot([mn, mx], [mn, mx], linewidth=2)  # y=x 参考线
plt.title("Val (2018) — Predicted vs Actual")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_val_pred_vs_actual.png"), dpi=160)

print("Saved plots (VAL): plot_val_timeseries_const_baselines.png / plot_val_residuals.png / plot_val_residual_hist.png / plot_val_pred_vs_actual.png")
# ========================= VAL (2018) — Model vs Baseline 柱状图 =========================
# 计算 CatBoost、GlobalMean、GlobalMedian 在 VAL 集上的 RMSE / MAE
rmse_cb  = mean_squared_error(y_va_raw, va_pred_raw, squared=False)
mae_cb   = mean_absolute_error(y_va_raw, va_pred_raw)

val_mean = np.full_like(y_va_raw, fill_value=global_mean, dtype=float)
val_med  = np.full_like(y_va_raw, fill_value=global_median, dtype=float)

rmse_mean = mean_squared_error(y_va_raw, val_mean, squared=False)
mae_mean  = mean_absolute_error(y_va_raw, val_mean)

rmse_median = mean_squared_error(y_va_raw, val_med, squared=False)
mae_median  = mean_absolute_error(y_va_raw, val_med)

labels = ["CatBoost", "GlobalMean", "GlobalMedian"]
rmse_vals = [rmse_cb, rmse_mean, rmse_median]
mae_vals  = [mae_cb,  mae_mean,  mae_median]

x = np.arange(len(labels))
width = 0.37

plt.figure(figsize=(7.8, 4.6))
plt.bar(x - width/2, rmse_vals, width, label="RMSE")
plt.bar(x + width/2, mae_vals,  width, label="MAE")
plt.xticks(x, labels)
plt.ylabel("Error")
plt.title("Val (2018) — Model vs Baselines (RMSE & MAE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_val_model_vs_baseline_bars.png"), dpi=160)

print("Saved plot: plot_val_model_vs_baseline_bars.png")

# ========================= Actual vs Predicted（仅两条线） =========================
# VAL (2018)
order = val_order
plt.figure(figsize=(12, 4.7))
plt.plot(val_out["time"].values[order], val_out["y_true_raw"].values[order], label="Actual")
plt.plot(val_out["time"].values[order], val_out["y_pred_raw"].values[order], label="Predicted")
plt.title("VAL — Time Series: Actual vs. Predicted")
plt.xlabel("Time"); plt.ylabel(TARGET_COL); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_val_timeseries_actual_vs_pred.png"), dpi=160)

# TEST (2019–2020)
order = test_order
plt.figure(figsize=(12, 4.7))
plt.plot(test_out["time"].values[order], test_out["y_true_raw"].values[order], label="Actual")
plt.plot(test_out["time"].values[order], test_out["y_pred_raw"].values[order], label="Predicted")
plt.title("TEST — Time Series: Actual vs. Predicted")
plt.xlabel("Time"); plt.ylabel(TARGET_COL); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_test_timeseries_actual_vs_pred.png"), dpi=160)

print("Saved plots: plot_val_timeseries_actual_vs_pred.png / plot_test_timeseries_actual_vs_pred.png")

# ========================= TEST (2019–2020) — Model vs Baseline 柱状图 =========================
# 计算 CatBoost、GlobalMean、GlobalMedian 在 TEST 集上的 RMSE / MAE
rmse_cb_te = mean_squared_error(y_te_raw, te_pred_raw, squared=False)
mae_cb_te  = mean_absolute_error(y_te_raw, te_pred_raw)

test_mean  = np.full_like(y_te_raw, fill_value=global_mean, dtype=float)
test_med   = np.full_like(y_te_raw, fill_value=global_median, dtype=float)

rmse_mean_te   = mean_squared_error(y_te_raw, test_mean, squared=False)
mae_mean_te    = mean_absolute_error(y_te_raw, test_mean)
rmse_median_te = mean_squared_error(y_te_raw, test_med,  squared=False)
mae_median_te  = mean_absolute_error(y_te_raw, test_med)

labels_te = ["CatBoost", "GlobalMean", "GlobalMedian"]
rmse_vals_te = [rmse_cb_te, rmse_mean_te, rmse_median_te]
mae_vals_te  = [mae_cb_te,  mae_mean_te,  mae_median_te]

x = np.arange(len(labels_te))
width = 0.37

plt.figure(figsize=(7.8, 4.6))
plt.bar(x - width/2, rmse_vals_te, width, label="RMSE")
plt.bar(x + width/2, mae_vals_te,  width, label="MAE")
plt.xticks(x, labels_te)
plt.ylabel("Error")
plt.title("Test (2019–2020) — Model vs Baselines (RMSE & MAE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_test_model_vs_baseline_bars.png"), dpi=160)

print("Saved plot: plot_test_model_vs_baseline_bars.png")


# In[ ]:




