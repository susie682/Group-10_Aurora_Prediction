# ============================================================
# Aurora Intensity CNN (Polars + PyTorch, no .numpy()/from_numpy)
# Author: Susie + Group 10 (COMPSCI 760)
# ------------------------------------------------------------
# NEW (kept):
# 1) Target: mild tail capping on TRAIN (q=0.995) + log1p; evaluate on original scale.
# 2) Rank-based reweighting for tail (q_ref≈0.96, α≈12, γ≈5).
# 3) Extra AR features (inject persistence):
#    y_lag1, y_rollmean_6 (shifted), y_rollmax_6 (shifted); persist baseline = y_lag1.
# 4) Smearing correction on inverse-transform (global Duan factor).
# 5) Model selection metric: combo_mae = 0.7*MAE_all + 0.3*MAE_tail(q=0.90)
# CHANGE YOU ASKED (kept):
# • Filter rows where keogram_mean/median/max are non-null, finite, and non-zero.
#
# ADDITIONAL METRICS (kept):
# • Pearson r (original scale), DTW (normalized), F1@extreme (q=0.90 on TRAIN)
# ------------------------------------------------------------
# ============================================================

import os
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import json
from itertools import product

# -------------------- Reproducibility --------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- Paths & Columns --------------------
CSV_PATH = "final-planb-24.csv"
TARGET_COL = "keogram_mean"
DROP_COLS = {"time"}
EXCLUDE_KP_AP = {
    "kp_index", "ap_index", "kpindex", "apindex",
    "keogram_median", "keogram_max", "keogram_mean"
}

# -------------------- Hyperparams / knobs --------------------
# Target transform
LOG_TRANSFORM       = True
WINSORIZE_TRAIN_Y   = True
WINSOR_UPPER_PCT    = 0.995   # mild upper-tail cap (tune 0.99–0.997 if needed)
WINSOR_LOWER_PCT    = None

# Rank-based tail reweight
REWEIGHT_MODE       = "weights"     # {"weights", "oversample", "none"}
RANK_QREF           = 0.96
RANK_ALPHA          = 12.0
RANK_GAMMA          = 5.0

# F1 extreme threshold (on TRAIN, original scale)
F1_EXTREME_PCT      = 0.90

# DTW window (Sakoe-Chiba band); None = full
DTW_WINDOW_FRAC     = 0.10

# Smearing (Duan) correction for inverse-transform
SMEARING_ON         = True

# Training parameters
EPOCHS   = 200
BATCH    = 256
LR       = 1e-3
PATIENCE = 10

# --------- Hyperparameter Grid ----------
GRID_NUM_BLOCKS   = [2, 3, 5]
GRID_BASE_FILTERS = [24, 48, 96]
GRID_DROPOUT      = [0, 0.15, 0.3]

# -------------------- 1) Load with Polars --------------------
assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
df = pl.read_csv(CSV_PATH, try_parse_dates=True)

# Ensure 'time' column is Datetime
if df.schema.get("time") != pl.Datetime:
    df = df.with_columns(pl.col("time").str.strptime(pl.Datetime, strict=False))

print(f"Loaded: {CSV_PATH} | Shape: {df.shape}")
if df.select(pl.col("time").is_not_null().sum()).item() > 0:
    tmin = df.select(pl.col("time").min()).item()
    tmax = df.select(pl.col("time").max()).item()
    print("Time range:", tmin, "->", tmax)

# Columns exist?
REQUIRED_NONZERO = [TARGET_COL, "keogram_median", "keogram_max"]
missing = [c for c in REQUIRED_NONZERO if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------- Clean → Filter → AR features --------------------
# 1) Cast to Float64 for numeric ops (non-numeric -> null)
NUM_COLS = [TARGET_COL, "keogram_median", "keogram_max"]
df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in NUM_COLS])

# 2) Filter: drop null/NaN/Inf/0 rows on all three keogram columns
def ok_numeric(colname: str):
    c = pl.col(colname)
    return c.is_not_null() & c.is_finite() & (c != 0)

_before = df.shape
df = df.filter(
    ok_numeric(TARGET_COL) &
    ok_numeric("keogram_median") &
    ok_numeric("keogram_max")
)
print(f"After clean-numeric + null/NaN/Inf/zero filter: {_before} -> {df.shape}")

# 3) Sort then build AR/short-window features; shift(1) to avoid leakage at t
df = df.sort("time").with_columns([
    pl.col(TARGET_COL).shift(1).alias("y_lag1"),
    pl.col(TARGET_COL).rolling_mean(window_size=6).shift(1).alias("y_rollmean_6"),
    pl.col(TARGET_COL).rolling_max(window_size=6).shift(1).alias("y_rollmax_6"),
]).drop_nulls(subset=["y_lag1", "y_rollmean_6", "y_rollmax_6"])

print("After AR feature creation & drop_nulls:", df.shape)

# -------------------- 2) Feature selection -------------------
numeric_dtypes = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64}
numeric_cols = [c for c, dt in df.schema.items() if dt in numeric_dtypes]

def is_kp_ap(col: str) -> bool:
    return col.lower() in EXCLUDE_KP_AP

base_features = [c for c in numeric_cols if c not in DROP_COLS and not is_kp_ap(c) and c != TARGET_COL]

# Append AR features explicitly (ensure order)
AR_FEATURES = ["y_lag1", "y_rollmean_6", "y_rollmax_6"]
for c in AR_FEATURES:
    if c not in base_features:
        base_features.append(c)

features = base_features
if not features:
    raise ValueError("No numeric features left after exclusions.")
print(f"Using {len(features)} features (kp/ap excluded) + AR features injected.")

# -------------------- 3) Time-based splits (sorted) -------------------
# Note: variable names kept from prior script for continuity.
START_2021 = datetime(2018, 1, 1)
START_2023 = datetime(2019, 1, 1)
START_2025 = datetime(2021, 1, 1)

train_df = df.filter(pl.col("time") < START_2021).sort("time")
val_df   = df.filter((pl.col("time") >= START_2021) & (pl.col("time") < START_2023)).sort("time")
test_df  = df.filter((pl.col("time") >= START_2023) & (pl.col("time") < START_2025)).sort("time")

print("Split sizes:",
      "train =", train_df.height,
      "val =",   val_df.height,
      "test =",  test_df.height)

for part, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if d.height == 0:
        raise ValueError(f"{part} split is empty. Check your time range and data.")

# Persist baseline = lag1 (already shifted)
persist_train = train_df.select("y_lag1").to_numpy().astype(np.float32).ravel()
persist_val   = val_df.select("y_lag1").to_numpy().astype(np.float32).ravel()
persist_test  = test_df.select("y_lag1").to_numpy().astype(np.float32).ravel()

# Extract matrices (features)
X_train = train_df.select(features).to_numpy().astype(np.float32)
X_val   = val_df.select(features).to_numpy().astype(np.float32)
X_test  = test_df.select(features).to_numpy().astype(np.float32)

# Extract raw targets (original scale)
y_train = train_df.select(TARGET_COL).to_numpy().astype(np.float32).ravel()
y_val   = val_df.select(TARGET_COL).to_numpy().astype(np.float32).ravel()
y_test  = test_df.select(TARGET_COL).to_numpy().astype(np.float32).ravel()

# -------------------- 4) Impute (median) + Standardize X -------------------
med = np.nanmedian(X_train, axis=0)
med[~np.isfinite(med)] = 0.0

def impute(arr, med):
    out = arr.copy()
    mask = ~np.isfinite(out)
    if mask.any():
        col_idx = np.nonzero(mask)[1]
        out[mask] = med[col_idx]
    return out

X_train = impute(X_train, med)
X_val   = impute(X_val, med)
X_test  = impute(X_test, med)

mu  = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std < 1e-12] = 1.0

def standardize(a, mu, std): return ((a - mu) / std).astype(np.float32)
X_train = standardize(X_train, mu, std)
X_val   = standardize(X_val,   mu, std)
X_test  = standardize(X_test,  mu, std)

# -------------------- 5) Target: winsor (train) + log1p --------------------
def winsorize_train(y, lower_pct=None, upper_pct=None):
    y_w = y.copy()
    lo = None if lower_pct is None else float(np.quantile(y, lower_pct))
    hi = None if upper_pct is None else float(np.quantile(y, upper_pct))
    if lo is not None:
        y_w = np.maximum(y_w, lo)
    if hi is not None:
        y_w = np.minimum(y_w, hi)
    return y_w, lo, hi

if WINSORIZE_TRAIN_Y:
    y_train_cap, q_low, q_high = winsorize_train(
        y_train, WINSOR_LOWER_PCT, WINSOR_UPPER_PCT
    )
else:
    y_train_cap, q_low, q_high = y_train.copy(), None, None

SHIFT = 0.0
if LOG_TRANSFORM:
    min_after = float(np.min(y_train_cap))
    if min_after < 0:
        SHIFT = -min_after + 1e-6
    y_train_tr = np.log1p(y_train_cap + SHIFT).astype(np.float32)
    y_val_tr   = np.log1p(np.maximum(y_val + SHIFT, 0.0)).astype(np.float32)
else:
    y_train_tr = y_train_cap.astype(np.float32)
    y_val_tr   = y_val.astype(np.float32)

print(f"[Target transform] WINSOR(train): lower={WINSOR_LOWER_PCT} upper={WINSOR_UPPER_PCT} | "
      f"LOG={LOG_TRANSFORM} | SHIFT={SHIFT:.4g}")
if q_low is not None or q_high is not None:
    print(f"[Train y quantiles] q_low={q_low}, q_high={q_high}")

# F1 extreme threshold (on original scale, from TRAIN)
extreme_thresh = float(np.quantile(y_train, F1_EXTREME_PCT))
print(f"[Extreme threshold] q={int(F1_EXTREME_PCT*100)} -> {extreme_thresh:.6f}")

# -------------------- 6) Rank-based tail weights / oversampling --------------------
n_tr = len(y_train)
order = np.argsort(y_train)
ranks = np.empty_like(y_train, dtype=np.float64)
ranks[order] = (np.arange(n_tr, dtype=np.float64) + 0.5) / n_tr  # mid-ranks in [0,1)

w_train = np.ones_like(y_train, dtype=np.float32)
mask = ranks >= RANK_QREF
w_train[mask] = 1.0 + RANK_ALPHA * ((ranks[mask] - RANK_QREF) / (1.0 - RANK_QREF))**RANK_GAMMA
print(f"[Reweight(rank)] mode={REWEIGHT_MODE} | q_ref={RANK_QREF} α={RANK_ALPHA} γ={RANK_GAMMA} | "
      f"max_w={w_train.max():.2f}")

# -------------------- 6.1 Metrics helpers --------------------
def mse_np(a,b): return float(np.mean((a-b)**2))
def rmse_np(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mae_np(a,b): return float(np.mean(np.abs(a-b)))
def r2_np(a,b):
    ss_res = float(np.sum((a-b)**2)); ss_tot = float(np.sum((a - np.mean(a))**2))
    return 0.0 if ss_tot == 0 else float(1 - ss_res/ss_tot)

def mae_tail_np(y_true, y_pred, thresh):
    m = y_true >= thresh
    if not np.any(m): return float("nan")
    return mae_np(y_true[m], y_pred[m])

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    am = a.mean(); bm = b.mean()
    av = a - am; bv = b - bm
    denom = np.sqrt((av*av).sum()) * np.sqrt((bv*bv).sum())
    if denom == 0:
        return 0.0
    return float((av*bv).sum() / denom)

def dtw_distance(s: np.ndarray, t: np.ndarray, window_frac: Optional[float] = DTW_WINDOW_FRAC) -> float:
    s = s.astype(np.float64); t = t.astype(np.float64)
    n, m = len(s), len(t)
    if n == 0 or m == 0:
        return float("nan")
    if window_frac is None:
        w = max(n, m)
    else:
        w = int(max(n, m) * float(window_frac))
        w = max(w, abs(n - m))
    INF = 1e30
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end   = min(m, i + w)
        si = s[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(si - t[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    dist = D[n, m]
    return float(dist / (n + m))

def f1_for_extremes(y_true: np.ndarray, y_pred: np.ndarray, threshold: float):
    y_true_pos = (y_true >= threshold)
    y_pred_pos = (y_pred >= threshold)
    tp = int(np.sum(y_true_pos & y_pred_pos))
    fp = int(np.sum(~y_true_pos & y_pred_pos))
    fn = int(np.sum(y_true_pos & ~y_pred_pos))
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall    = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return float(f1), float(precision), float(recall), tp, fp, fn

# -------------------- 7) Torch tensors --------------------
# reshape to (N,1,L) for Conv1D; avoid torch.from_numpy()
X_train_t = torch.tensor(X_train.tolist(), dtype=torch.float32).unsqueeze(1)
X_val_t   = torch.tensor(X_val.tolist(),   dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test.tolist(),  dtype=torch.float32).unsqueeze(1)

y_train_t_tr = torch.tensor(y_train_tr.tolist(), dtype=torch.float32)  # transformed target for loss
y_val_t_tr   = torch.tensor(y_val_tr.tolist(),   dtype=torch.float32)
y_train_t_w  = torch.tensor(w_train.tolist(),    dtype=torch.float32)  # sample weights

# -------------------- 8) PyTorch 1-D CNN + Grid Search -------------------
class CNN1D(nn.Module):
    def __init__(self, length: int, num_blocks: int, base_filters: int, dropout: float):
        super().__init__()
        assert num_blocks >= 1
        self.blocks = nn.ModuleList()
        in_ch = 1
        ch = base_filters
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_ch = ch
            ch *= 2
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_ch, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):  # x: (N,1,L)
        for blk in self.blocks:
            x = blk(x)
        x = self.gap(x).squeeze(-1)  # (N, C_last)
        x = self.act(self.fc1(x))
        y = self.fc2(x).squeeze(-1)  # (N,)
        return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = X_train_t.shape[-1]

def make_train_loader():
    if REWEIGHT_MODE == "oversample":
        sample_weights = torch.tensor(w_train.tolist(), dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        return DataLoader(TensorDataset(X_train_t, y_train_t_tr),
                          batch_size=BATCH, sampler=sampler)
    else:
        return DataLoader(TensorDataset(X_train_t, y_train_t_tr, y_train_t_w),
                          batch_size=BATCH, shuffle=True)

def inverse_transform_core(y_pred_tr):
    if LOG_TRANSFORM:
        y_back = np.expm1(y_pred_tr) - SHIFT
        return np.maximum(y_back, 0.0)
    else:
        return y_pred_tr

def apply_smearing(yhat_tr_t_np, y_train_tr_np):
    if not SMEARING_ON or not LOG_TRANSFORM:
        return 1.0, lambda arr: inverse_transform_core(arr)
    eps = y_train_tr_np - yhat_tr_t_np  # residuals in transformed space
    k = float(np.mean(np.exp(eps)))
    def inv_with_smear(arr):
        base = inverse_transform_core(arr)
        return base * k
    return k, inv_with_smear

def fit_and_eval(config):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = CNN1D(length=L,
                  num_blocks=config["num_blocks"],
                  base_filters=config["base_filters"],
                  dropout=config["dropout"]).to(device)

    train_loader = make_train_loader()
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t_tr), batch_size=1024, shuffle=False)

    use_weighted = (REWEIGHT_MODE == "weights")
    criterion_tr = nn.MSELoss(reduction='none') if use_weighted else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf"); best_state = None; pat = 0; best_epoch = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        for batch in train_loader:
            if use_weighted:
                xb, yb, wb = batch
                wb = wb.to(device)
            else:
                xb, yb = batch
            xb = xb.to(device); yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            if use_weighted:
                loss_vec = criterion_tr(pred, yb)
                loss = (loss_vec * wb).mean()
            else:
                loss = criterion_tr(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_losses.append(nn.MSELoss()(pred, yb).item())
        va_mse = float(np.mean(val_losses))

        if va_mse < best_val - 1e-6:
            best_val = va_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep; pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        yhat_tr_t = model(X_train_t.to(device)).detach().cpu().squeeze()
        yhat_va_t = model(X_val_t.to(device)).detach().cpu().squeeze()

    yhat_tr_t_np = np.array(yhat_tr_t.tolist(), dtype=np.float32)
    yhat_va_t_np = np.array(yhat_va_t.tolist(), dtype=np.float32)
    y_train_tr_np = np.array(y_train_tr.tolist(), dtype=np.float32)

    smear_factor, inv_fn = apply_smearing(yhat_tr_t_np, y_train_tr_np)

    yhat_tr = inv_fn(yhat_tr_t_np)
    yhat_va = inv_fn(yhat_va_t_np)

    metrics = {
        "train_mse": mse_np(y_train, yhat_tr),
        "train_rmse": rmse_np(y_train, yhat_tr),
        "train_mae": mae_np(y_train, yhat_tr),
        "train_r2" : r2_np(y_train, yhat_tr),
        "val_mse"  : mse_np(y_val,   yhat_va),
        "val_rmse" : rmse_np(y_val,  yhat_va),
        "val_mae"  : mae_np(y_val,   yhat_va),
        "val_r2"   : r2_np(y_val,    yhat_va),
        "best_epoch": int(best_epoch),
        "smear_factor": float(smear_factor),
    }

    metrics["train_corr"] = pearson_r(y_train, yhat_tr)
    metrics["val_corr"]   = pearson_r(y_val,   yhat_va)
    metrics["train_dtw"]  = dtw_distance(y_train, yhat_tr, DTW_WINDOW_FRAC)
    metrics["val_dtw"]    = dtw_distance(y_val,   yhat_va, DTW_WINDOW_FRAC)

    f1_tr, p_tr, r_tr, tp_tr, fp_tr, fn_tr = f1_for_extremes(y_train, yhat_tr, extreme_thresh)
    f1_va, p_va, r_va, tp_va, fp_va, fn_va = f1_for_extremes(y_val,   yhat_va, extreme_thresh)

    metrics.update({
        "train_f1_extreme": f1_tr,
        "train_precision_extreme": p_tr,
        "train_recall_extreme": r_tr,
        "train_tp_extreme": tp_tr,
        "train_fp_extreme": fp_tr,
        "train_fn_extreme": fn_tr,
        "val_f1_extreme": f1_va,
        "val_precision_extreme": p_va,
        "val_recall_extreme": r_va,
        "val_tp_extreme": tp_va,
        "val_fp_extreme": fp_va,
        "val_fn_extreme": fn_va,
    })

    tail_mae_val = mae_tail_np(y_val, yhat_va, extreme_thresh)
    combo_val = 0.7 * metrics["val_mae"] + 0.3 * tail_mae_val
    metrics["val_tail_mae_q90"] = float(tail_mae_val)
    metrics["val_combo_mae"] = float(combo_val)

    mean_train = float(np.mean(y_train))
    median_train = float(np.median(y_train))
    val_baseline_mean   = np.full_like(y_val, mean_train, dtype=np.float32)
    val_baseline_median = np.full_like(y_val, median_train, dtype=np.float32)
    metrics.update({
        "val_baseline_persist_mae": mae_np(y_val, persist_val),
        "val_baseline_mean_mae": mae_np(y_val, val_baseline_mean),
        "val_baseline_median_mae": mae_np(y_val, val_baseline_median),
    })

    return model, best_state, metrics

results = []
best_overall = None  # (val_combo_mae, state_dict, config, metrics)

total = len(GRID_NUM_BLOCKS) * len(GRID_BASE_FILTERS) * len(GRID_DROPOUT)
trial = 0

for nb, bf, dr in product(GRID_NUM_BLOCKS, GRID_BASE_FILTERS, GRID_DROPOUT):
    trial += 1
    cfg = {"num_blocks": nb, "base_filters": bf, "dropout": dr}
    print(f"\n[Trial {trial}/{total}] config={cfg}")
    model, state, metrics = fit_and_eval(cfg)
    print(f" -> VAL(orig): MSE={metrics['val_mse']:.6f}  MAE={metrics['val_mae']:.6f}  "
          f"TailMAE@q90={metrics['val_tail_mae_q90']:.6f}  ComboMAE={metrics['val_combo_mae']:.6f}  "
          f"R2={metrics['val_r2']:.4f}  r={metrics['val_corr']:.4f}  DTW={metrics['val_dtw']:.4f}  "
          f"F1@q90={metrics['val_f1_extreme']:.4f}  (best_epoch={metrics['best_epoch']}, smear={metrics['smear_factor']:.4f})")

    row = {**cfg, **metrics}
    results.append(row)

    if (best_overall is None) or (metrics["val_combo_mae"] < best_overall[0] - 1e-12):
        best_overall = (metrics["val_combo_mae"], state, cfg, metrics)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

res_df = pl.DataFrame(results).sort("val_combo_mae")
print("\n===== Grid Search Results (sorted by val_combo_mae on ORIGINAL scale) =====")
print(res_df)
res_df.write_csv("tuning_results_persist.csv")

# --------- Re-create the model with the best config and evaluate on the TEST set ----------
assert best_overall is not None
best_combo, best_state, best_cfg, best_metrics = best_overall
print(f"\nBest Config: {best_cfg}  -> VAL_ComboMAE(orig)={best_combo:.6f}")

best_model = CNN1D(length=L,
                   num_blocks=best_cfg["num_blocks"],
                   base_filters=best_cfg["base_filters"],
                   dropout=best_cfg["dropout"]).to(device)
best_model.load_state_dict(best_state)
best_model.eval()

with torch.no_grad():
    yhat_te_t = best_model(X_test_t.to(device)).detach().cpu().squeeze()

with torch.no_grad():
    yhat_tr_t_best = best_model(X_train_t.to(device)).detach().cpu().squeeze()
yhat_tr_t_best_np = np.array(yhat_tr_t_best.tolist(), dtype=np.float32)
y_train_tr_np = np.array(y_train_tr.tolist(), dtype=np.float32)
smear_factor_best, inv_fn_best = apply_smearing(yhat_tr_t_best_np, y_train_tr_np)

yhat_te = inv_fn_best(np.array(yhat_te_t.tolist(), dtype=np.float32))

test_metrics = {
    "test_mse": mse_np(y_test, yhat_te),
    "test_rmse": rmse_np(y_test, yhat_te),
    "test_mae": mae_np(y_test, yhat_te),
    "test_r2" : r2_np(y_test,  yhat_te),
    "test_tail_mae_q90": mae_tail_np(y_test, yhat_te, extreme_thresh),
    "test_corr": pearson_r(y_test, yhat_te),
    "test_dtw" : dtw_distance(y_test, yhat_te, DTW_WINDOW_FRAC),
}
f1_te, p_te, r_te, tp_te, fp_te, fn_te = f1_for_extremes(y_test, yhat_te, extreme_thresh)
test_metrics.update({
    "test_f1_extreme": f1_te,
    "test_precision_extreme": p_te,
    "test_recall_extreme": r_te,
    "test_tp_extreme": tp_te,
    "test_fp_extreme": fp_te,
    "test_fn_extreme": fn_te,
    "smear_factor": smear_factor_best,
})

print(f"TEST (original) -> RMSE: {test_metrics['test_rmse']:.4f}  MAE: {test_metrics['test_mae']:.4f}  "
      f"MSE: {test_metrics['test_mse']:.4f}  R2: {test_metrics['test_r2']:.4f}  r: {test_metrics['test_corr']:.4f}  "
      f"DTW: {test_metrics['test_dtw']:.4f}  F1@q90: {test_metrics['test_f1_extreme']:.4f}")

mean_train = float(np.mean(y_train))
median_train = float(np.median(y_train))
test_baseline_mean   = np.full_like(y_test, mean_train, dtype=np.float32)
test_baseline_median = np.full_like(y_test, median_train, dtype=np.float32)
print(f"Baselines(TEST) MAE -> persist:{mae_np(y_test, persist_test):.4f} "
      f"mean(train):{mae_np(y_test, test_baseline_mean):.4f} "
      f"median(train):{mae_np(y_test, test_baseline_median):.4f}")

save_payload = {
    "state_dict": {k: v.cpu() for k, v in best_state.items()},
    "config": best_cfg,
    "metrics": {"val": best_metrics, "test": test_metrics},
    "features": features,
    "mu": mu.tolist(),
    "std": std.tolist(),
    "target_transform": {
        "log_transform": LOG_TRANSFORM,
        "winsorize_train": WINSORIZE_TRAIN_Y,
        "winsor_upper_pct": WINSOR_UPPER_PCT,
        "winsor_lower_pct": WINSOR_LOWER_PCT,
        "q_low": None if 'q_low' not in locals() or q_low is None else float(q_low),
        "q_high": None if 'q_high' not in locals() or q_high is None else float(q_high),
        "shift": float(SHIFT),
        "smearing_on": SMEARING_ON,
        "smear_factor": float(test_metrics.get("smear_factor", 1.0)),
        "rank_qref": RANK_QREF,
        "rank_alpha": RANK_ALPHA,
        "rank_gamma": RANK_GAMMA,
        "f1_extreme_pct": F1_EXTREME_PCT,
        "extreme_threshold": float(extreme_thresh),
        "dtw_window_frac": DTW_WINDOW_FRAC,
        "reweight_mode": REWEIGHT_MODE,
    },
}
torch.save(save_payload, "best_cnn_persist.pt")
with open("best_config_persist.json", "w", encoding="utf-8") as f:
    json.dump(save_payload["config"], f, ensure_ascii=False, indent=2)
with open("best_metrics_persist.json", "w", encoding="utf-8") as f:
    json.dump(save_payload["metrics"], f, ensure_ascii=False, indent=2)

print("\nSaved: best_cnn_persist.pt, best_config_persist.json, best_metrics_persist.json, tuning_results_persist.csv")

# -------------------- 9) Permutation Importance (VAL; model fixed) --------------------
best_model.eval()
with torch.no_grad():
    base_pred_tr_t = best_model(X_val_t.to(device)).detach().cpu().squeeze()
base_pred_tr = np.array(base_pred_tr_t.tolist(), dtype=np.float32)
base_mse_tr = float(np.mean((y_val_tr - base_pred_tr)**2))  # transformed-space ΔMSE

L = X_train_t.shape[-1]
importances = np.zeros(L, dtype=np.float32)
Xv = X_val_t.clone()
torch.manual_seed(SEED + 1234)

for j in range(L):
    col = Xv[:, 0, j].clone()
    perm = torch.randperm(Xv.shape[0])
    Xv[:, 0, j] = Xv[:, 0, j][perm]
    with torch.no_grad():
        pred_tr_t = best_model(Xv.to(device)).detach().cpu().squeeze()
    pred_tr = np.array(pred_tr_t.tolist(), dtype=np.float32)
    importances[j] = float(np.mean((y_val_tr - pred_tr)**2) - base_mse_tr)
    Xv[:, 0, j] = col

order = np.argsort(importances)[::-1]
print("\nTop-15 permutation importances on VAL (ΔMSE in TRANSFORMED space) [Best Model]:")
for k in range(min(15, L)):
    j = int(order[k])
    print(f"{k+1:2d}. {features[j]:20s} +{importances[j]:.6f}")
