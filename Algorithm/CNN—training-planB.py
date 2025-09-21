# ============================================================
# Aurora Intensity CNN (Polars + PyTorch, no .numpy()/from_numpy)

# ------------------------------------------------------------
# - Target: mean
# - Missing values: Impute with training set column medians; Standardization: Use training set mean/std
# - Model: 1-D CNN
# - Evaluation: MSE/MAE/R²; Permutation importance on validation set (Top-15)
# ------------------------------------------------------------
# ============================================================

import os
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------- Reproducibility --------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- Paths & Columns --------------------
CSV_PATH = "final-planb-24.csv"           # Read directly from the current directory
TARGET_COL = "keogram_mean"
DROP_COLS = {"time"}
EXCLUDE_KP_AP = {"kp_index", "ap_index", "kpindex", "apindex"}

# -------------------- 1) Load with Polars --------------------
assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
df = pl.read_csv(CSV_PATH, try_parse_dates=True)

# Ensure 'time' column is of Datetime type
if df.schema.get("time") != pl.Datetime:
    df = df.with_columns(pl.col("time").str.strptime(pl.Datetime, strict=False))

print(f"Loaded: {CSV_PATH} | Shape: {df.shape}")
if df.select(pl.col("time").is_not_null().sum()).item() > 0:
    tmin = df.select(pl.col("time").min()).item()
    tmax = df.select(pl.col("time").max()).item()
    print("Time range:", tmin, "->", tmax)

# Check for target column existence & drop rows with y=NaN
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")
df = df.filter(pl.col(TARGET_COL).is_not_null())
print("After drop NaN targets:", df.shape)

# -------------------------------------------------------------------------------

# -------------------- 2) Feature selection -------------------
numeric_dtypes = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64}
numeric_cols = [c for c, dt in df.schema.items() if dt in numeric_dtypes]

def is_kp_ap(col: str) -> bool:
    return col.lower() in EXCLUDE_KP_AP

features = [c for c in numeric_cols if c not in DROP_COLS and not is_kp_ap(c)]
if not features:
    raise ValueError("No numeric features left after exclusions.")
print(f"Using {len(features)} features (kp/ap excluded).")

# -------------------- 3) Time-based splits -------------------
START_2017 = datetime(2017, 1, 1)
START_2019 = datetime(2019, 1, 1)
START_2021 = datetime(2021, 1, 1)

train_df = df.filter(pl.col("time") < START_2017)
val_df   = df.filter((pl.col("time") >= START_2017) & (pl.col("time") < START_2019))
test_df  = df.filter((pl.col("time") >= START_2019) & (pl.col("time") < START_2021))

print("Split sizes:",
      "train =", train_df.height,
      "val =",   val_df.height,
      "test =",  test_df.height)

for part, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    if d.height == 0:
        raise ValueError(f"{part} split is empty. Check your time range and data.")

# Extract matrices
X_train = train_df.select(features).to_numpy().astype(np.float32)
X_val   = val_df.select(features).to_numpy().astype(np.float32)
X_test  = test_df.select(features).to_numpy().astype(np.float32)
y_train = train_df.select(TARGET_COL).to_numpy().astype(np.float32).ravel()
y_val   = val_df.select(TARGET_COL).to_numpy().astype(np.float32).ravel()
y_test  = test_df.select(TARGET_COL).to_numpy().astype(np.float32).ravel()

# -------------------- 4) Impute (median) + Standardize -------------------
# Use statistics from the training set
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

# reshape to NCHW-like for 1D conv: (N, C=1, L)
L = X_train.shape[1]

# Avoid torch.from_numpy; use .tolist() + torch.tensor instead
X_train_t = torch.tensor(X_train.tolist(), dtype=torch.float32).unsqueeze(1)  # (N,1,L)
X_val_t   = torch.tensor(X_val.tolist(),   dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test.tolist(),  dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train.tolist(), dtype=torch.float32)
y_val_t   = torch.tensor(y_val.tolist(),   dtype=torch.float32)
y_test_t  = torch.tensor(y_test.tolist(),  dtype=torch.float32)

# -------------------- 5) PyTorch 1-D CNN + Grid Search -------------------
import json
from itertools import product

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
            ch *= 2  # Double the number of channels for each block (32->64->128...)
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

# Training parameters (can be adjusted as needed)
EPOCHS   = 200
BATCH    = 256
LR       = 1e-3
PATIENCE = 10
L        = X_train_t.shape[-1]

def mse_np(a,b): return float(np.mean((a-b)**2))
def mae_np(a,b): return float(np.mean(np.abs(a-b)))
def r2_np(a,b):
    ss_res = float(np.sum((a-b)**2)); ss_tot = float(np.sum((a - np.mean(a))**2))
    return 0.0 if ss_tot == 0 else float(1 - ss_res/ss_tot)

def fit_and_eval(config):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = CNN1D(length=L,
                  num_blocks=config["num_blocks"],
                  base_filters=config["base_filters"],
                  dropout=config["dropout"]).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=1024, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    pat = 0
    best_epoch = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        va_mse = float(np.mean(val_losses))

        # Early stopping tracking
        if va_mse < best_val - 1e-6:
            best_val = va_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                # print(f"[ES] epoch={ep} best_val={best_val:.6f}")
                break

    # Restore the best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Calculate final training/validation metrics
    model.eval()
    with torch.no_grad():
        yhat_tr_t = model(X_train_t.to(device)).detach().cpu().squeeze()
        yhat_va_t = model(X_val_t.to(device)).detach().cpu().squeeze()
    yhat_tr = np.array(yhat_tr_t.tolist(), dtype=np.float32)
    yhat_va = np.array(yhat_va_t.tolist(), dtype=np.float32)

    metrics = {
        "train_mse": mse_np(y_train, yhat_tr),
        "train_mae": mae_np(y_train, yhat_tr),
        "train_r2" : r2_np(y_train, yhat_tr),
        "val_mse"  : mse_np(y_val, yhat_va),
        "val_mae"  : mae_np(y_val, yhat_va),
        "val_r2"   : r2_np(y_val, yhat_va),
        "best_epoch": int(best_epoch)
    }
    return model, best_state, metrics

# --------- Hyperparameter Grid (Core: num_layers + num_channels; Extra: dropout) ----------
GRID_NUM_BLOCKS   = [1, 2, 3]
GRID_BASE_FILTERS = [32, 48, 64]
GRID_DROPOUT      = [0.0, 0.15, 0.30]

results = []
best_overall = None  # (val_mse, state_dict, config, metrics)

total = len(GRID_NUM_BLOCKS) * len(GRID_BASE_FILTERS) * len(GRID_DROPOUT)
trial = 0

for nb, bf, dr in product(GRID_NUM_BLOCKS, GRID_BASE_FILTERS, GRID_DROPOUT):
    trial += 1
    cfg = {"num_blocks": nb, "base_filters": bf, "dropout": dr}
    print(f"\n[Trial {trial}/{total}] config={cfg}")
    model, state, metrics = fit_and_eval(cfg)
    print(f" -> VAL: MSE={metrics['val_mse']:.6f}  MAE={metrics['val_mae']:.6f}  R2={metrics['val_r2']:.6f}  (best_epoch={metrics['best_epoch']})")

    row = {**cfg, **metrics}
    results.append(row)

    if (best_overall is None) or (metrics["val_mse"] < best_overall[0] - 1e-12):
        best_overall = (metrics["val_mse"], state, cfg, metrics)

    # Release the temporary model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save and display the comparison table
res_df = pl.DataFrame(results).sort("val_mse")
print("\n===== Grid Search Results (sorted by val_mse) =====")
print(res_df)
res_df.write_csv("tuning_results.csv")

# --------- Re-create the model with the best config and evaluate on the TEST set ----------
assert best_overall is not None
best_val_mse, best_state, best_cfg, best_metrics = best_overall
print(f"\nBest Config: {best_cfg}  -> VAL_MSE={best_val_mse:.6f}")

best_model = CNN1D(length=L,
                   num_blocks=best_cfg["num_blocks"],
                   base_filters=best_cfg["base_filters"],
                   dropout=best_cfg["dropout"]).to(device)
best_model.load_state_dict(best_state)
best_model.eval()

with torch.no_grad():
    yhat_te_t = best_model(X_test_t.to(device)).detach().cpu().squeeze()
yhat_te = np.array(yhat_te_t.tolist(), dtype=np.float32)

test_metrics = {
    "test_mse": mse_np(y_test, yhat_te),
    "test_mae": mae_np(y_test, yhat_te),
    "test_r2" : r2_np(y_test, yhat_te),
}
print(f"TEST -> MSE: {test_metrics['test_mse']:.4f}  MAE: {test_metrics['test_mae']:.4f}  R2: {test_metrics['test_r2']:.4f}")

# --------- Save the best model (including standardization params and feature names for deployment) ----------
save_payload = {
    "state_dict": {k: v.cpu() for k, v in best_state.items()},
    "config": best_cfg,
    "metrics": {"val": best_metrics, "test": test_metrics},
    "features": features,
    "mu": mu.tolist(),
    "std": std.tolist(),
}
torch.save(save_payload, "best_cnn.pt")
with open("best_config.json", "w", encoding="utf-8") as f:
    json.dump(save_payload["config"], f, ensure_ascii=False, indent=2)
with open("best_metrics.json", "w", encoding="utf-8") as f:
    json.dump(save_payload["metrics"], f, ensure_ascii=False, indent=2)

print("\nSaved: best_cnn2.pt, best_config2.json, best_metrics2.json, tuning_results2.csv")

# -------------------- 6) Permutation Importance (VAL for best model) -------------------
best_model.eval()
with torch.no_grad():
    base_pred_t = best_model(X_val_t.to(device)).detach().cpu().squeeze()
    base_pred   = np.array(base_pred_t.tolist(), dtype=np.float32)
base_mse = float(np.mean((y_val - base_pred)**2))

importances = np.zeros(L, dtype=np.float32)
Xv = X_val_t.clone()
torch.manual_seed(SEED + 1234)

for j in range(L):
    col = Xv[:, 0, j].clone()
    perm = torch.randperm(Xv.shape[0])
    Xv[:, 0, j] = Xv[:, 0, j][perm]

    with torch.no_grad():
        pred_t = best_model(Xv.to(device)).detach().cpu().squeeze()
        pred   = np.array(pred_t.tolist(), dtype=np.float32)

    importances[j] = float(np.mean((y_val - pred)**2) - base_mse)
    Xv[:, 0, j] = col

order = np.argsort(importances)[::-1]
print("\nTop-15 permutation importances on VAL (ΔMSE when shuffled) [Best Model]:")
for k in range(min(15, L)):
    j = int(order[k])
    print(f"{k+1:2d}. {features[j]:20s} +{importances[j]:.6f}")
