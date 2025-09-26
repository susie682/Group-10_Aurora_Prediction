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













# ======================== Visualization (TTF auto-discovery) =========================
# Pillow-only drawing with absolute font sizes and robust TrueType auto-discovery.

import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import PIL

# -------- Font sizes (increase these to make text larger) --------
TITLE_PT = 34
LABEL_PT = 28
TICK_PT  = 20
LEG_PT   = 24

# -------- Output dir --------
SAVE_DIR_NAME = "figs_cnn"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, SAVE_DIR_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

def _save(img, path):
    img.save(path, format="PNG")
    print(f"[Saved] {path}")

# -------- TrueType discovery (no env vars, no matplotlib) --------
_PREFERRED_NAMES = ("DejaVuSans", "DejaVu Sans", "Arial", "LiberationSans",
                    "Liberation Sans", "NotoSans", "Noto Sans", "FreeSans", "Free Sans")

def _discover_ttf(max_files_scanned=5000):
    roots = [
        BASE_DIR,
        os.path.join(os.path.dirname(PIL.__file__), "fonts"),  # PIL packaged fonts
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/Library/Fonts",
        "/System/Library/Fonts",
        "C:\\Windows\\Fonts",
    ]

    # 1) Quick direct hits
    quick = [
        os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
    ]
    for p in quick:
        if os.path.exists(p):
            return p

    # 2) Recursive scan (prefer by name)
    found_any = None
    scanned = 0
    def _want(path):
        name = os.path.basename(path).lower()
        return name.endswith(".ttf") or name.endswith(".otf")

    def _score(path):
        base = os.path.basename(path).lower()
        for i, pref in enumerate(_PREFERRED_NAMES):
            if pref.lower().replace(" ", "") in base.replace(" ", ""):
                return 1000 - i  # higher is better
        return 0

    best_path, best_score = None, -1
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                scanned += 1
                if scanned > max_files_scanned:
                    break
                full = os.path.join(dirpath, fn)
                if _want(full):
                    found_any = found_any or full
                    s = _score(full)
                    if s > best_score:
                        best_score, best_path = s, full
            if scanned > max_files_scanned:
                break
        if best_path:
            break

    return best_path or found_any  # may be None

def _load_font(size):
    # Try a few common direct names first (Pillow can sometimes resolve bundled fonts by name)
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass

    # Auto-discover in system/Pillow paths
    ttf = _discover_ttf()
    if ttf and os.path.exists(ttf):
        try:
            return ImageFont.truetype(ttf, size=size)
        except Exception:
            pass

    print("[Warn] No TrueType font found; falling back to default bitmap font (size won't scale).")
    return ImageFont.load_default()

TITLE_FONT = _load_font(TITLE_PT)
LABEL_FONT = _load_font(LABEL_PT)
TICK_FONT  = _load_font(TICK_PT)
LEG_FONT   = _load_font(LEG_PT)

def _text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        pass
    try:
        return font.getsize(text)
    except Exception:
        pass
    size = getattr(font, "size", 24) or 24
    return int(len(text) * size * 0.6), int(size * 1.2)

def _text_center(draw, x, y, text, font, fill=(0, 0, 0)):
    w, _ = _text_size(draw, text, font)
    draw.text((x - w / 2, y), text, fill=fill, font=font)

# -------- Reuse tensors/arrays produced above --------
y_val_pred  = base_pred
y_test_pred = yhat_te

y_val_list       = y_val.tolist()        if hasattr(y_val, "tolist")        else list(y_val)
y_test_list      = y_test.tolist()       if hasattr(y_test, "tolist")       else list(y_test)
y_val_pred_list  = y_val_pred.tolist()   if hasattr(y_val_pred, "tolist")   else list(y_val_pred)
y_test_pred_list = y_test_pred.tolist()  if hasattr(y_test_pred, "tolist")  else list(y_test_pred)

def _to_epoch_list(df_pl):
    arr = df_pl.select(pl.col("time")).to_numpy().ravel().tolist()
    out = []
    for x in arr:
        if isinstance(x, datetime):
            out.append(x.timestamp())
        else:
            s = str(x).replace("Z", "").strip()
            try:
                out.append(datetime.fromisoformat(s).timestamp())
            except Exception:
                try:
                    s2 = s.replace("T", " ")
                    out.append(datetime.fromisoformat(s2).timestamp())
                except Exception:
                    try:
                        out.append(float(x))
                    except:
                        out.append(float(len(out)))
    return out

t_val_sec  = _to_epoch_list(val_df)
t_test_sec = _to_epoch_list(test_df)

TARGET_COL = TARGET_COL if 'TARGET_COL' in globals() else "keogram_mean"

# -------- Metrics (RMSE from MSE) --------
def _mse(y_true, y_pred):
    n = max(1, len(y_true))
    return sum((float(a) - float(b))**2 for a, b in zip(y_true, y_pred)) / n

def _mae(y_true, y_pred):
    n = max(1, len(y_true))
    return sum(abs(float(a) - float(b)) for a, b in zip(y_true, y_pred)) / n

def _mean(vals):
    n = max(1, len(vals))
    return sum(map(float, vals)) / n

def _median(vals):
    v = sorted(map(float, vals))
    n = len(v)
    if n == 0: return 0.0
    m = n // 2
    return v[m] if n % 2 else (v[m-1] + v[m]) / 2.0

# -------- Canvas and colors --------
W, H = 1600, 1000
MARG_L, MARG_R, MARG_T, MARG_B = 210, 140, 190, 240
AX_W, AX_H = W - MARG_L - MARG_R, H - MARG_T - MARG_B
C0 = (31, 119, 180)
C1 = (255, 127, 14)
BLACK = (0, 0, 0)
GRID = (220, 220, 220)

def _draw_axes(draw, title, x_label, y_label, show_frame=True, grid=False):
    if show_frame:
        draw.rectangle((MARG_L, H - MARG_B - AX_H, MARG_L + AX_W, H - MARG_B), outline=BLACK, width=2)
    else:
        draw.line((MARG_L, H - MARG_B, MARG_L + AX_W, H - MARG_B), fill=BLACK, width=2)
        draw.line((MARG_L, H - MARG_B, MARG_L, H - MARG_B - AX_H), fill=BLACK, width=2)
    _text_center(draw, W / 2, 28, title, TITLE_FONT)
    _text_center(draw, W / 2, H - 70, x_label, LABEL_FONT)
    draw.text((28, H / 2 - _text_size(draw, y_label, LABEL_FONT)[1] / 2), y_label, fill=BLACK, font=LABEL_FONT)
    if grid:
        for g in range(1, 6):
            x = MARG_L + AX_W * g / 6.0
            draw.line((x, H - MARG_B - AX_H, x, H - MARG_B), fill=GRID, width=1)

def _scale(val, lo, hi, pix_lo, pix_hi):
    if hi == lo: return (pix_lo + pix_hi) * 0.5
    r = (float(val) - float(lo)) / (float(hi) - float(lo))
    return pix_lo + r * (pix_hi - pix_lo)

def _nice_ticks(lo, hi, k=6):
    if hi == lo: return [lo]*k
    step = (hi - lo) / max(1, (k - 1))
    return [lo + i * step for i in range(k)]

def _fmt(v):
    v = float(v)
    if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0.0): return f"{v:.2e}"
    return f"{v:.2f}"

def _dashed(draw, x0, y0, x1, y1, color, width=4, dash=12, gap=12):
    import math
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist == 0: return
    ux, uy = dx / dist, dy / dist
    s = 0.0
    while s < dist:
        e = min(s + dash, dist)
        draw.line((x0 + ux*s, y0 + uy*s, x1 if e == dist else x0 + ux*e, y1 if e == dist else y0 + uy*e),
                  fill=color, width=width)
        s = e + gap

# -------- 1) Pred vs Actual (CNN) --------
def plot_pred_vs_actual(y_true, y_pred, split, path):
    y_min = min(min(y_true), min(y_pred))
    y_max = max(max(y_true), max(y_pred))
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_axes(draw, f"{split} - CNN Predicted vs Actual", "Actual", "Predicted", show_frame=True)
    x0 = _scale(y_min, y_min, y_max, MARG_L, MARG_L + AX_W)
    y0 = _scale(y_min, y_min, y_max, H - MARG_B, H - MARG_B - AX_H)
    x1 = _scale(y_max, y_min, y_max, MARG_L, MARG_L + AX_W)
    y1 = _scale(y_max, y_min, y_max, H - MARG_B, H - MARG_B - AX_H)
    _dashed(draw, x0, y0, x1, y1, color=C0, width=4, dash=12, gap=12)
    r = 4
    for a, b in zip(y_true, y_pred):
        x = _scale(a, y_min, y_max, MARG_L, MARG_L + AX_W)
        y = _scale(b, y_min, y_max, H - MARG_B, H - MARG_B - AX_H)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=C0)
    for v in _nice_ticks(y_min, y_max, k=6):
        x = _scale(v, y_min, y_max, MARG_L, MARG_L + AX_W)
        y = _scale(v, y_min, y_max, H - MARG_B, H - MARG_B - AX_H)
        draw.line((x, H - MARG_B, x, H - MARG_B + 10), fill=BLACK, width=2)
        draw.text((x - 28, H - MARG_B + 18), _fmt(v), fill=BLACK, font=TICK_FONT)
        draw.line((MARG_L - 10, y, MARG_L, y), fill=BLACK, width=2)
        draw.text((MARG_L - 90, y - 16), _fmt(v), fill=BLACK, font=TICK_FONT)
    _save(img, path)

# -------- 2) Time series (CNN) --------
def plot_time_series(t_sec, y_true, y_pred, split, path):
    t_min, t_max = min(t_sec), max(t_sec)
    y_min = min(min(y_true), min(y_pred))
    y_max = max(max(y_true), max(y_pred))
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_axes(draw, f"{split} - Time Series: Actual vs CNN Predicted", "Time", TARGET_COL, show_frame=True)
    def _polyline(vals, color, lw=4):
        if len(vals) < 2: return
        prev = None
        for t, y in zip(t_sec, vals):
            x = _scale(t, t_min, t_max, MARG_L, MARG_L + AX_W)
            yy = _scale(y, y_min, y_max, H - MARG_B, H - MARG_B - AX_H)
            if prev is not None:
                draw.line((*prev, x, yy), fill=color, width=lw)
            prev = (x, yy)
    _polyline(y_true, C0, lw=4)
    _polyline(y_pred, C1, lw=4)
    lx, ly, lw_box, lh_box = MARG_L + 12, MARG_T + 10, 340, 120
    draw.rectangle((lx, ly, lx + lw_box, ly + lh_box), outline=BLACK, width=2, fill=(255, 255, 255))
    draw.line((lx + 16, ly + 30, lx + 100, ly + 30), fill=C0, width=6); draw.text((lx + 116, ly + 12),  "Actual",        fill=BLACK, font=LEG_FONT)
    draw.line((lx + 16, ly + 72, lx + 100, ly + 72), fill=C1, width=6); draw.text((lx + 116, ly + 54), "CNN Predicted", fill=BLACK, font=LEG_FONT)
    for v in _nice_ticks(t_min, t_max, k=6):
        x = _scale(v, t_min, t_max, MARG_L, MARG_L + AX_W)
        draw.line((x, H - MARG_B, x, H - MARG_B + 10), fill=BLACK, width=2)
        try:
            s = datetime.fromtimestamp(v).strftime("%Y-%m-%d")
        except Exception:
            s = _fmt(v)
        draw.text((x - 58, H - MARG_B + 18), s, fill=BLACK, font=TICK_FONT)
    for v in _nice_ticks(y_min, y_max, k=6):
        y = _scale(v, y_min, y_max, H - MARG_B, H - MARG_B - AX_H)
        draw.line((MARG_L - 10, y, MARG_L, y), fill=BLACK, width=2)
        draw.text((MARG_L - 96, y - 16), _fmt(v), fill=BLACK, font=TICK_FONT)
    _save(img, path)

# -------- 3) Residuals histogram (CNN) --------
def plot_residuals_hist(y_true, y_pred, split, path, bins=40):
    resid = [float(b) - float(a) for a, b in zip(y_true, y_pred)]
    rmin, rmax = min(resid), max(resid)
    if rmax == rmin: rmax = rmin + 1.0
    bw = (rmax - rmin) / bins
    counts = [0] * bins
    for r in resid:
        idx = int((r - rmin) / bw)
        idx = 0 if idx < 0 else (bins - 1 if idx >= bins else idx)
        counts[idx] += 1
    cmax = max(counts) or 1
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_axes(draw, f"{split} - Residuals Histogram (CNN)", "Residual (Pred - Actual)", "Count", show_frame=True)
    for i, c in enumerate(counts):
        x0 = _scale(rmin + i * bw, rmin, rmax, MARG_L, MARG_L + AX_W)
        x1 = _scale(rmin + (i + 1) * bw, rmin, rmax, MARG_L, MARG_L + AX_W)
        y1 = H - MARG_B
        y0 = _scale(c, 0, cmax, H - MARG_B, H - MARG_B - AX_H)
        draw.rectangle((x0, y0, x1 - 1, y1), fill=C0)
    for v in _nice_ticks(rmin, rmax, k=6):
        x = _scale(v, rmin, rmax, MARG_L, MARG_L + AX_W)
        draw.line((x, H - MARG_B, x, H - MARG_B + 10), fill=BLACK, width=2)
        draw.text((x - 44, H - MARG_B + 18), _fmt(v), fill=BLACK, font=TICK_FONT)
    for v in _nice_ticks(0, cmax, k=6):
        y = _scale(v, 0, cmax, H - MARG_B, H - MARG_B - AX_H)
        draw.line((MARG_L - 10, y, MARG_L, y), fill=BLACK, width=2)
        draw.text((MARG_L - 96, y - 16), str(int(v)), fill=BLACK, font=TICK_FONT)
    _save(img, path)

# -------- 4) CNN vs baselines (RMSE from MSE) --------
def plot_bar_metric_comparison(split, methods, rmse, mae, path):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_axes(draw, f"{split} - CNN vs Baselines", "Method", "Score", show_frame=True)
    n = max(1, len(methods))
    group_span = AX_W / n
    inner_pad_ratio = 0.08
    bar_ratio       = 0.36
    gap_ratio       = 0.12
    vmax = max(max(rmse), max(mae), 1e-9)
    for i in range(n):
        gx0 = MARG_L + i * group_span
        inner_pad = group_span * inner_pad_ratio
        bar_w     = group_span * bar_ratio
        gap_w     = group_span * gap_ratio
        x0  = gx0 + inner_pad
        x1  = x0 + bar_w
        y1  = H - MARG_B
        y0  = _scale(rmse[i], 0, vmax, H - MARG_B, H - MARG_B - AX_H)
        draw.rectangle((x0, y0, x1, y1), fill=C0)
        x0b = x1 + gap_w
        x1b = x0b + bar_w
        y0b = _scale(mae[i], 0, vmax, H - MARG_B, H - MARG_B - AX_H)
        draw.rectangle((x0b, y0b, x1b, y1), fill=C1)
        lbl = methods[i][:22]
        lw, _ = _text_size(draw, lbl, TICK_FONT)
        cx = gx0 + group_span / 2
        draw.text((cx - lw / 2, H - MARG_B + 24), lbl, fill=BLACK, font=TICK_FONT)
    for v in _nice_ticks(0, vmax, k=6):
        y = _scale(v, 0, vmax, H - MARG_B, H - MARG_B - AX_H)
        draw.line((MARG_L - 10, y, MARG_L, y), fill=BLACK, width=2)
        draw.text((MARG_L - 96, y - 16), _fmt(v), fill=BLACK, font=TICK_FONT)
    lx, ly, lw_box, lh_box = W - 340, MARG_T + 10, 300, 130
    draw.rectangle((lx, ly, lx + lw_box, ly + lh_box), outline=BLACK, width=2, fill=(255, 255, 255))
    draw.rectangle((lx + 18, ly + 30, lx + 54, ly + 58), fill=C0); draw.text((lx + 72, ly + 24), "RMSE", fill=BLACK, font=LEG_FONT)
    draw.rectangle((lx + 18, ly + 78, lx + 54, ly + 106), fill=C1); draw.text((lx + 72, ly + 72), "MAE",  fill=BLACK, font=LEG_FONT)
    _save(img, path)

# -------- 5) Permutation importances (CNN) --------
def plot_perm_importance(features, importances, path, k=15):
    imp = importances.tolist() if hasattr(importances, "tolist") else list(importances)
    order = sorted(range(len(imp)), key=lambda j: imp[j], reverse=True)
    topk = order[:min(k, len(order))]
    names = [str(features[j]) for j in topk]
    vals  = [float(imp[j]) for j in topk]
    vmax  = max(vals) if vals else 1.0
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_axes(draw, "Top-15 Feature Importances (CNN)", "", "", show_frame=True)
    n = len(vals)
    if n == 0:
        _text_center(draw, W / 2, H / 2 - 8, "No features", TICK_FONT)
        _save(img, path); return
    bar_h = AX_H / (n * 1.2)
    for i, (name, val) in enumerate(zip(names, vals)):
        y_center = MARG_T + bar_h * (1.2 * i + 0.6)
        y0 = y_center - bar_h / 2
        y1 = y_center + bar_h / 2
        x0 = MARG_L
        x1 = _scale(val, 0, vmax, MARG_L, MARG_L + AX_W)
        draw.rectangle((x0, y0, x1, y1), fill=C0)
        draw.text((26, y0), name[:32], fill=BLACK, font=TICK_FONT)
        draw.text((x1 + 12, y0), _fmt(val), fill=BLACK, font=TICK_FONT)
    for v in _nice_ticks(0, vmax, k=6):
        x = _scale(v, 0, vmax, MARG_L, MARG_L + AX_W)
        draw.line((x, H - MARG_B, x, H - MARG_B + 10), fill=BLACK, width=2)
        draw.text((x - 28, H - MARG_B + 18), _fmt(v), fill=BLACK, font=TICK_FONT)
    _save(img, path)

# ===================== Create PNGs =====================
plot_pred_vs_actual(y_val_list,  y_val_pred_list,  "VAL",  os.path.join(SAVE_DIR, "val_pred_vs_actual.png"))
plot_pred_vs_actual(y_test_list, y_test_pred_list, "TEST", os.path.join(SAVE_DIR, "test_pred_vs_actual.png"))

plot_time_series(t_val_sec,  y_val_list,  y_val_pred_list,  "VAL",  os.path.join(SAVE_DIR, "val_timeseries.png"))
plot_time_series(t_test_sec, y_test_list, y_test_pred_list, "TEST", os.path.join(SAVE_DIR, "test_timeseries.png"))

plot_residuals_hist(y_val_list,  y_val_pred_list,  "VAL",  os.path.join(SAVE_DIR, "val_residuals_hist.png"))
plot_residuals_hist(y_test_list, y_test_pred_list, "TEST", os.path.join(SAVE_DIR, "test_residuals_hist.png"))

mse_v_model  = _mse(y_val_list,  y_val_pred_list);  rmse_v_model  = mse_v_model ** 0.5
mae_v_model  = _mae(y_val_list,  y_val_pred_list)
mse_v_mean   = _mse(y_val_list,  [_mean(y_val_list)]   * len(y_val_list));  rmse_v_mean   = mse_v_mean ** 0.5
mae_v_mean   = _mae(y_val_list,  [_mean(y_val_list)]   * len(y_val_list))
mse_v_median = _mse(y_val_list,  [_median(y_val_list)] * len(y_val_list));  rmse_v_median = mse_v_median ** 0.5
mae_v_median = _mae(y_val_list,  [_median(y_val_list)] * len(y_val_list))

mse_t_model  = _mse(y_test_list, y_test_pred_list);  rmse_t_model  = mse_t_model ** 0.5
mae_t_model  = _mae(y_test_list, y_test_pred_list)
mse_t_mean   = _mse(y_test_list, [_mean(y_test_list)]   * len(y_test_list));  rmse_t_mean   = mse_t_mean ** 0.5
mae_t_mean   = _mae(y_test_list, [_mean(y_test_list)]   * len(y_test_list))
mse_t_median = _mse(y_test_list, [_median(y_test_list)] * len(y_test_list));  rmse_t_median = mse_t_median ** 0.5
mae_t_median = _mae(y_test_list, [_median(y_test_list)] * len(y_test_list))

plot_bar_metric_comparison(
    "VAL",
    ["CNN", "Baseline-mean", "Baseline-median"],
    [rmse_v_model, rmse_v_mean, rmse_v_median],
    [mae_v_model,  mae_v_mean,  mae_v_median],
    os.path.join(SAVE_DIR, "val_model_vs_baselines.png")
)
plot_bar_metric_comparison(
    "TEST",
    ["CNN", "Baseline-mean", "Baseline-median"],
    [rmse_t_model, rmse_t_mean, rmse_t_median],
    [mae_t_model,  mae_t_mean,  mae_t_median],
    os.path.join(SAVE_DIR, "test_model_vs_baselines.png")
)

plot_perm_importance(features, importances, os.path.join(SAVE_DIR, "feature_importance_top15.png"))
# ====================== END Visualization ======================
