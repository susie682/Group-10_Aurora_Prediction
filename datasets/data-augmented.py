# ============================================================
'''
Read the original CSV file.
Filtering: Retain only the rows where the three columns (keogram_mean/keogram_median/keogram_max) have no missing values.
On the basis of the filtered data, perform three types of data augmentation and merge the augmented data with the original data for output:
Jitter: Add small-magnitude noise based on the characteristics of each column.
Mixup: Conduct linear mixing of data at adjacent time points within the same "night".
Extreme event oversampling: Apply Jitter again to the samples in the high quantile range of keogram_max.
Maintain physical boundaries:
Humidity: [0, 100]
Cloud cover: [0, 1]
Precipitation: ≥ 0
Kp index: ∈ [0, 9] (quantized at 1/3 intervals)
ap index: ≥ 0, etc.
For uniqueness: The "time" of augmented samples will be fine-tuned by ±5 minutes around the original time.
Output file: final-planb-24_augmented.csv (includes filtered original samples + augmented samples)
'''
# ============================================================
import csv
import math
import random
from datetime import datetime, timedelta

# --------------------- Config ---------------------
IN_PATH  = "final-planb-24.csv"
OUT_PATH = "final-planb-24_augmented.csv"

# How much to augment:
JITTER_RATE  = 0.50   # generate jittered copy for ~50% of base rows
MIXUP_RATE   = 0.30   # generate mixup samples ≈ 30% of base size
EXTREME_PCTL = 0.90   # top quantile for keogram_max to oversample
EXTREME_MULT = 1      # number of extra jitter copies per extreme row

# Noise strengths (feature-aware)
NOISE_SCALE_DEFAULT = 0.15  # fallback fraction of robust IQR scale
HUMIDITY_SIGMA      = 2.0   # absolute jitter for humidity_*
TCC_SIGMA           = 0.02  # absolute jitter for tcc_*
TEMP_SIGMA          = 0.3   # absolute jitter for temp_*
KP_SIGMA            = 1/3   # jitter for Kp (will be quantized to 1/3)
AP_SIGMA            = 3.0   # absolute jitter for ap
PRECIP_MULT_JITTER  = 0.10  # multiplicative jitter for tp_mm_* if > 0

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGETS = ["keogram_mean", "keogram_median", "keogram_max"]

# --------------------- Utils ---------------------
def sniff_delimiter(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
    if "\t" in sample and "," not in sample.splitlines()[0]:
        return "\t"
    return ","  # default

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_time(s):
    s = s.strip()
    fmts = ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    raise ValueError(f"Unrecognized time format: {s}")

def fmt_time(dt):
    # Match "YYYY/M/D H:MM" (no zero-padding on Y/M/D/H, but minute is padded)
    return f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:{dt.minute:02d}"

def quantile(vals, q):
    a = [v for v in vals if v is not None and math.isfinite(v)]
    n = len(a)
    if n == 0:
        return None
    a.sort()
    if n == 1:
        return a[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return a[lo]
    w = pos - lo
    return a[lo] * (1 - w) + a[hi] * w

def iqr(vals):
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1

def night_id(dt):
    # Assign samples to a "night" bucket: hours < 12 belong to previous date
    d = dt.date()
    if dt.hour < 12:
        return (d - timedelta(days=1)).isoformat()
    return d.isoformat()

def clamp_physical(col, v):
    if v is None:
        return None
    # humidity_* in [0, 100]
    if col.startswith("humidity_"):
        return min(100.0, max(0.0, v))
    # tcc_* in [0, 1]
    if col.startswith("tcc_"):
        return min(1.0, max(0.0, v))
    # precip >= 0
    if col.startswith("tp_mm_") or col == "tp_mm_mean_aw":
        return max(0.0, v)
    # fraction in [0,1]
    if col == "tp_frac_gt_0.1":
        return min(1.0, max(0.0, v))
    # Kp in [0,9], quantized to thirds
    if col == "Kp":
        v = min(9.0, max(0.0, v))
        # quantize to 1/3 steps: round to nearest multiple of 1/3
        step = 1.0/3.0
        return round(v / step) * step
    # ap >= 0 (integer)
    if col == "ap":
        return float(max(0, int(round(v))))
    # temps can be negative; leave as is
    return v

def copy_row(row):
    return {k: v for k, v in row.items()}

# --------------------- Load & filter ---------------------
delimiter = sniff_delimiter(IN_PATH)

with open(IN_PATH, "r", encoding="utf-8", newline="") as f:
    rdr = csv.DictReader(f, delimiter=delimiter)
    cols = rdr.fieldnames
    if cols is None:
        raise ValueError("Empty CSV.")
    if "time" not in cols:
        raise ValueError("Missing 'time' column.")
    rows = []
    for r in rdr:
        rows.append(r)

# Convert types and sort by time
for r in rows:
    r["_time"] = parse_time(r["time"])
    for c in cols:
        if c == "time":
            continue
        r[c] = to_float(r[c])

rows.sort(key=lambda x: x["_time"])

# Keep only rows where all keogram targets exist
targets_present = [t for t in TARGETS if t in cols]
if len(targets_present) < 3:
    raise ValueError("The CSV must have keogram_mean, keogram_median, keogram_max.")

base = []
for r in rows:
    ok = True
    for t in targets_present:
        v = r[t]
        if v is None or not math.isfinite(v):
            ok = False
            break
    if ok:
        base.append(r)

# Identify feature columns (exclude time and targets)
feature_cols = [c for c in cols if c not in ("time",) + tuple(targets_present)]

# --------------------- Precompute robust scales ---------------------
# Use IQR as robust spread; fallback to per-feature defaults
col_iqr = {}
for c in feature_cols + targets_present:
    vals = [r[c] for r in base]
    col_iqr[c] = iqr(vals)

# --------------------- Jitter (feature-aware) ---------------------
def jitter_row(r):
    nr = copy_row(r)
    for c in feature_cols + targets_present:
        v = nr[c]
        if v is None or not math.isfinite(v):
            continue

        # Column-specific jitter rules
        if c.startswith("humidity_"):
            v_new = v + random.gauss(0.0, HUMIDITY_SIGMA)
        elif c.startswith("tcc_"):
            v_new = v + random.gauss(0.0, TCC_SIGMA)
        elif c.startswith("temp_"):
            v_new = v + random.gauss(0.0, TEMP_SIGMA)
        elif c.startswith("tp_mm_") or c == "tp_mm_mean_aw":
            if v > 0:
                # multiplicative noise: v * (1 + eps), eps ~ N(0, PRECIP_MULT_JITTER)
                eps = random.gauss(0.0, PRECIP_MULT_JITTER)
                v_new = v * (1.0 + eps)
            else:
                # keep most zeros, add tiny drizzle with small probability
                if random.random() < 0.05:
                    v_new = 0.01
                else:
                    v_new = 0.0
        elif c == "Kp":
            v_new = v + random.gauss(0.0, KP_SIGMA)
        elif c == "ap":
            v_new = v + random.gauss(0.0, AP_SIGMA)
        else:
            # default: use IQR-based sigma
            spread = col_iqr.get(c, None)
            sigma = NOISE_SCALE_DEFAULT * (spread if (spread is not None and spread > 0) else 1.0)
            v_new = v + random.gauss(0.0, sigma)

        nr[c] = clamp_physical(c, v_new)

    # small time shift to avoid exact duplicates
    shift = random.randint(-5, 5)
    if shift == 0:
        shift = 1
    new_t = r["_time"] + timedelta(minutes=shift)
    nr["_time"] = new_t
    nr["time"]  = fmt_time(new_t)
    return nr

# --------------------- Mixup (within-night, nearby timestamps) ---------------------
# Build night buckets (index lists)
night_to_idx = {}
for idx, r in enumerate(base):
    nid = night_id(r["_time"])
    night_to_idx.setdefault(nid, []).append(idx)

def mix_rows(a, b, lam):
    nr = {}
    # time: mid-point + jitter
    mid = a["_time"] + (b["_time"] - a["_time"]) / 2
    nr["_time"] = mid + timedelta(minutes=random.randint(-5, 5))
    nr["time"]  = fmt_time(nr["_time"])
    # blend numeric cols
    for c in cols:
        if c == "time":
            continue
        va = a[c]; vb = b[c]
        if va is None and vb is None:
            nr[c] = None
        elif va is None:
            nr[c] = vb
        elif vb is None:
            nr[c] = va
        else:
            nr[c] = lam * va + (1 - lam) * vb
        nr[c] = clamp_physical(c, nr[c])
    return nr

def mixup_samples(target_count):
    out = []
    # Try to generate ~target_count mixup examples
    attempts = 0
    max_attempts = target_count * 10 + 100
    while len(out) < target_count and attempts < max_attempts:
        attempts += 1
        # pick a night with at least 2 samples
        nid = random.choice([k for k, v in night_to_idx.items() if len(v) >= 2])
        idxs = night_to_idx[nid]
        i = random.choice(idxs)
        # choose a neighbor within the same night, prefer nearby in time
        j_candidates = [j for j in idxs if j != i and abs(j - i) <= 2]
        if not j_candidates:
            j_candidates = [j for j in idxs if j != i]
        j = random.choice(j_candidates)
        a, b = base[i], base[j]
        lam = random.uniform(0.35, 0.65)
        out.append(mix_rows(a, b, lam))
    return out

# --------------------- Extreme oversampling ---------------------
kmax_vals = [r["keogram_max"] for r in base]
kmax_p90  = quantile(kmax_vals, EXTREME_PCTL)

def is_extreme(r):
    return r["keogram_max"] is not None and kmax_p90 is not None and r["keogram_max"] >= kmax_p90

# --------------------- Build augmented set ---------------------
augmented = []

# 1) Jitter a subset of base rows
for r in base:
    if random.random() < JITTER_RATE:
        augmented.append(jitter_row(r))

# 2) Mixup samples
mix_count = int(len(base) * MIXUP_RATE)
augmented.extend(mixup_samples(mix_count))

# 3) Oversample extremes by additional jitter copies
if kmax_p90 is not None and EXTREME_MULT > 0:
    for r in base:
        if is_extreme(r):
            for _ in range(EXTREME_MULT):
                augmented.append(jitter_row(r))

# --------------------- Write output ---------------------
# We will write: filtered base rows + augmented rows
out_rows = base + augmented

with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter=delimiter)
    w.writerow(cols)  # original header
    for r in out_rows:
        row_out = []
        for c in cols:
            if c == "time":
                row_out.append(r["time"])
            else:
                v = r.get(c, None)
                row_out.append("" if v is None else f"{v}")
        w.writerow(row_out)

print(f"Saved: {OUT_PATH}")
print(f"Base (filtered) rows: {len(base)}")
print(f"Augmented rows:      {len(augmented)}")
print(f"Total rows written:  {len(out_rows)}")
if kmax_p90 is not None:
    print(f"keogram_max 90th percentile: {kmax_p90:.3f}")
