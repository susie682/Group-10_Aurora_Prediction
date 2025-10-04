# ============================================================
'''
Parse the "time" column into a time index and sort the data by time.
Perform outlier handling on the input features:
First, clip the data based on physical boundaries (e.g., humidity: 0–100, cloud cover: 0–1, precipitation: ≥0, Kp index: 0–9, ap index: ≥0).
Then, apply winsorizing clipping using the Interquartile Range (IQR) method (with a threshold of 1.5×IQR).
Process the three prediction targets (keogram_mean/median/max):
First, identify extreme outliers using the Median Absolute Deviation (MAD) method (robust z-score) and set these outliers to missing values.
Next, perform limited linear interpolation: only fill small gaps with no more than 2 consecutive missing points to avoid "over-filling".
Fill the remaining missing values with the mean value of the corresponding day.
If a day has completely missing data for the prediction targets, keep the missing values unchanged and do not fill them.

'''
# ============================================================

import csv
import math
from datetime import datetime, date

IN_PATH = "final-planb-24.csv"
OUT_PATH = "final-planb-24_preprocessed.csv"

TARGETS = ["keogram_mean", "keogram_median", "keogram_max"]

# ---------- Small helpers ----------

def sniff_delimiter(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        # Fallback: try tab, then comma
        return "\t" if "\t" in sample else ","

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

def try_parse_time(s):
    s = s.strip()
    # Try a few common formats
    fmts = ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # Best-effort: split and pad
    raise ValueError(f"Unrecognized time format: {s}")

def median(vals):
    a = [v for v in vals if v is not None and math.isfinite(v)]
    n = len(a)
    if n == 0: 
        return None
    a.sort()
    mid = n // 2
    if n % 2 == 1:
        return a[mid]
    return (a[mid - 1] + a[mid]) / 2.0

def quantile(vals, q):
    """Linear interpolation between order statistics, q in [0,1]."""
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
    weight = pos - lo
    return a[lo] * (1 - weight) + a[hi] * weight

def iqr_bounds(vals, k=1.5):
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    if q1 is None or q3 is None:
        return None, None
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr == 0:
        return None, None
    return (q1 - k * iqr, q3 + k * iqr)

def mad(vals):
    """Median Absolute Deviation (MAD)."""
    m = median(vals)
    if m is None:
        return None
    abs_dev = [abs(v - m) for v in vals if v is not None and math.isfinite(v)]
    if not abs_dev:
        return None
    return median(abs_dev)

def robust_outliers_mad(vals, thresh=6.0):
    """
    Return a boolean list marking outliers by robust z = 0.6745*(x - med)/MAD.
    If MAD is 0/None, returns all False (no outliers).
    """
    arr = [v if (v is not None and math.isfinite(v)) else None for v in vals]
    m = median(arr)
    md = mad(arr)
    flags = [False] * len(arr)
    if m is None or md in (None, 0) or not math.isfinite(md):
        return flags
    for i, v in enumerate(arr):
        if v is None:
            continue
        z = 0.6745 * (v - m) / md
        if abs(z) > thresh:
            flags[i] = True
    return flags

def same_day(d1: date, d2: date) -> bool:
    return d1 == d2

# ---------- Load CSV ----------

delimiter = sniff_delimiter(IN_PATH)

with open(IN_PATH, "r", newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f, delimiter=delimiter)
    header = rdr.fieldnames
    if header is None:
        raise ValueError("Empty CSV or failed to read header.")
    if "time" not in header:
        raise ValueError("Missing required 'time' column.")
    rows = []
    for row in rdr:
        rows.append(row)

# Column order to preserve on write-out
col_order = header[:]

# Parse time and convert numerics
for row in rows:
    row["time_parsed"] = try_parse_time(row["time"])
    for col in col_order:
        if col == "time":
            continue
        row[col] = to_float(row[col])

# Sort by time
rows.sort(key=lambda r: r["time_parsed"])

# Identify targets actually present and numeric feature columns
targets = [c for c in TARGETS if c in col_order]
feature_cols = [c for c in col_order if c not in ("time", "time_parsed") and c not in targets]

# ---------- Feature domain clipping ----------

def clip_domain_row(row):
    # humidity_* in [0, 100]
    for c in [c for c in feature_cols if c.startswith("humidity_")]:
        v = row[c]
        if v is not None:
            row[c] = min(100.0, max(0.0, v))
    # tcc_* in [0, 1]
    for c in [c for c in feature_cols if c.startswith("tcc_")]:
        v = row[c]
        if v is not None:
            row[c] = min(1.0, max(0.0, v))
    # tp_frac_gt_0.1 in [0, 1]
    if "tp_frac_gt_0.1" in feature_cols:
        v = row["tp_frac_gt_0.1"]
        if v is not None:
            row["tp_frac_gt_0.1"] = min(1.0, max(0.0, v))
    # tp_mm_* and tp_mm_mean_aw >= 0
    for c in [c for c in feature_cols if c.startswith("tp_mm_")] + (["tp_mm_mean_aw"] if "tp_mm_mean_aw" in feature_cols else []):
        v = row.get(c)
        if v is not None:
            row[c] = max(0.0, v)
    # Kp in [0, 9], ap >= 0
    if "Kp" in feature_cols and rows is not None:
        v = row.get("Kp")
        if v is not None:
            row["Kp"] = min(9.0, max(0.0, v))
    if "ap" in feature_cols:
        v = row.get("ap")
        if v is not None:
            row["ap"] = max(0.0, v)
    # Temperature left unchanged (negatives allowed)

for row in rows:
    clip_domain_row(row)

# ---------- Feature winsorization by IQR (k=1.5) ----------

def collect_column(col):
    return [r[col] for r in rows]

def apply_winsorize(col, k=1.5):
    vals = collect_column(col)
    lo, hi = iqr_bounds(vals, k=k)
    if lo is None or hi is None:
        return  # nothing to do
    for r in rows:
        v = r[col]
        if v is None:
            continue
        if v < lo:
            r[col] = lo
        elif v > hi:
            r[col] = hi

for c in feature_cols:
    apply_winsorize(c, k=1.5)

# ---------- Targets: outlier removal via MAD ----------

for t in targets:
    vals = collect_column(t)
    flags = robust_outliers_mad(vals, thresh=6.0)
    for i, flag in enumerate(flags):
        if flag:
            rows[i][t] = None  # set extreme targets to missing

# ---------- Targets: limited interpolation (≤2 gap) ----------

def interpolate_small_gaps(col, max_gap=2):
    # Assumes (approximately) regular hourly intervals
    series = [r[col] for r in rows]
    n = len(series)

    # Helper to set a block of Nones between i_prev and i_next if gap <= max_gap
    i = 0
    while i < n:
        # Skip existing valid values
        if series[i] is not None:
            i += 1
            continue
        # Start of a NaN block
        start = i
        while i < n and series[i] is None:
            i += 1
        end = i  # [start, end) is the missing block
        gap_len = end - start

        if gap_len <= max_gap:
            prev_idx = start - 1
            next_idx = end
            prev_val = series[prev_idx] if prev_idx >= 0 else None
            next_val = series[next_idx] if next_idx < n else None

            if prev_val is not None and next_val is not None:
                # Linear interpolation across the block
                steps = gap_len + 1  # total segments between prev and next
                for k in range(gap_len):
                    frac = (k + 1) / steps
                    fill = prev_val * (1 - frac) + next_val * frac
                    series[start + k] = fill
            elif prev_val is not None and next_val is None:
                # Edge forward-fill for small gap at the end
                for k in range(gap_len):
                    series[start + k] = prev_val
            elif prev_val is None and next_val is not None:
                # Edge backward-fill for small gap at the start
                for k in range(gap_len):
                    series[start + k] = next_val
            # else both None: cannot fill (should be rare for such a short block)

    # Write back
    for idx, v in enumerate(series):
        rows[idx][col] = v

for t in targets:
    interpolate_small_gaps(t, max_gap=2)

# ---------- Targets: same-day mean fallback (keep full-day-missing as None) ----------

def fill_day_mean(col):
    # Compute mean per day using available values
    # day -> (sum, count)
    sums = {}
    counts = {}
    for r in rows:
        d = r["time_parsed"].date()
        v = r[col]
        if v is not None and math.isfinite(v):
            sums[d] = sums.get(d, 0.0) + v
            counts[d] = counts.get(d, 0) + 1

    means = {d: (sums[d] / counts[d]) for d in counts if counts[d] > 0}

    # Fill remaining Nones with that day's mean (if present)
    for r in rows:
        if r[col] is None:
            d = r["time_parsed"].date()
            if d in means:
                r[col] = means[d]
            # if day has no mean (full-day-missing), leave as None

for t in targets:
    fill_day_mean(t)

# ---------- Write out CSV ----------

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=delimiter)
    # Preserve the original header order; ensure we drop internal helper column
    out_cols = [c for c in col_order if c != "time_parsed"]
    w.writerow(out_cols)
    for r in rows:
        out = []
        for c in out_cols:
            if c == "time":
                # keep original textual time
                out.append(r[c])
            else:
                v = r.get(c, None)
                out.append("" if v is None else f"{v}")
        w.writerow(out)

# ---------- Simple report ----------
def count_nans(col):
    vals = [r[col] for r in rows]
    return sum(1 for v in vals if (v is None or (isinstance(v, float) and not math.isfinite(v))))

print(f"Saved: {OUT_PATH}")
for t in targets:
    print(f"{t}: remaining missing = {count_nans(t)}")
