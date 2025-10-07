#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

OMNI_TXT = Path("omni2_extra_data.txt")      # <-- your screenshot file
KEOGRAM_CSV = Path("final-planb-weighted.csv")
OUT_OMNI    = Path("omni_parsed.csv")
OUT_MERGED  = Path("merged_keogram_omni.csv")

# ---- helpers ----
def normalize_keogram_time(df: pd.DataFrame) -> pd.DataFrame:
    # finds a time-ish column and normalizes it to hourly
    for c in ["time","Time","timestamp","Timestamp","date","datetime","Date","Datetime"]:
        if c in df.columns:
            col = c
            break
    else:
        raise ValueError(f"No time-like column found in {list(df.columns)}")
    ts = pd.to_datetime(df[col].astype(str).str.replace(r"\.", ":", regex=True), errors="coerce")
    if ts.isna().any():
        ts = pd.to_datetime(df[col], errors="coerce")
    if ts.isna().any():
        raise ValueError("Could not parse some keogram timestamps.")
    out = df.copy()
    out["timestamp"] = ts.dt.floor("H")
    out["timestamp_str"] = month_day_str(out["timestamp"])
    out["month"] = out["timestamp"].dt.month
    out["day"]   = out["timestamp"].dt.day
    out["hour"]  = out["timestamp"].dt.hour
    return out
import re
def doy_to_timestamp(year, doy, hour):
    base = pd.to_datetime(year.astype(int).astype(str), format="%Y", errors="raise")
    return base + pd.to_timedelta(doy.astype(int) - 1, unit="D") \
                + pd.to_timedelta(hour.astype(int),       unit="h")

def month_day_str(ts: pd.Series) -> pd.Series:
    return (ts.dt.year.astype(str) + "/" +
            ts.dt.month.astype(str) + "/" +
            ts.dt.day.astype(str) + " " +
            ts.dt.hour.astype(str) + ":" +
            ts.dt.minute.astype(str).str.zfill(2))

# ---- parse this exact OMNI layout ----
def read_omni_whitespace(path: Path) -> pd.DataFrame:
    # Column order matches what you showed in the screenshot
    names = [
        "YEAR","DOY","Hour",
        "ScalarB_nT","By_GSM_nT","Bz_GSM_nT",
        "SW_Temp_K","Proton_Density_ncc","SW_Speed_kms",
        "Flow_Pressure_nPa","ap_index_nT","AE_index_nT"
    ]
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, names=names)
    # build timestamp
    df["timestamp"] = doy_to_timestamp(df["YEAR"], df["DOY"], df["Hour"])
    df["month"] = df["timestamp"].dt.month
    df["day"]   = df["timestamp"].dt.day
    df["hour"]  = df["timestamp"].dt.hour
    df["minute"]= df["timestamp"].dt.minute
    df["timestamp_str"] = month_day_str(df["timestamp"])
    # tidy order
    keep = [
        "timestamp","timestamp_str","YEAR","DOY","month","day","hour",
        "ScalarB_nT","By_GSM_nT","Bz_GSM_nT",
        "SW_Temp_K","Proton_Density_ncc","SW_Speed_kms",
        "Flow_Pressure_nPa","ap_index_nT","AE_index_nT"
    ]
    return df[keep]


TIME_COL_PATTERN = re.compile(
    r'^(timestamp(_str)?(_omni)?|time|Time|date|Date|datetime|Datetime|'
    r'YEAR|DOY|year(_omni)?|month(_omni)?|day(_omni)?|hour(_omni)?|minute(_omni)?)$'
)

def drop_time_like_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not TIME_COL_PATTERN.match(c)]
    return df[keep]

OMNI_TIME_COLS_PATTERN = re.compile(
    r'^(timestamp(_str)?|YEAR|DOY|year(_omni)?|month(_omni)?|day(_omni)?|hour(_omni)?|minute(_omni)?)$',
    re.IGNORECASE
)

def drop_time_like_from_omni(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if OMNI_TIME_COLS_PATTERN.match(c)]
    return df.drop(columns=cols_to_drop, errors="ignore")

def main():
    OMNI_TXT = Path("omni2_extra_data.txt")       # your OMNI data file
    KEOGRAM_CSV = Path("final-planb-weighted.csv")
    OUT_OMNI    = Path("omni_parsed.csv")
    OUT_MERGED  = Path("merged_keogram_omni.csv")

    # 1) Parse OMNI (includes timestamp for merging)
    omni_full = read_omni_whitespace(OMNI_TXT)
    omni_full.to_csv(OUT_OMNI, index=False)  # keep full parsed OMNI for reference

    # 2) Prepare a version of OMNI for merging with NO time-like columns
    #    (but we keep 'timestamp' temporarily for the join)
    omni_for_merge = omni_full.copy()
    # Keep 'timestamp' just for merge key, drop all other time-ish columns first
    keep_timestamp = omni_for_merge["timestamp"]
    omni_for_merge = drop_time_like_from_omni(omni_for_merge)
    omni_for_merge.insert(0, "timestamp", keep_timestamp)

    # 3) Load keogram. DO NOT modify/rename/remove its original time columns.
    kdf = pd.read_csv(KEOGRAM_CSV)

    # Internal merge key from keogram time (create and later drop)
    # Try to parse a time-ish column without altering the original columns
    time_col = next((c for c in ["time","Time","timestamp","Timestamp","date","Date","datetime","Datetime"] if c in kdf.columns), None)
    if time_col is None:
        raise ValueError(f"No time-like column in keogram CSV: {list(kdf.columns)}")

    merge_key = pd.to_datetime(kdf[time_col].astype(str).str.replace(r"\.", ":", regex=True), errors="coerce")
    if merge_key.isna().any():
        merge_key2 = pd.to_datetime(kdf[time_col], errors="coerce")
        merge_key = merge_key.fillna(merge_key2)
    if merge_key.isna().any():
        bad = kdf.loc[merge_key.isna(), time_col].head(5).tolist()
        raise ValueError(f"Failed to parse some keogram timestamps, examples: {bad}")

    kdf["_merge_key_hour"] = merge_key.dt.floor("H")

    # 4) Merge: keogram (left) on its _merge_key_hour  vs OMNI 'timestamp'
    merged = pd.merge(
        kdf,
        omni_for_merge,
        left_on="_merge_key_hour",
        right_on="timestamp",
        how="left",
        suffixes=("", "_omni")
    )

    # 5) Drop temporary keys and ANY remaining OMNI time-like columns (including 'timestamp')
    merged = merged.drop(columns=["_merge_key_hour", "timestamp"], errors="ignore")

    # IMPORTANT: we did NOT touch keogram’s own time columns — they remain unchanged.

    merged.to_csv(OUT_MERGED, index=False)
    print(f"Wrote {OUT_OMNI}  ({len(omni_full)} rows)")
    print(f"Wrote {OUT_MERGED} ({len(merged)} rows)")

if __name__ == "__main__":
    main()