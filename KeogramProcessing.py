#!/usr/bin/env python3
# batch_split_keograms.py

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import csv
import re
from pathlib import Path


# ---------- reuse your helper functions ----------
def convert_image_to_RGBarray(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    return arr, H, W

def convert_RGB_to_grayscale(image_array):
    return (0.299 * image_array[:, :, 0] +
            0.587 * image_array[:, :, 1] +
            0.114 * image_array[:, :, 2])

from matplotlib import colors as mcolors

# Detect "red" robustly in HSV (tunable)
def make_red_mask_hsv(rgb,
                      # red hue ranges (wrap around 0)
                      red_h1=(0.00, 0.05),   # 0°–18° in [0,1]
                      red_h2=(0.92, 1.00),   # 330°–360°
                      s_min=0.30,            # min saturation
                      v_min=0.20):           # min value
    arr = rgb.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(arr)  # (H, W, 3) in [0,1]
    Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
    is_red = (
        ((Hh >= red_h1[0]) & (Hh <= red_h1[1])) |
        ((Hh >= red_h2[0]) & (Hh <= red_h2[1]))
    ) & (Ss >= s_min) & (Vv >= v_min)
    return is_red

def valid_mask_white_purple(rgb,
                            white_v_min=0.95, white_s_max=0.20,
                            magenta_h1=(0.83, 1.00), magenta_h2=(0.00, 0.07),
                            magenta_s_min=0.35, magenta_v_min=0.25):
    arr = rgb.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(arr)
    Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
    white = (Vv >= white_v_min) & (Ss <= white_s_max)
    magenta = (
        ((Hh >= magenta_h1[0]) & (Hh <= magenta_h1[1])) |
        ((Hh >= magenta_h2[0]) & (Hh <= magenta_h2[1]))
    ) & (Ss >= magenta_s_min) & (Vv >= magenta_v_min)
    valid = ~(white | magenta)
    return valid

def hour_to_col(h, width):
    return int(np.floor((h / 24.0) * width))

def split_into_sections(rgb, n_sections=8):
    H, W = rgb.shape[:2]
    edges = [hour_to_col(24 * i / n_sections, W) for i in range(n_sections)]
    edges.append(W)

    slices, col_ranges, hour_ranges = [], [], []
    for i in range(n_sections):
        c0, c1 = edges[i], edges[i + 1]
        section = rgb[:, c0:c1, :]
        slices.append(section)
        col_ranges.append((c0, c1))
        hour_ranges.append((24 * i / n_sections, 24 * (i + 1) / n_sections))
    return slices, col_ranges, hour_ranges

def plot_sections(slices, hour_ranges, nrows=2, ncols=4, figsize=(16, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()
    for i, (ax, section) in enumerate(zip(axes, slices)):
        ax.imshow(section)
        ax.axis("off")
        sh, eh = hour_ranges[i]
        ax.set_title(f"{int(sh)}–{int(eh)} h")
    plt.tight_layout()
    plt.show()

def plot_gray_sections(slices, hour_ranges, nrows=2, ncols=4, figsize=(16, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()
    for i, (ax, section) in enumerate(zip(axes, slices)):
        gray = convert_RGB_to_grayscale(section)
        ax.imshow(gray, cmap="gray", aspect="auto")
        ax.axis("off")
        sh, eh = hour_ranges[i]
        ax.set_title(f"{int(sh)}–{int(eh)} h (gray)")
    plt.tight_layout()
    plt.show()

def print_intensity_stats_by_block(rgb, n_sections=8):
    gray = convert_RGB_to_grayscale(rgb)
    intensity_per_col = gray.mean(axis=0)
    W = gray.shape[1]
    for i in range(n_sections):
        start_h = 24 * i / n_sections
        end_h   = 24 * (i + 1) / n_sections
        c0 = hour_to_col(start_h, W)
        c1 = hour_to_col(end_h,   W)
        vals = intensity_per_col[c0:c1]
        print(f"{int(start_h):02d}–{int(end_h):02d}h "
              f"(cols {c0}–{c1-1}): mean={vals.mean():.2f}, median={np.median(vals):.2f}")

def save_sections(slices, out_dir, base_name="slice"):
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for i, section in enumerate(slices, 1):
        p = os.path.join(out_dir, f"{base_name}_{i:02d}.png")
        Image.fromarray(section).save(p)
        out_paths.append(p)
    return out_paths

#below functions are for the csv
def _extract_yyyymmdd(stem: str) -> str:
    """Return first 8-digit date (YYYYMMDD) from a filename stem."""
    m = re.search(r"\d{8}", stem)
    if not m:
        raise ValueError(f"Cannot find YYYYMMDD in '{stem}'")
    return m.group(0)

def save_grid(slices, hour_ranges, out_path, nrows=2, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6))
    axes = axes.ravel()
    for i, (ax, section) in enumerate(zip(axes, slices)):
        ax.imshow(section)
        ax.axis("off")
        sh, eh = hour_ranges[i]
        ax.set_title(f"{int(sh)}–{int(eh)} h")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def write_segment_stats_csv(input_dir="keogram-out2",
                            output_csv="keogram_segment_stats.csv",
                            n_sections=8):
    """
    For every *inpaint.png (and *inpant.png) under input_dir:
      - compute mean/median/max for each of 8 segments,
      - append one CSV row per segment: YYYY-MM-DD-h1:h2, mean, median, max
    """
    in_dir = Path(input_dir)
    files = sorted(set(in_dir.glob("**/*inpaint.png")) |
                   set(in_dir.glob("**/*inpant.png")))
    if not files:
        print(f"[warn] No inpainted PNGs found under {in_dir}")
        return

    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "mean", "median", "max"])  # header

        for p in files:
            rgb, H, W = convert_image_to_RGBarray(p)
            gray = convert_RGB_to_grayscale(rgb)
            yyyymmdd = _extract_yyyymmdd(p.stem)
            date_fmt = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

            for i in range(n_sections):
                start_h = int(24 * i / n_sections)
                end_h   = int(24 * (i + 1) / n_sections)
                c0 = hour_to_col(start_h, W)
                c1 = hour_to_col(end_h,   W)

                seg_vals = gray[:, c0:c1].ravel()
                mean_v   = float(seg_vals.mean())
                median_v = float(np.median(seg_vals))
                max_v    = float(seg_vals.max())

                seg_label = f"{date_fmt}-{start_h}:{end_h}"
                w.writerow([seg_label, f"{mean_v:.2f}", f"{median_v:.2f}", f"{max_v:.2f}"])

            print(f"[ok] {p.name} -> 8 rows")

    print(f"[done] Wrote segment stats to {output_csv}")

def write_segment_stats_csv_with_red_filter(
        input_dir="keogram-out2",
        output_csv="keogram_segment_stats.csv",
        n_sections=8,
        red_ratio_threshold=0.50,
        require_min_valid_frac=0.30,
        use_invalid_mask=True
    ):
    """
    For every *inpaint.png (and *inpant.png) under input_dir:
      - for each segment:
          * compute valid mask (optional)
          * compute red_ratio = (# red & valid) / (# valid)
          * add sunlight_contamination column (1 if red_ratio > threshold, else 0)
          * always write row with: segment, mean, median, max, sunlight_contamination
    """
    in_dir = Path(input_dir)
    files = sorted(set(in_dir.glob("**/*inpaint.png")) |
                   set(in_dir.glob("**/*inpant.png")))
    if not files:
        print(f"[warn] No inpainted PNGs found under {in_dir}")
        return

    with open(output_csv, "w", newline="") as f_out:
        out_w = csv.writer(f_out)
        out_w.writerow(["segment", "mean", "median", "max", "sunlight_contamination"])

        for p in files:
            rgb, H, W = convert_image_to_RGBarray(p)
            gray = convert_RGB_to_grayscale(rgb)
            yyyymmdd = _extract_yyyymmdd(p.stem)
            date_fmt = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

            for i in range(n_sections):
                start_h = int(24 * i / n_sections)
                end_h   = int(24 * (i + 1) / n_sections)
                c0 = hour_to_col(start_h, W)
                c1 = hour_to_col(end_h,   W)

                seg_rgb  = rgb[:, c0:c1, :]
                seg_gray = gray[:, c0:c1]

                # valid pixels + red ratio
                valid = valid_mask_white_purple(seg_rgb) if use_invalid_mask else np.ones(seg_gray.shape, bool)
                valid_count = int(valid.sum())
                valid_frac  = valid_count / (seg_gray.size) if seg_gray.size > 0 else 0

                # determine sunlight contamination
                sunlight_contamination = 0
                
                if valid_frac < require_min_valid_frac:
                    sunlight_contamination = 1  # mark as contaminated if not enough valid pixels
                else:
                    red_mask = make_red_mask_hsv(seg_rgb)
                    red_ratio = (red_mask & valid).sum() / valid_count if valid_count > 0 else 0
                    if red_ratio > red_ratio_threshold:
                        sunlight_contamination = 1

                # compute stats on valid pixels
                seg_vals = seg_gray[valid].astype(float) if valid_count > 0 else np.array([0.0])
                mean_v   = float(seg_vals.mean()) if len(seg_vals) > 0 else 0.0
                median_v = float(np.median(seg_vals)) if len(seg_vals) > 0 else 0.0
                max_v    = float(seg_vals.max()) if len(seg_vals) > 0 else 0.0

                seg_label = f"{date_fmt}-{start_h}"
                out_w.writerow([seg_label, f"{mean_v:.2f}", f"{median_v:.2f}", f"{max_v:.2f}", sunlight_contamination])

            print(f"[ok] {p.name} -> {n_sections} rows written")

    print(f"[done] Stats with sunlight_contamination flag -> {output_csv}")

def process_multiple_years(
        base_dir=".",
        year_range=None,
        n_sections=24,
        red_ratio_threshold=0.50,
        require_min_valid_frac=0.30,
        use_invalid_mask=True
    ):
    """
    Process multiple years of keogram data automatically.
    
    Args:
        base_dir: Base directory containing year folders
        year_range: Tuple of (start_year, end_year) inclusive, or None to auto-detect
        n_sections: Number of time segments per day
        red_ratio_threshold: Threshold for red pixel ratio (0.50 = 50%)
        require_min_valid_frac: Minimum fraction of valid pixels required
        use_invalid_mask: Whether to use white/purple masking
    """
    base_path = Path(base_dir)
    
    # Auto-detect year directories if not specified
    if year_range is None:
        year_dirs = []
        for item in base_path.iterdir():
            if item.is_dir():
                # Look for directories with 'keogram-out' followed by 4 digits
                match = re.search(r'keogram-out(\d{4})', item.name)
                if match:
                    year_dirs.append((int(match.group(1)), item))
        year_dirs.sort()
    else:
        start_year, end_year = year_range
        year_dirs = []
        for year in range(start_year, end_year + 1):
            input_dir = base_path / f"keogram-out{year}"
            if input_dir.exists():
                year_dirs.append((year, input_dir))
    
    if not year_dirs:
        print(f"[warn] No year directories found in {base_path}")
        return
    
    print(f"[info] Found {len(year_dirs)} year directories to process")
    
    for year, input_dir in year_dirs:
        output_csv = base_path / f"keogram_segment_stats{year}_{n_sections}hours_weighted.csv"
        
        print(f"\n{'='*60}")
        print(f"[info] Processing year {year}")
        print(f"[info] Input: {input_dir}")
        print(f"[info] Output: {output_csv}")
        print(f"{'='*60}")
        
        write_segment_stats_csv_with_red_filter(
            input_dir=str(input_dir),
            output_csv=str(output_csv),
            n_sections=n_sections,
            red_ratio_threshold=red_ratio_threshold,
            require_min_valid_frac=require_min_valid_frac,
            use_invalid_mask=use_invalid_mask
        )
    
    print(f"\n{'='*60}")
    print(f"[done] Processed all {len(year_dirs)} years")
    print(f"{'='*60}")

def process_all_with_csv(input_dir="keogram-out2",
                         output_dir="segments-out",
                         output_csv="keogram_segment_stats.csv",
                         n_sections=8):
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(set(in_dir.glob("**/*inpaint.png")) |
                   set(in_dir.glob("**/*inpant.png")))
    if not files:
        print(f"[warn] No inpainted PNGs found under {in_dir}")
        return

    # open CSV once; append rows as we go
    with open(output_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["segment", "mean", "median", "max"])

        for p in files:
            print(f"[info] {p.name}")
            rgb, H, W = convert_image_to_RGBarray(p)
            slices, col_ranges, hour_ranges = split_into_sections(rgb, n_sections=n_sections)

            # save slices
            stem = p.stem
            per_img_out = out_root / stem
            out_paths = save_sections(slices, per_img_out, base_name=stem)

            # save grid preview (no display)
            save_grid(slices, hour_ranges, per_img_out / f"{stem}_grid.png")

            # per-segment stats -> CSV rows
            gray = convert_RGB_to_grayscale(rgb)
            for i in range(n_sections):
                start_h = int(24 * i / n_sections)
                end_h   = int(24 * (i + 1) / n_sections)
                c0 = hour_to_col(start_h, W)
                c1 = hour_to_col(end_h,   W)
                seg_vals = gray[:, c0:c1].ravel()
                mean_v   = float(seg_vals.mean())
                median_v = float(np.median(seg_vals))
                max_v    = float(seg_vals.max())

                yyyymmdd = _extract_yyyymmdd(stem)
                date_fmt = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
                seg_label = f"{date_fmt}-{start_h}:{end_h}"
                w.writerow([seg_label, f"{mean_v:.2f}", f"{median_v:.2f}", f"{max_v:.2f}"])

            print(f"[ok] Saved {len(out_paths)} slices + grid, wrote CSV rows for {p.name}")

    print(f"[done] All files processed. CSV -> {output_csv}, slices -> {output_dir}")

# ---------- batch driver ----------
def process_all(input_dir="keogram-out2", output_dir="segments-out", n_sections=8):
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = sorted(set(in_dir.glob("**/*inpaint.png")) |
                        set(in_dir.glob("**/*inpant.png")))

    if not candidates:
        print(f"[warn] No matching files in {in_dir}")
        return

    for img_path in candidates:
        print(f"[info] Processing {img_path.name}")
        rgb, H, W = convert_image_to_RGBarray(img_path)
        slices, col_ranges, hour_ranges = split_into_sections(rgb, n_sections=n_sections)

        # per-image folder
        stem = img_path.stem
        per_img_out = out_root / stem
        out_paths = save_sections(slices, per_img_out, base_name=stem)

        # also save a grid preview
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for ax, section, (sh, eh) in zip(axes.ravel(), slices, hour_ranges):
            ax.imshow(section)
            ax.axis("off")
            ax.set_title(f"{int(sh)}–{int(eh)} h")
        plt.tight_layout()
        fig.savefig(per_img_out / f"{stem}_grid.png", dpi=150)
        plt.close(fig)

        print(f"[ok] Saved {len(out_paths)} slices + grid -> {per_img_out}")

if __name__ == "__main__":
    # OPTION 1: Process all years automatically (2012-2024)
    # Auto-detects all keogram-out#### directories
    process_multiple_years(
        base_dir=".",
        year_range=None,  # Auto-detect, or specify (2012, 2024)
        n_sections=8,
        red_ratio_threshold=0.50,
        require_min_valid_frac=0.30,
        use_invalid_mask=True
    )
    
    # OPTION 2: Process specific year range
    # process_multiple_years(
    #     base_dir=".",
    #     year_range=(2012, 2024),
    #     n_sections=24,
    #     red_ratio_threshold=0.50,
    #     require_min_valid_frac=0.30,
    #     use_invalid_mask=True
    # )
    
    # OPTION 3: Process single year (legacy method)
    # write_segment_stats_csv_with_red_filter(
    #     input_dir="keogram-out2021",
    #     output_csv="keogram_segment_stats2021_24hours.csv",
    #     n_sections=24,
    #     red_ratio_threshold=0.50,
    #     require_min_valid_frac=0.30,
    #     use_invalid_mask=True
    # )