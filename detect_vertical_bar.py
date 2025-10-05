import cv2, os, csv
import numpy as np
from pathlib import Path
import re

def detect_vertical_white_purple_bars(img_bgr: np.ndarray) -> np.ndarray:
    """
    检测左右两侧贯穿垂直方向的白条/紫条。
    返回掩膜 (uint8, 0/255)，若无则全黑。
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    Hc, S, V = cv2.split(hsv)

    # 白色门控
    white  = (S <= 70) & (V >= 210)
    # 紫色门控
    purple = (Hc >= 135) & (Hc <= 175) & (S >= 70) & (V >= 120)
    cand = (white | purple).astype(np.uint8)

    # 每列占比 + 平滑
    col_frac = cand.sum(axis=0) / float(H)
    kernel = np.ones(21, np.float32)/21
    col_frac = np.convolve(col_frac, kernel, mode="same")
    cols = (col_frac >= 0.35).astype(np.uint8)

    # 合并连续列 -> 掩膜
    bar_mask = np.zeros((H,W), np.uint8)
    in_bar, start = False, 0
    for x in range(W):
        if cols[x] and not in_bar:
            in_bar, start = True, x
        if (not cols[x] or x==W-1) and in_bar:
            end = x if not cols[x] else x
            w = end - start + 1
            if w >= 4 and w <= 0.2*W:  # 合理宽度
                bar_mask[:, start:end+1] = 255
            in_bar = False

    # 纵向覆盖过滤
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bar_mask, connectivity=8)
    left_candidates, right_candidates = [], []
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if h/float(H) >= 0.85 and h/max(w,1) >= 6:
            cx = x+w/2
            if cx <= 0.5*W:
                left_candidates.append((area,i))
            else:
                right_candidates.append((area,i))

    # 保留左右各一个（面积最大）
    final = np.zeros_like(bar_mask)
    if left_candidates:
        _, idx = max(left_candidates, key=lambda t:t[0])
        final[labels==idx] = 255
    if right_candidates:
        _, idx = max(right_candidates, key=lambda t:t[0])
        final[labels==idx] = 255

    return final


def run_detect(input_path, output_dir,
               save_overlay=True, save_mask=True, save_inpaint=True,
               csv_path=None):
    """
    遍历文件夹，检测白条/紫条竖边。
    """
    in_path = Path(input_path)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    img_files = [p for p in in_path.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]]

    total_files = len(img_files)
    processed = 0
    detected = 0

    for p in img_files:
        img = cv2.imread(str(p))
        if img is None: 
            continue

        mask = detect_vertical_white_purple_bars(img)

        processed += 1

        if mask.sum() == 0:
            # 没有检测到，跳过
            continue

        detected += 1
        stem = p.stem
        if save_mask:
            cv2.imwrite(str(out_dir/f"{stem}_mask.png"), mask)
        if save_inpaint:
            inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(str(out_dir/f"{stem}_inpaint.png"), inpainted)
        if save_overlay:
            overlay = img.copy()
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,255), 2)
            cv2.imwrite(str(out_dir/f"{stem}_overlay.png"), overlay)

        rows.append([str(p), "bars_detected"])

    # 保存CSV
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file","status"])
            writer.writerows(rows)

    print(f"  [stats] Processed: {processed}/{total_files} files, Detected bars: {detected} files")
    return rows


def process_multiple_years(
        base_dir=".",
        year_range=None,
        input_dir_pattern="{}",  # e.g., "{}" for "2015" or "data_{}" for "data_2015"
        output_dir_pattern="keogram-out{}",  # output: "keogram-out2015"
        save_overlay=True,
        save_mask=True,
        save_inpaint=True,
        save_csv=True
    ):
    """
    Process multiple years of keogram images automatically.
    
    Args:
        base_dir: Base directory containing year folders
        year_range: Tuple of (start_year, end_year) inclusive, or None to auto-detect
        input_dir_pattern: Pattern for input directories (use {} for year placeholder)
        output_dir_pattern: Pattern for output directories (use {} for year placeholder)
        save_overlay: Save overlay images with detected bars highlighted
        save_mask: Save mask images
        save_inpaint: Save inpainted images (bars removed)
        save_csv: Save detection log CSV
    """
    base_path = Path(base_dir)
    
    # Auto-detect year directories if not specified
    if year_range is None:
        year_dirs = []
        for item in base_path.iterdir():
            if item.is_dir():
                # Try to extract 4-digit year from directory name
                match = re.search(r'\b(20\d{2})\b', item.name)
                if match:
                    year = int(match.group(1))
                    year_dirs.append((year, item))
        year_dirs.sort()
    else:
        start_year, end_year = year_range
        year_dirs = []
        for year in range(start_year, end_year + 1):
            input_dir = base_path / input_dir_pattern.format(year)
            if input_dir.exists():
                year_dirs.append((year, input_dir))
    
    if not year_dirs:
        print(f"[warn] No year directories found in {base_path}")
        print(f"[info] Looking for pattern: {input_dir_pattern.format('YYYY')}")
        return
    
    print(f"[info] Found {len(year_dirs)} year directories to process")
    print(f"{'='*70}")
    
    for year, input_dir in year_dirs:
        output_dir = base_path / output_dir_pattern.format(year)
        csv_path = output_dir / "detections.csv" if save_csv else None
        
        print(f"\n[info] Processing year {year}")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        
        rows = run_detect(
            input_path=str(input_dir),
            output_dir=str(output_dir),
            save_overlay=save_overlay,
            save_mask=save_mask,
            save_inpaint=save_inpaint,
            csv_path=str(csv_path) if csv_path else None
        )
        
        if rows:
            print(f"  [ok] Detected bars in {len(rows)} images")
        else:
            print(f"  [warn] No bars detected in any images")
    
    print(f"\n{'='*70}")
    print(f"[done] Processed all {len(year_dirs)} years")
    print(f"[info] Output directories ready for KeogramProcessing.py")
    print(f"{'='*70}")


def process_single_year(year, base_dir=".",
                       input_dir_pattern="{}",
                       output_dir_pattern="keogram-out{}",
                       save_overlay=True, save_mask=True, save_inpaint=True):
    """
    Process a single year (convenience function).
    """
    base_path = Path(base_dir)
    input_dir = base_path / input_dir_pattern.format(year)
    output_dir = base_path / output_dir_pattern.format(year)
    
    if not input_dir.exists():
        print(f"[error] Input directory not found: {input_dir}")
        return None
    
    csv_path = output_dir / "detections.csv"
    
    print(f"[info] Processing year {year}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    
    rows = run_detect(
        input_path=str(input_dir),
        output_dir=str(output_dir),
        save_overlay=save_overlay,
        save_mask=save_mask,
        save_inpaint=save_inpaint,
        csv_path=str(csv_path)
    )
    
    print(f"[done] Processed {len(rows)} images with detected bars")
    return rows


if __name__ == "__main__":
    # OPTION 1: Process all years automatically (recommended)
    # Auto-detects directories like "2012", "2013", ..., "2024"
    # Outputs to "keogram-out2012", "keogram-out2013", etc.
    process_multiple_years(
        base_dir=".",
        year_range=None,  # Auto-detect
        input_dir_pattern="{}",  # Looks for "2015", "2016", etc.
        output_dir_pattern="keogram-out{}",  # Creates "keogram-out2015", etc.
        save_overlay=True,
        save_mask=True,
        save_inpaint=True,
        save_csv=False
    )
    
    # OPTION 2: Process specific year range
    # process_multiple_years(
    #     base_dir=".",
    #     year_range=(2012, 2024),
    #     input_dir_pattern="{}",
    #     output_dir_pattern="keogram-out{}",
    #     save_overlay=True,
    #     save_mask=True,
    #     save_inpaint=True,
    #     save_csv=True
    # )
    
    # OPTION 3: Process single year (legacy method)
    # process_single_year(
    #     year=2015,
    #     base_dir=".",
    #     input_dir_pattern="{}",
    #     output_dir_pattern="keogram-out{}",
    #     save_overlay=True,
    #     save_mask=True,
    #     save_inpaint=True
    # )
    
    # OPTION 4: Custom directory patterns
    # If your input dirs are "data_2015", "data_2016", etc.
    # process_multiple_years(
    #     base_dir=".",
    #     year_range=(2015, 2024),
    #     input_dir_pattern="data_{}",  # Looks for "data_2015", etc.
    #     output_dir_pattern="keogram-out{}",
    #     save_overlay=True,
    #     save_mask=True,
    #     save_inpaint=True,
    #     save_csv=True
    # )