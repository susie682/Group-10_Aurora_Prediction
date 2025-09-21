import cv2, os, csv
import numpy as np
from pathlib import Path

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

    for p in img_files:
        img = cv2.imread(str(p))
        if img is None: continue

        mask = detect_vertical_white_purple_bars(img)

        if mask.sum() == 0:
            # 没有检测到，跳过
            continue

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

    return rows


rows2 = run_detect(
    input_path="./2012",
    output_dir="./keogram-out2012",
    save_overlay=True, save_mask=True, save_inpaint=True,
    csv_path="./keogram-out2/detections.csv"
)
print("finished")
