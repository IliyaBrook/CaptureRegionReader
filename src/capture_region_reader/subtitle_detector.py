import logging

import cv2
import numpy as np

from capture_region_reader.subtitle_binarizer import binarize_subtitle
from capture_region_reader.subtitle_cleaner import clean_artifacts

logger = logging.getLogger(__name__)


def _find_all_text_chars(
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_dim = max(2, int(min(h, w) * 0.005))

    chars = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_dim or ch < min_dim:
            continue
        if cw > w * 0.3 or ch > h * 0.5:
            continue
        if ch == 0 or cw / ch > 10 or cw / ch < 0.05:
            continue
        chars.append((x, y, cw, ch))

    return chars


def _compute_char_backgrounds(
    chars: list[tuple[int, int, int, int]],
    gray: np.ndarray,
) -> list[float]:
    h, w = gray.shape
    results = []
    for x, y, cw, ch in chars:
        pad = max(2, int(ch * 0.6))
        sy, sx = max(0, y - pad), max(0, x - pad)
        ey, ex = min(h, y + ch + pad), min(w, x + cw + pad)
        neighborhood = gray[sy:ey, sx:ex]
        char_y0, char_x0 = y - sy, x - sx
        mask = np.ones(neighborhood.shape, dtype=bool)
        mask[char_y0 : char_y0 + ch, char_x0 : char_x0 + cw] = False
        bg_pixels = neighborhood[mask]
        if len(bg_pixels) >= 5:
            results.append(float(np.mean(bg_pixels)))
        else:
            results.append(-1.0)
    return results


def _filter_by_dark_background(
    chars: list[tuple[int, int, int, int]],
    gray: np.ndarray,
    otsu_val: float,
) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    dark_threshold = otsu_val * 0.65

    result = []
    for x, y, cw, ch in chars:
        pad = max(2, int(ch * 0.6))
        sy, sx = max(0, y - pad), max(0, x - pad)
        ey, ex = min(h, y + ch + pad), min(w, x + cw + pad)
        neighborhood = gray[sy:ey, sx:ex]

        char_y0, char_x0 = y - sy, x - sx
        mask = np.ones(neighborhood.shape, dtype=bool)
        mask[char_y0 : char_y0 + ch, char_x0 : char_x0 + cw] = False
        bg_pixels = neighborhood[mask]

        if len(bg_pixels) < 5:
            continue

        bg_p90 = float(np.percentile(bg_pixels, 90))
        if bg_p90 < dark_threshold:
            result.append((x, y, cw, ch))

    return result


def _filter_by_size_consistency(
    chars: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(chars) < 3:
        return chars

    heights = np.array([c[3] for c in chars])
    median_h = float(np.median(heights))

    return [
        (x, y, cw, ch)
        for x, y, cw, ch in chars
        if median_h * 0.25 < ch < median_h * 3.5
        and cw < median_h * 8
    ]


def _filter_chars_by_bg_similarity(
    chars: list[tuple[int, int, int, int]],
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    bg_means = _compute_char_backgrounds(chars, gray)

    valid = [
        (chars[i], bg_means[i])
        for i in range(len(chars))
        if bg_means[i] >= 0
    ]
    if len(valid) < 3:
        return []

    valid.sort(key=lambda v: v[1])
    bg_sorted = [v[1] for v in valid]

    bg_range = bg_sorted[-1] - bg_sorted[0]
    window = max(20.0, bg_range * 0.15)

    best_start = 0
    best_count = 0
    j = 0
    for i in range(len(bg_sorted)):
        while j < len(bg_sorted) and bg_sorted[j] - bg_sorted[i] <= window:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = i

    cluster_low = bg_sorted[best_start]
    cluster_high = cluster_low + window

    result = [
        v[0]
        for v in valid
        if cluster_low <= v[1] <= cluster_high
    ]

    if len(result) < 3:
        return []

    text_ys = [(c[1], c[3]) for c in result]
    median_h = float(np.median([h for _, h in text_ys]))
    y_ranges = [(y - median_h * 0.3, y + h + median_h * 0.3) for y, h in text_ys]

    expanded = list(result)
    result_set = set(result)
    for char, bg_val in valid:
        if char in result_set:
            continue
        cy = char[1] + char[3] / 2.0
        for y_lo, y_hi in y_ranges:
            if y_lo <= cy <= y_hi:
                expanded.append(char)
                break

    return expanded


def _y_overlap_ratio(
    ay: int, ah: int, by: int, bh: int,
) -> float:
    overlap_start = max(ay, by)
    overlap_end = min(ay + ah, by + bh)
    overlap = max(0, overlap_end - overlap_start)
    smaller_h = min(ah, bh)
    if smaller_h == 0:
        return 0.0
    return overlap / smaller_h


def _merge_overlapping_blocks(
    blocks: list[tuple[int, int, int, int, int]],
    median_char_h: float,
) -> list[tuple[int, int, int, int, int]]:
    if len(blocks) <= 1:
        return blocks

    vertical_margin = median_char_h * 3.0

    merged = list(blocks)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            bx_i, by_i, bw_i, bh_i, bc_i = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx_j, by_j, bw_j, bh_j, bc_j = merged[j]

                overlap = _y_overlap_ratio(by_i, bh_i, by_j, bh_j)
                y_gap = max(
                    0,
                    max(by_i, by_j) - min(by_i + bh_i, by_j + bh_j),
                )

                if overlap > 0.3 or y_gap < vertical_margin:
                    nx = min(bx_i, bx_j)
                    ny = min(by_i, by_j)
                    nr = max(bx_i + bw_i, bx_j + bw_j)
                    nb = max(by_i + bh_i, by_j + bh_j)
                    bx_i, by_i = nx, ny
                    bw_i, bh_i = nr - nx, nb - ny
                    bc_i = bc_i + bc_j
                    used[j] = True
                    changed = True

            new_merged.append((bx_i, by_i, bw_i, bh_i, bc_i))
        merged = new_merged

    merged.sort(key=lambda b: b[4], reverse=True)
    return merged


def _group_into_blocks(
    chars: list[tuple[int, int, int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, int, int, int]]:
    if len(chars) < 2:
        return []

    median_h = float(np.median([c[3] for c in chars]))
    char_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, cw, ch in chars:
        char_mask[y : y + ch, x : x + cw] = 255

    kern_w = max(3, int(median_h * 3.0)) | 1
    kern_h_val = max(3, int(median_h * 0.5)) | 1
    kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kern_w, kern_h_val)
    )
    dilated = cv2.dilate(char_mask, kern, iterations=2)

    kern_v_w = max(3, int(median_h * 0.5)) | 1
    kern_v_h = max(3, int(median_h * 2.0)) | 1
    kern_v = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kern_v_w, kern_v_h)
    )
    dilated = cv2.dilate(dilated, kern_v, iterations=1)

    block_contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []
    for bc in block_contours:
        bx, by, bw, bh = cv2.boundingRect(bc)
        inside = [
            (cx, cy, ccw, cch)
            for cx, cy, ccw, cch in chars
            if bx <= cx
            and cy >= by
            and cx + ccw <= bx + bw
            and cy + cch <= by + bh
        ]
        if len(inside) >= 2:
            tight_x = min(c[0] for c in inside)
            tight_y = min(c[1] for c in inside)
            tight_r = max(c[0] + c[2] for c in inside)
            tight_b = max(c[1] + c[3] for c in inside)
            blocks.append(
                (
                    tight_x,
                    tight_y,
                    tight_r - tight_x,
                    tight_b - tight_y,
                    len(inside),
                )
            )

    blocks = _merge_overlapping_blocks(blocks, median_h)
    blocks.sort(key=lambda b: b[4], reverse=True)
    return blocks


def _trim_block_outliers(
    chars: list[tuple[int, int, int, int]],
    median_char_h: float,
) -> tuple[list[tuple[int, int, int, int]], int, int, int, int]:
    y_centers = np.array([c[1] + c[3] / 2.0 for c in chars])
    median_y = float(np.median(y_centers))

    iqr_margin = max(median_char_h * 4.0, float(np.std(y_centers)) * 2.0)
    y_lo = median_y - iqr_margin
    y_hi = median_y + iqr_margin

    filtered = [
        c for c, yc in zip(chars, y_centers)
        if y_lo <= yc <= y_hi
    ]
    if len(filtered) < 3:
        filtered = chars

    tx = min(c[0] for c in filtered)
    ty = min(c[1] for c in filtered)
    tr = max(c[0] + c[2] for c in filtered)
    tb = max(c[1] + c[3] for c in filtered)
    return filtered, tx, ty, tr - tx, tb - ty


def _is_consistent_strip(
    gray: np.ndarray,
    y_from: int,
    y_to: int,
    x_left: int,
    x_right: int,
    otsu_val: float,
    bg_reference: float | None = None,
) -> bool:
    if y_to <= y_from or x_right <= x_left:
        return True
    strip = gray[y_from:y_to, x_left:x_right]
    if strip.size == 0:
        return True
    strip_mean = float(np.mean(strip))
    if bg_reference is not None:
        tolerance = max(20.0, bg_reference * 0.35)
        return abs(strip_mean - bg_reference) < tolerance
    return strip_mean < otsu_val * 0.5


def _find_vertical_bounds(
    gray: np.ndarray,
    tx: int,
    ty: int,
    tw: int,
    th: int,
    median_char_h: float,
    otsu_val: float = 0.0,
    bg_reference: float | None = None,
) -> tuple[int, int]:
    h, w = gray.shape

    if bg_reference is not None:
        pad = max(2, int(median_char_h * 0.5))
        top_edge = max(0, ty - pad)
        bottom_edge = min(h, ty + th + pad)
        return top_edge, bottom_edge

    if otsu_val <= 0:
        otsu_val, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    scan_left = max(0, tx)
    scan_right = min(w, tx + tw)
    if scan_right <= scan_left:
        return ty, ty + th

    region = gray[:, scan_left:scan_right]
    sobel_y = cv2.Sobel(
        region.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3
    )
    row_edge = np.mean(np.abs(sobel_y), axis=1)

    search_limit = max(int(th * 2), int(h * 0.4))

    top_edge = ty
    search_start = max(0, ty - 1)
    search_end = max(0, ty - search_limit)
    if search_start > search_end:
        sub_edges = row_edge[search_end : search_start + 1]
        if len(sub_edges) > 2:
            edge_thresh = np.percentile(sub_edges, 65)
            for y_scan in range(search_start, search_end - 1, -1):
                if y_scan < 1 or y_scan >= h - 1:
                    continue
                if row_edge[y_scan] < edge_thresh:
                    continue
                row_above = float(
                    np.mean(
                        gray[y_scan - 1, scan_left:scan_right]
                    )
                )
                row_below = float(
                    np.mean(
                        gray[y_scan + 1, scan_left:scan_right]
                    )
                )
                if abs(row_above - row_below) > 6:
                    if _is_consistent_strip(
                        gray, y_scan, ty,
                        scan_left, scan_right, otsu_val,
                    ):
                        top_edge = y_scan
                    break

    bottom_edge = ty + th
    search_start_b = min(h - 1, ty + th + 1)
    search_end_b = min(h, ty + th + search_limit)
    if search_end_b > search_start_b:
        sub_edges = row_edge[search_start_b:search_end_b]
        if len(sub_edges) > 2:
            edge_thresh = np.percentile(sub_edges, 65)
            for y_scan in range(search_start_b, search_end_b):
                if y_scan < 1 or y_scan >= h - 1:
                    continue
                if row_edge[y_scan] < edge_thresh:
                    continue
                row_above = float(
                    np.mean(
                        gray[y_scan - 1, scan_left:scan_right]
                    )
                )
                row_below = float(
                    np.mean(
                        gray[y_scan + 1, scan_left:scan_right]
                    )
                )
                if abs(row_above - row_below) > 6:
                    if _is_consistent_strip(
                        gray, ty + th, y_scan,
                        scan_left, scan_right, otsu_val,
                    ):
                        bottom_edge = y_scan
                    break

    pad = max(1, int(median_char_h * 0.25))
    top_edge = max(0, top_edge - pad)
    bottom_edge = min(h, bottom_edge + pad)

    return top_edge, bottom_edge


def _find_horizontal_bounds(
    gray: np.ndarray,
    tx: int,
    ty: int,
    tw: int,
    th: int,
    top: int,
    bottom: int,
    median_char_h: float,
    otsu_val: float,
    bg_reference: float | None = None,
) -> tuple[int, int]:
    h, w = gray.shape

    if bg_reference is not None:
        pad = max(2, int(median_char_h * 0.5))
        left_bound = max(0, tx - pad)
        right_bound = min(w, tx + tw + pad)
        return left_bound, right_bound

    text_region = gray[top:bottom, :]
    if text_region.size == 0:
        return tx, tx + tw

    col_p10 = np.percentile(text_region, 10, axis=0)
    dark_thresh = otsu_val * 0.25
    is_bg_col = col_p10 < dark_thresh

    in_box = is_bg_col.copy()
    in_box[tx:tx + tw] = True

    box_left = tx
    for x in range(tx - 1, -1, -1):
        if in_box[x]:
            box_left = x
        else:
            break

    box_right = tx + tw
    for x in range(tx + tw, w):
        if in_box[x]:
            box_right = x
        else:
            break

    box_width = box_right - box_left

    if box_width > w * 0.85:
        pad = max(1, int(median_char_h * 1.5))
        left_bound = max(0, tx - pad)
        right_bound = min(w, tx + tw + pad)
        return left_bound, right_bound

    pad = max(1, int(median_char_h * 0.5))
    left_bound = max(0, box_left - pad)
    right_bound = min(w, box_right + pad)

    return left_bound, right_bound


def detect_and_crop(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    original_h, original_w = image.shape[:2]
    if original_h < 10 or original_w < 10:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if np.std(gray) < 3:
        return None

    otsu_val, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    all_chars = _find_all_text_chars(gray)
    if len(all_chars) < 2:
        return None

    sized_chars = _filter_by_size_consistency(all_chars)

    dark_chars = _filter_by_dark_background(sized_chars, gray, otsu_val)

    dark_block = None
    if len(dark_chars) >= 3:
        dark_blocks = _group_into_blocks(dark_chars, original_h, original_w)
        if dark_blocks:
            dark_block = dark_blocks[0]

    dark_is_valid = (
        dark_block is not None
        and dark_block[4] >= 3
        and dark_block[2] > dark_block[3]
    )

    if dark_is_valid:
        subtitle_chars = dark_chars
        used_dark_path = True
    else:
        cluster_chars = _filter_chars_by_bg_similarity(sized_chars, gray)
        if len(cluster_chars) >= 3:
            subtitle_chars = cluster_chars
            used_dark_path = False
        elif len(dark_chars) >= 3:
            subtitle_chars = dark_chars
            used_dark_path = True
        else:
            subtitle_chars = sized_chars
            used_dark_path = False

    if len(subtitle_chars) < 3:
        return None

    blocks = _group_into_blocks(
        subtitle_chars, original_h, original_w
    )

    if not blocks:
        return None

    tx, ty, tw, th = blocks[0][:4]

    if tw < original_w * 0.03 or th < original_h * 0.01:
        return None

    block_chars = [
        c for c in subtitle_chars
        if tx <= c[0] and c[1] >= ty
        and c[0] + c[2] <= tx + tw
        and c[1] + c[3] <= ty + th
    ]
    if not block_chars:
        block_chars = subtitle_chars

    char_heights = [c[3] for c in block_chars]
    median_char_h = (
        float(np.median(char_heights))
        if char_heights
        else th / 3.0
    )

    if not used_dark_path and len(block_chars) > 5:
        block_chars, tx, ty, tw, th = _trim_block_outliers(
            block_chars, median_char_h
        )

    bg_reference = None
    if not used_dark_path:
        bg_means = _compute_char_backgrounds(subtitle_chars, gray)
        valid_bgs = [b for b in bg_means if b >= 0]
        if valid_bgs:
            bg_reference = float(np.median(valid_bgs))

    top, bottom = _find_vertical_bounds(
        gray, tx, ty, tw, th, median_char_h, otsu_val, bg_reference
    )

    top = max(0, top)
    bottom = min(original_h, bottom)

    if bottom <= top:
        return None

    left, right = _find_horizontal_bounds(
        gray, tx, ty, tw, th, top, bottom, median_char_h, otsu_val,
        bg_reference,
    )

    left = max(0, left)
    right = min(original_w, right)

    if right <= left:
        return None

    crop_h = bottom - top
    crop_w = right - left

    if crop_h >= original_h and crop_w >= original_w:
        return None

    if crop_w / crop_h < 1.0:
        return None

    min_crop_w = int(crop_h * original_w / original_h) + 1
    if crop_w < min_crop_w and min_crop_w <= original_w:
        deficit = min_crop_w - crop_w
        expand_left = deficit // 2
        expand_right = deficit - expand_left

        if left - expand_left < 0:
            expand_right += expand_left - left
            expand_left = left
        if right + expand_right > original_w:
            extra = right + expand_right - original_w
            expand_left = min(left, expand_left + extra)
            expand_right = original_w - right

        left = max(0, left - expand_left)
        right = min(original_w, right + expand_right)
        crop_w = right - left

    cropped = image[top:bottom, left:right].copy()

    if crop_w < original_w:
        scale_factor = original_w / crop_w
        new_h = max(1, int(crop_h * scale_factor))
        if new_h >= original_h:
            new_h = original_h - 1
        scaled = cv2.resize(
            cropped,
            (original_w, new_h),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return clean_artifacts(binarize_subtitle(scaled))

    return clean_artifacts(binarize_subtitle(cropped))
