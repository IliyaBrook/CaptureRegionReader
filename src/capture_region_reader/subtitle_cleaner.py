import cv2
import numpy as np


def _get_components(
    gray: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    text_mask = (gray < 128).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_mask, connectivity=8,
    )
    return num_labels, labels, stats, centroids


def _median_char_height(
    stats: np.ndarray,
    num_labels: int,
    h: int,
    w: int,
) -> float:
    if num_labels <= 1:
        return h * 0.1

    total_area = h * w
    min_area = max(3, int(total_area * 0.00005))

    valid_heights: list[float] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]

        if area < min_area:
            continue
        if comp_h > h * 0.7 or comp_w > w * 0.7:
            continue
        if comp_h == 0:
            continue

        aspect = comp_w / comp_h
        if aspect < 0.08 or aspect > 12.0:
            continue

        valid_heights.append(float(comp_h))

    if len(valid_heights) >= 3:
        return float(np.median(valid_heights))
    elif len(valid_heights) > 0:
        return float(np.mean(valid_heights))
    return h * 0.1


def _is_char_like(
    comp_h: int,
    comp_w: int,
    area: int,
    median_h: float,
    h: int,
    w: int,
) -> bool:
    total_area = h * w
    min_area = max(3, int(total_area * 0.00005))

    if area < min_area:
        return False
    if comp_h < median_h * 0.2 or comp_h > median_h * 3.5:
        return False
    if comp_h > h * 0.7:
        return False
    if comp_w > w * 0.5:
        return False

    aspect = comp_w / max(1, comp_h)
    bbox_area = max(1, comp_w * comp_h)
    fill = area / bbox_area

    if aspect > 8.0:
        return False
    if fill < 0.02:
        return False

    return True


def _find_char_candidates(
    stats: np.ndarray,
    num_labels: int,
    median_h: float,
    h: int,
    w: int,
) -> list[int]:
    result: list[int] = []
    for i in range(1, num_labels):
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if _is_char_like(comp_h, comp_w, area, median_h, h, w):
            result.append(i)
    return result


def _group_into_lines(
    candidates: list[int],
    stats: np.ndarray,
    centroids: np.ndarray,
    median_h: float,
    h: int,
) -> list[list[int]]:
    if len(candidates) == 0:
        return []

    sorted_by_y = sorted(candidates, key=lambda i: centroids[i, 1])
    tolerance = max(median_h * 0.6, h * 0.02)

    lines: list[list[int]] = []
    current: list[int] = [sorted_by_y[0]]
    current_cy = float(centroids[sorted_by_y[0], 1])

    for k in range(1, len(sorted_by_y)):
        idx = sorted_by_y[k]
        cy = float(centroids[idx, 1])
        if abs(cy - current_cy) <= tolerance:
            current.append(idx)
            current_cy = float(np.mean([centroids[c, 1] for c in current]))
        else:
            lines.append(current)
            current = [idx]
            current_cy = cy

    lines.append(current)
    return lines


def _line_score(
    line: list[int],
    stats: np.ndarray,
    centroids: np.ndarray,
    median_h: float,
    h: int,
    w: int,
) -> float:
    if len(line) < 2:
        return 0.0

    heights = [float(stats[i, cv2.CC_STAT_HEIGHT]) for i in line]
    h_mean = float(np.mean(heights))
    h_std = float(np.std(heights))

    height_consistency = 1.0 - min(1.0, h_std / max(1.0, h_mean))

    good_count = sum(
        1 for hh in heights if median_h * 0.3 < hh < median_h * 2.5
    )
    good_ratio = good_count / max(1, len(line))

    lefts = [stats[i, cv2.CC_STAT_LEFT] for i in line]
    rights = [stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in line]
    span = max(rights) - min(lefts)
    span_ratio = span / max(1, w)

    count_score = min(1.0, len(line) / 5.0)

    score = (
        count_score * 0.3
        + span_ratio * 0.25
        + height_consistency * 0.2
        + good_ratio * 0.25
    )

    return score


def _split_line_by_gaps(
    line: list[int],
    stats: np.ndarray,
    median_h: float,
) -> list[list[int]]:
    if len(line) < 3:
        return [line]

    sorted_ids = sorted(line, key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    gaps: list[float] = []
    for j in range(1, len(sorted_ids)):
        prev_right = (
            stats[sorted_ids[j - 1], cv2.CC_STAT_LEFT]
            + stats[sorted_ids[j - 1], cv2.CC_STAT_WIDTH]
        )
        curr_left = stats[sorted_ids[j], cv2.CC_STAT_LEFT]
        gaps.append(float(curr_left - prev_right))

    if len(gaps) == 0:
        return [line]

    median_gap = float(np.median(gaps))
    gap_thresh = max(median_h * 4.0, median_gap * 5.0)

    splits: list[int] = []
    for j, g in enumerate(gaps):
        if g > gap_thresh:
            splits.append(j + 1)

    if len(splits) == 0:
        return [line]

    segments: list[list[int]] = []
    prev = 0
    for sp in splits:
        seg = sorted_ids[prev:sp]
        if len(seg) > 0:
            segments.append(seg)
        prev = sp
    tail = sorted_ids[prev:]
    if len(tail) > 0:
        segments.append(tail)

    return segments


def _pick_best_segment(
    segments: list[list[int]],
    stats: np.ndarray,
) -> list[int]:
    if len(segments) <= 1:
        return segments[0] if segments else []
    best = max(
        segments,
        key=lambda s: sum(stats[c, cv2.CC_STAT_AREA] for c in s),
    )
    return best


def _select_main_lines(
    lines: list[list[int]],
    stats: np.ndarray,
    centroids: np.ndarray,
    median_h: float,
    h: int,
    w: int,
) -> list[list[int]]:
    if len(lines) == 0:
        return []

    cleaned: list[list[int]] = []
    for line in lines:
        segs = _split_line_by_gaps(line, stats, median_h)
        best = _pick_best_segment(segs, stats)
        if len(best) >= 2:
            cleaned.append(best)

    if len(cleaned) == 0:
        return []

    scores = [
        _line_score(line, stats, centroids, median_h, h, w)
        for line in cleaned
    ]

    max_score = max(scores)
    if max_score <= 0:
        return cleaned

    edge_margin = max(1, int(median_h * 0.3))
    is_edge = []
    for line in cleaned:
        min_y = min(stats[i, cv2.CC_STAT_TOP] for i in line)
        max_y = max(
            stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            for i in line
        )
        is_edge.append(min_y < edge_margin or max_y > h - edge_margin)

    non_edge = [i for i in range(len(cleaned)) if not is_edge[i]]
    if non_edge:
        seed = max(non_edge, key=lambda i: scores[i])
    else:
        seed = max(range(len(cleaned)), key=lambda i: scores[i])

    selected: set[int] = {seed}

    line_cy = []
    line_h_cv = []
    line_span_ratio = []
    for line in cleaned:
        if len(line) > 0:
            line_cy.append(float(np.mean([centroids[c, 1] for c in line])))
            heights = [float(stats[c, cv2.CC_STAT_HEIGHT]) for c in line]
            h_mean = float(np.mean(heights))
            h_cv = float(np.std(heights)) / max(1.0, h_mean)
            line_h_cv.append(h_cv)
            lefts = [stats[c, cv2.CC_STAT_LEFT] for c in line]
            rights = [stats[c, cv2.CC_STAT_LEFT] + stats[c, cv2.CC_STAT_WIDTH] for c in line]
            line_span_ratio.append((max(rights) - min(lefts)) / max(1, w))
        else:
            line_cy.append(h / 2.0)
            line_h_cv.append(1.0)
            line_span_ratio.append(0.0)

    seed_span = line_span_ratio[seed]
    seed_h_cv = line_h_cv[seed]

    max_gap = median_h * 4.0

    changed = True
    while changed:
        changed = False
        for i in range(len(cleaned)):
            if i in selected:
                continue

            min_dist = min(abs(line_cy[i] - line_cy[j]) for j in selected)
            if min_dist > max_gap:
                continue

            is_nearby = min_dist <= median_h * 2.5

            if is_edge[i]:
                if line_h_cv[i] > 0.3:
                    continue
                if line_h_cv[i] > 0.2 and len(cleaned[i]) < 5:
                    continue
                if line_span_ratio[i] < 0.1 and len(cleaned[i]) < 5:
                    continue

            if is_edge[i] and not is_nearby:
                threshold = max_score * 0.55
            elif is_edge[i] and is_nearby:
                threshold = max_score * 0.2
            else:
                threshold = max_score * 0.25

            if scores[i] < threshold:
                continue

            if len(cleaned[i]) < 3 and is_edge[i] and not is_nearby:
                continue

            selected.add(i)
            changed = True

    return [cleaned[i] for i in sorted(selected)]


def _line_bounds(
    line: list[int],
    stats: np.ndarray,
) -> tuple[int, int, int, int]:
    top = min(stats[i, cv2.CC_STAT_TOP] for i in line)
    bottom = max(
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in line
    )
    left = min(stats[i, cv2.CC_STAT_LEFT] for i in line)
    right = max(
        stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in line
    )
    return top, bottom, left, right


def _compute_keep_set(
    stats: np.ndarray,
    centroids: np.ndarray,
    num_labels: int,
    selected_lines: list[list[int]],
    median_h: float,
    h: int,
    w: int,
) -> set[int]:
    keep: set[int] = set()
    for line in selected_lines:
        keep.update(line)

    if len(keep) == 0:
        return keep

    line_ranges = [_line_bounds(line, stats) for line in selected_lines]

    global_top = min(r[0] for r in line_ranges)
    global_bottom = max(r[1] for r in line_ranges)
    global_left = min(r[2] for r in line_ranges)
    global_right = max(r[3] for r in line_ranges)

    v_pad = max(2, int(median_h * 0.5))
    h_pad = max(2, int(median_h * 1.0))

    region_top = max(0, global_top - v_pad)
    region_bottom = min(h, global_bottom + v_pad)
    region_left = max(0, global_left - h_pad)
    region_right = min(w, global_right + h_pad)

    total_area = h * w
    min_area = max(3, int(total_area * 0.00005))

    for i in range(1, num_labels):
        if i in keep:
            continue

        comp_top = stats[i, cv2.CC_STAT_TOP]
        comp_left = stats[i, cv2.CC_STAT_LEFT]
        comp_h_val = stats[i, cv2.CC_STAT_HEIGHT]
        comp_w_val = stats[i, cv2.CC_STAT_WIDTH]
        comp_bottom = comp_top + comp_h_val
        comp_right = comp_left + comp_w_val
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if comp_top < region_top or comp_bottom > region_bottom:
            continue
        if comp_left < region_left or comp_right > region_right:
            continue

        if comp_h_val > median_h * 2.5:
            continue
        if comp_w_val > median_h * 4.0:
            continue

        aspect = comp_w_val / max(1, comp_h_val)
        if aspect > 6.0:
            continue

        keep.add(i)

    return keep


def _apply_and_crop(
    image: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    keep: set[int],
    median_h: float,
    h: int,
    w: int,
) -> np.ndarray:
    erase_mask = np.zeros((h, w), dtype=bool)
    for i in range(1, num_labels):
        if i not in keep:
            erase_mask |= (labels == i)

    result = image.copy()
    if len(result.shape) == 3:
        result[erase_mask] = [255, 255, 255]
    else:
        result[erase_mask] = 255

    if len(result.shape) == 3:
        check_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        check_gray = result

    row_has = np.any(check_gray < 128, axis=1)
    col_has = np.any(check_gray < 128, axis=0)
    content_rows = np.where(row_has)[0]
    content_cols = np.where(col_has)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return result

    pad_v = max(2, int(median_h * 0.4))
    pad_h = max(2, int(median_h * 0.3))

    trim_top = max(0, int(content_rows[0]) - pad_v)
    trim_bottom = min(h, int(content_rows[-1]) + 1 + pad_v)
    trim_left = max(0, int(content_cols[0]) - pad_h)
    trim_right = min(w, int(content_cols[-1]) + 1 + pad_h)

    if trim_top >= trim_bottom or trim_left >= trim_right:
        return result

    cropped = result[trim_top:trim_bottom, trim_left:trim_right].copy()

    if cropped.shape[1] < w:
        canvas = np.full((cropped.shape[0], w) + cropped.shape[2:], 255, dtype=np.uint8)
        x_offset = (w - cropped.shape[1]) // 2
        canvas[:, x_offset:x_offset + cropped.shape[1]] = cropped
        return canvas

    return cropped


def clean_artifacts(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    if h < 5 or w < 5:
        return image

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if len(image.shape) != 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    num_labels, labels, stats, centroids = _get_components(gray)

    if num_labels <= 1:
        return bgr

    median_h = _median_char_height(stats, num_labels, h, w)

    candidates = _find_char_candidates(stats, num_labels, median_h, h, w)

    if len(candidates) == 0:
        return bgr

    lines = _group_into_lines(candidates, stats, centroids, median_h, h)

    selected = _select_main_lines(
        lines, stats, centroids, median_h, h, w,
    )

    if len(selected) == 0:
        return bgr

    keep = _compute_keep_set(
        stats, centroids, num_labels, selected, median_h, h, w,
    )

    if len(keep) == 0:
        return bgr

    all_ids = set(range(1, num_labels))
    orig_area = sum(stats[i, cv2.CC_STAT_AREA] for i in all_ids)
    keep_area = sum(stats[i, cv2.CC_STAT_AREA] for i in keep)

    if orig_area > 0 and keep_area / orig_area < 0.05:
        return bgr

    return _apply_and_crop(bgr, labels, num_labels, keep, median_h, h, w)
