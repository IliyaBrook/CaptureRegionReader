"""Isolate subtitle text from a screenshot before OCR.

Instead of sending the raw screenshot to Tesseract (where background
logos, scene text, and UI elements confuse it), we first detect where
the actual subtitle text characters are, crop tightly around them,
and create a clean high-contrast image.

Pipeline:
1. Threshold to find bright regions (text candidates)
2. Find character contours and bounding boxes
3. Cluster contours into horizontal text lines
4. Score lines and select the best subtitle block
5. Crop to subtitle bounding box
6. Create clean black-on-white image for Tesseract
"""

from __future__ import annotations

import cv2
import numpy as np


def _find_char_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find individual character bounding boxes.

    Uses both bright-text and dark-text detection to work with any
    subtitle style.

    Returns list of (x, y, w, h) for each character-like contour.
    """
    img_h, img_w = gray.shape

    # Size limits for characters
    min_h = max(6, int(img_h * 0.05))
    max_h = int(img_h * 0.65)
    min_w = 2
    max_w = int(img_w * 0.25)
    min_area = 15

    all_boxes: list[tuple[int, int, int, int]] = []

    # Strategy 1: find BRIGHT text (white/light on dark bg)
    # Use Otsu to find optimal threshold
    _, bright_thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours_bright, _ = cv2.findContours(
        bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours_bright:
        x, y, w, h = cv2.boundingRect(cnt)
        if min_h <= h <= max_h and min_w <= w <= max_w:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                all_boxes.append((x, y, w, h))

    # Strategy 2: find DARK text (dark on light bg) if strategy 1 found few
    if len(all_boxes) < 3:
        _, dark_thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours_dark, _ = cv2.findContours(
            dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours_dark:
            x, y, w, h = cv2.boundingRect(cnt)
            if min_h <= h <= max_h and min_w <= w <= max_w:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    all_boxes.append((x, y, w, h))

    return all_boxes


def _cluster_into_lines(
    boxes: list[tuple[int, int, int, int]],
    img_h: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Cluster character boxes into horizontal text lines.

    Two boxes are on the same line if their vertical centers are close
    (within half the average character height).
    """
    if not boxes:
        return []

    # Sort by vertical center
    sorted_boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)

    lines: list[list[tuple[int, int, int, int]]] = []
    current_line: list[tuple[int, int, int, int]] = [sorted_boxes[0]]
    current_y_center = sorted_boxes[0][1] + sorted_boxes[0][3] / 2

    for box in sorted_boxes[1:]:
        box_y_center = box[1] + box[3] / 2
        avg_h = np.mean([b[3] for b in current_line])
        tolerance = max(avg_h * 0.6, 5)

        if abs(box_y_center - current_y_center) <= tolerance:
            current_line.append(box)
            current_y_center = np.mean([b[1] + b[3] / 2 for b in current_line])
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
            current_y_center = box_y_center

    lines.append(sorted(current_line, key=lambda b: b[0]))

    return lines


def _find_dense_core(
    line: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Within a line, find the densely packed core of characters.

    Subtitle characters are tightly spaced (normal word spacing).
    Logos or UI elements on the sides are far from the main text block.

    Removes outlier boxes that are far from the main cluster.
    """
    if len(line) <= 3:
        return line

    # Sort by x position
    sorted_line = sorted(line, key=lambda b: b[0])

    # Compute gaps between consecutive boxes
    gaps = []
    for i in range(1, len(sorted_line)):
        prev_end = sorted_line[i - 1][0] + sorted_line[i - 1][2]
        curr_start = sorted_line[i][0]
        gaps.append(curr_start - prev_end)

    if not gaps:
        return line

    # Median gap — the "normal" spacing between characters
    median_gap = float(np.median(gaps))
    # A gap is "too large" if it's > 3x the median (or > 50px absolute)
    max_allowed_gap = max(median_gap * 3, 50)

    # Find the longest run of characters with normal gaps
    best_start = 0
    best_end = 0
    best_count = 0

    run_start = 0
    for i, gap in enumerate(gaps):
        if gap > max_allowed_gap:
            run_count = i + 1 - run_start
            if run_count > best_count:
                best_start = run_start
                best_end = i + 1  # inclusive
                best_count = run_count
            run_start = i + 1

    # Don't forget the last run
    run_count = len(sorted_line) - run_start
    if run_count > best_count:
        best_start = run_start
        best_end = len(sorted_line)

    core = sorted_line[best_start:best_end]
    return core if len(core) >= 3 else line


def _select_subtitle_lines(
    lines: list[list[tuple[int, int, int, int]]],
    img_w: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Select the lines most likely to be subtitle text.

    For each line:
    1. Extract the dense core (remove far-away outliers like logos)
    2. Score by character count, width, consistency, density
    3. Keep the best-scoring lines
    """
    if not lines:
        return []

    scored: list[tuple[float, list]] = []

    for line in lines:
        # First, extract the dense core — remove distant outliers
        core = _find_dense_core(line)

        n = len(core)
        if n < 3:
            continue

        x_min = min(b[0] for b in core)
        x_max = max(b[0] + b[2] for b in core)
        line_width = x_max - x_min

        # Character height consistency
        heights = [b[3] for b in core]
        h_variation = np.std(heights) / (np.mean(heights) + 1)

        # Density: fraction of line width covered by chars
        char_total_w = sum(b[2] for b in core)
        density = char_total_w / (line_width + 1)

        score = 0.0
        score += n * 3
        score += (line_width / img_w) * 15
        score += max(0, 1 - h_variation) * 5
        score += min(density, 0.9) * 5

        if line_width < img_w * 0.08:
            score -= 15

        scored.append((score, core))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][0]

    result = []
    for score, line in scored:
        if score >= best * 0.35 and score > 5:
            result.append(line)
        if len(result) >= 4:
            break

    result.sort(key=lambda line: np.mean([b[1] for b in line]))
    return result


def isolate_text(rgb: np.ndarray) -> np.ndarray | None:
    """Isolate text from a screenshot, returning a clean image for OCR.

    Returns a clean black-text-on-white RGB image, or None if no text found.
    """
    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Step 1: find character boxes
    boxes = _find_char_boxes(gray)
    if len(boxes) < 3:
        return None

    # Step 2: cluster into lines
    all_lines = _cluster_into_lines(boxes, h)

    # Step 3: select subtitle lines
    subtitle_lines = _select_subtitle_lines(all_lines, w)
    if not subtitle_lines:
        return None

    # Collect all boxes from selected lines
    all_sub_boxes = [box for line in subtitle_lines for box in line]

    # Step 4: compute crop bounds
    crop_x_start = min(b[0] for b in all_sub_boxes)
    crop_x_end = max(b[0] + b[2] for b in all_sub_boxes)
    crop_y_start = min(b[1] for b in all_sub_boxes)
    crop_y_end = max(b[1] + b[3] for b in all_sub_boxes)

    # Padding
    pad_x = max(6, int((crop_x_end - crop_x_start) * 0.03))
    pad_y = max(4, int((crop_y_end - crop_y_start) * 0.15))
    crop_x_start = max(0, crop_x_start - pad_x)
    crop_x_end = min(w, crop_x_end + pad_x)
    crop_y_start = max(0, crop_y_start - pad_y)
    crop_y_end = min(h, crop_y_end + pad_y)

    # Step 5: create clean output
    cropped_gray = gray[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Build text mask in cropped coordinates
    ch, cw = cropped_gray.shape
    text_mask = np.zeros((ch, cw), dtype=np.uint8)
    for bx, by, bw, bh in all_sub_boxes:
        rx = bx - crop_x_start
        ry = by - crop_y_start
        rx_end = min(rx + bw, cw)
        ry_end = min(ry + bh, ch)
        rx = max(0, rx)
        ry = max(0, ry)
        text_mask[ry:ry_end, rx:rx_end] = 255

    # Determine text color
    text_pixels = cropped_gray[text_mask > 0]
    bg_pixels = cropped_gray[text_mask == 0]
    text_is_bright = True
    if len(text_pixels) > 0 and len(bg_pixels) > 0:
        text_is_bright = np.mean(text_pixels) > np.mean(bg_pixels)

    # Binarize with Otsu
    _, binary = cv2.threshold(
        cropped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if text_is_bright:
        # Invert: make text black on white
        result = cv2.bitwise_not(binary)
    else:
        result = binary

    # Clear non-text areas to white
    result[text_mask == 0] = 255

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
