"""Isolate subtitle text from a screenshot before OCR.

Pipeline:
1. Find character-like contours (brightness-filtered) on raw grayscale
2. Cluster into horizontal text lines
3. Score lines and pick the best subtitle block
4. Crop tightly around the text
5. HDR-adaptive enhancement on the cropped region (if needed)
6. Threshold by text color: if text is bright → darken everything
   that isn't bright (background), invert to black-on-white.
   No contour masks — just a global brightness threshold on the
   cropped region. This preserves letter interiors (о, е, а, д).
"""

from __future__ import annotations

import cv2
import numpy as np


# (bounding_box, mean_brightness)
CharBox = tuple[tuple[int, int, int, int], float]


def _preprocess_adaptive(rgb: np.ndarray) -> np.ndarray:
    """HDR-adaptive preprocessing for EasyOCR path.

    Analyzes image brightness/contrast and applies enhanced processing
    when the image has low contrast, extreme brightness, or HDR artifacts
    (common with game screenshots, semi-transparent overlays, etc.).

    NOTE: This function is used by EasyOCR (which has its own text detector)
    and as a post-crop enhancement in the Tesseract pipeline.  It must NOT
    be used before _find_char_boxes() because CLAHE redistributes brightness
    values and breaks the hardcoded brightness thresholds (160/100).

    Returns a grayscale image.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return _enhance_gray(gray)


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    """Apply HDR-adaptive enhancement to a grayscale image.

    Used on cropped regions (after contour detection) to improve
    Otsu thresholding on difficult backgrounds.
    """
    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))

    # Auto-detect when enhanced processing is needed:
    # - Low contrast (std < 40): HDR tone-mapped, flat images
    # - Very bright (mean > 200): washed-out, light backgrounds
    # - Very dark (mean < 55): dark game scenes with dim subtitles
    needs_enhanced = std_br < 40 or mean_br > 200 or mean_br < 55

    if needs_enhanced:
        # CLAHE: adaptive histogram equalization — boosts local contrast
        # in tile regions without over-amplifying noise globally.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter: reduces noise while preserving text edges.
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

        # Sharpening: 3x3 unsharp mask to make text edges crisper.
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(denoised, -1, kernel=kernel)

    return gray


def _is_blank_frame(gray: np.ndarray) -> bool:
    """Detect frames with no meaningful text content.

    Checks dynamic range and pixel variance to reject uniform/empty
    frames before Otsu amplifies noise into fake characters.

    Must be conservative: game subtitles can be dim/colored (not white),
    so only reject truly uniform images with almost zero contrast.
    """
    # Standard deviation of all pixel values.
    # A truly black/uniform screen has stdev < 3-5 (only JPEG noise).
    # Any screen with text (even dim) has stdev > 8-10.
    std = float(np.std(gray))
    if std < 5:
        return True

    return False


def _find_char_boxes(
    gray: np.ndarray,
) -> tuple[list[CharBox], bool]:
    """Find character bounding boxes with brightness measurement.

    Returns (list_of_CharBox, text_is_bright).
    Each CharBox is ((x, y, w, h), mean_brightness_of_pixels).

    Uses Otsu to find contours, measures brightness of each,
    then filters out dim contours (background bleed-through).
    """
    img_h, img_w = gray.shape

    min_h = max(6, int(img_h * 0.05))
    max_h = int(img_h * 0.65)
    min_w = 2
    max_w = int(img_w * 0.25)
    min_area = 15

    text_is_bright = True

    # --- Strategy 1: find BRIGHT text ---
    _, bright_thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bright_boxes: list[CharBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if min_h <= h <= max_h and min_w <= w <= max_w:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                roi = gray[y : y + h, x : x + w]
                # Measure brightness using Otsu mask region
                roi_thresh = bright_thresh[y : y + h, x : x + w]
                pixels = roi[roi_thresh > 0]
                br = float(np.mean(pixels)) if len(pixels) > 0 else float(np.mean(roi))
                bright_boxes.append(((x, y, w, h), br))

    # --- Strategy 2: find DARK text if strategy 1 found few ---
    if len(bright_boxes) < 3:
        _, dark_thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours_dark, _ = cv2.findContours(
            dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        dark_boxes: list[CharBox] = []
        for cnt in contours_dark:
            x, y, w, h = cv2.boundingRect(cnt)
            if min_h <= h <= max_h and min_w <= w <= max_w:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    roi = gray[y : y + h, x : x + w]
                    roi_thresh = dark_thresh[y : y + h, x : x + w]
                    pixels = roi[roi_thresh > 0]
                    br = float(np.mean(pixels)) if len(pixels) > 0 else float(np.mean(roi))
                    dark_boxes.append(((x, y, w, h), br))

        if len(dark_boxes) > len(bright_boxes):
            text_is_bright = False
            all_boxes = dark_boxes
        else:
            all_boxes = bright_boxes
    else:
        all_boxes = bright_boxes

    if not all_boxes:
        return [], text_is_bright

    # --- Brightness-based filtering ---
    brightnesses = np.array([cb[1] for cb in all_boxes])

    if text_is_bright:
        target = float(np.percentile(brightnesses, 75))
        min_br = max(target * 0.70, 160)
        filtered = [cb for cb in all_boxes if cb[1] >= min_br]
    else:
        target = float(np.percentile(brightnesses, 25))
        max_br = min(target * 1.5, 100)
        filtered = [cb for cb in all_boxes if cb[1] <= max_br]

    if len(filtered) < 3:
        filtered = all_boxes

    # --- Noise plausibility check ---
    # On noisy frames, Otsu produces many tiny scattered contours.
    # Real text has consistent heights and reasonable character size.
    if len(filtered) >= 3:
        areas = [cb[0][2] * cb[0][3] for cb in filtered]
        heights = [cb[0][3] for cb in filtered]
        median_area = float(np.median(areas))
        median_height = float(np.median(heights))

        # Real characters have median area >= 50px; noise blobs are ~4-16px.
        if median_area < 50:
            return [], text_is_bright

        # Real text has consistent heights (same font size).
        # Noise contours have wildly varying heights.
        if len(heights) >= 5:
            h_cv = float(np.std(heights)) / (median_height + 1)
            if h_cv > 0.8:
                return [], text_is_bright

    return filtered, text_is_bright


def _cluster_into_lines(
    boxes: list[CharBox],
    img_h: int,
) -> list[list[CharBox]]:
    """Cluster character boxes into horizontal text lines."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda cb: cb[0][1] + cb[0][3] / 2)

    lines: list[list[CharBox]] = []
    current_line: list[CharBox] = [sorted_boxes[0]]
    current_y = sorted_boxes[0][0][1] + sorted_boxes[0][0][3] / 2

    for cb in sorted_boxes[1:]:
        box = cb[0]
        by = box[1] + box[3] / 2
        avg_h = np.mean([c[0][3] for c in current_line])
        tolerance = max(avg_h * 0.6, 5)

        if abs(by - current_y) <= tolerance:
            current_line.append(cb)
            current_y = np.mean([c[0][1] + c[0][3] / 2 for c in current_line])
        else:
            lines.append(sorted(current_line, key=lambda c: c[0][0]))
            current_line = [cb]
            current_y = by

    lines.append(sorted(current_line, key=lambda c: c[0][0]))
    return lines


def _find_dense_core(line: list[CharBox]) -> list[CharBox]:
    """Find the densely packed core of a text line.

    Removes far-away outliers (logos, icons) while keeping
    normal text including punctuation and dashes.
    """
    if len(line) <= 3:
        return line

    sorted_line = sorted(line, key=lambda c: c[0][0])

    gaps = []
    for i in range(1, len(sorted_line)):
        prev = sorted_line[i - 1][0]
        curr = sorted_line[i][0]
        gaps.append(curr[0] - (prev[0] + prev[2]))

    if not gaps:
        return line

    median_gap = float(np.median(gaps))
    avg_w = np.mean([c[0][2] for c in sorted_line])
    # Very generous: don't split on dashes, punctuation, etc.
    max_gap = max(median_gap * 5, avg_w * 2, 80)

    best_start = 0
    best_end = 0
    best_count = 0
    run_start = 0

    for i, gap in enumerate(gaps):
        if gap > max_gap:
            run_count = i + 1 - run_start
            if run_count > best_count:
                best_start = run_start
                best_end = i + 1
                best_count = run_count
            run_start = i + 1

    run_count = len(sorted_line) - run_start
    if run_count > best_count:
        best_start = run_start
        best_end = len(sorted_line)

    core = sorted_line[best_start:best_end]
    return core if len(core) >= 3 else line


def _select_subtitle_lines(
    lines: list[list[CharBox]],
    img_w: int,
    img_h: int,
) -> list[list[CharBox]]:
    """Score and select the best subtitle lines.

    Uses a two-pass approach:
    1. Score each line individually, pick the best one.
    2. Include nearby lines that look like part of the same subtitle
       block (similar char height, close vertical distance).
    """
    if not lines:
        return []

    # Build scored list with metadata for proximity checks
    scored: list[tuple[float, list[CharBox], float, float]] = []
    # Each entry: (score, core_boxes, y_center, avg_char_height)

    for line in lines:
        core = _find_dense_core(line)
        n = len(core)
        if n < 3:
            continue

        boxes = [c[0] for c in core]
        x_min = min(b[0] for b in boxes)
        x_max = max(b[0] + b[2] for b in boxes)
        line_w = x_max - x_min

        heights = [b[3] for b in boxes]
        avg_h = float(np.mean(heights))
        h_var = float(np.std(heights)) / (avg_h + 1)

        char_w_total = sum(b[2] for b in boxes)
        density = char_w_total / (line_w + 1)

        y_center = float(np.mean([b[1] + b[3] / 2 for b in boxes]))
        y_pos = y_center / img_h

        score = 0.0
        score += n * 3
        score += (line_w / img_w) * 15
        score += max(0, 1 - h_var) * 5
        score += min(density, 0.9) * 5
        score += y_pos * 8

        if line_w < img_w * 0.08:
            score -= 15

        scored.append((score, core, y_center, avg_h))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_line, best_y, best_h = scored[0]

    # Pass 1: always include the best line
    result: list[list[CharBox]] = [best_line]
    # Track y-centers of accepted lines for proximity to nearest neighbor
    accepted_ys: list[float] = [best_y]

    # Pass 2: include nearby lines that are part of the same subtitle block.
    # A companion line must:
    #   - have similar character height (within 50% of best)
    #   - be vertically close to ANY already-accepted line (not just best)
    #   - have a minimum score (not total garbage)
    # We loop multiple times so line 3 can be accepted via proximity to line 2.
    remaining = [(score, line, y_center, avg_h) for score, line, y_center, avg_h in scored[1:]]
    changed = True
    while changed and len(result) < 6:
        changed = False
        still_remaining = []
        for score, line, y_center, avg_h in remaining:
            # Check character height similarity
            h_ratio = avg_h / (best_h + 1)
            if h_ratio < 0.5 or h_ratio > 1.8:
                still_remaining.append((score, line, y_center, avg_h))
                continue

            # Check vertical proximity to nearest accepted line.
            # Use the larger char height of the two lines for a robust check.
            # Subtitle line spacing (leading) is typically 1.2-1.5x font size,
            # but contour height < font em-height, so center-to-center distance
            # can be 2.2-2.5x contour height.  Use 3.0x for safe headroom.
            ref_h = max(avg_h, best_h)
            min_dist = min(abs(y_center - ay) for ay in accepted_ys)
            if min_dist > ref_h * 3.0:
                still_remaining.append((score, line, y_center, avg_h))
                continue

            # Minimum quality: must have at least some substance
            if score > 5:
                result.append(line)
                accepted_ys.append(y_center)
                changed = True
            else:
                still_remaining.append((score, line, y_center, avg_h))
        remaining = still_remaining

    result.sort(key=lambda line: np.mean([c[0][1] for c in line]))
    return result


def isolate_text(rgb: np.ndarray) -> np.ndarray | None:
    """Isolate subtitle text from a screenshot.

    Returns a clean black-text-on-white RGB image, or None.

    Approach: find text lines → crop around them → apply a simple
    brightness threshold to separate text from background.
    No contour masks — preserves full letter shapes (о, е, а, д).
    """
    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        return None

    # Use RAW grayscale for contour detection — brightness thresholds in
    # _find_char_boxes() are calibrated for unprocessed pixel values.
    # CLAHE/HDR enhancement is applied later, only on the cropped region.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Early rejection: blank/uniform frames produce noise under Otsu
    if _is_blank_frame(gray):
        return None

    # Step 1: find char boxes (brightness-filtered)
    boxes, text_is_bright = _find_char_boxes(gray)
    if len(boxes) < 3:
        return None

    # Step 2: cluster into lines
    all_lines = _cluster_into_lines(boxes, h)

    # Step 3: select subtitle lines
    sub_lines = _select_subtitle_lines(all_lines, w, h)
    if not sub_lines:
        return None

    # Collect all boxes from selected lines
    all_sub = [cb for line in sub_lines for cb in line]
    all_rects = [cb[0] for cb in all_sub]

    # Step 4: crop bounds with padding
    cx1 = min(b[0] for b in all_rects)
    cx2 = max(b[0] + b[2] for b in all_rects)
    cy1 = min(b[1] for b in all_rects)
    cy2 = max(b[1] + b[3] for b in all_rects)

    pad_x = max(6, int((cx2 - cx1) * 0.03))
    # Generous vertical padding: use character height as baseline, not block height.
    # This prevents clipping descenders/ascenders and catches nearby text.
    avg_char_h = float(np.mean([b[3] for b in all_rects]))
    pad_y = max(6, int(avg_char_h * 0.5))
    cx1 = max(0, cx1 - pad_x)
    cx2 = min(w, cx2 + pad_x)
    cy1 = max(0, cy1 - pad_y)
    cy2 = min(h, cy2 + pad_y)

    cropped = gray[cy1:cy2, cx1:cx2]

    # Step 5: HDR-adaptive enhancement on the cropped region.
    # CLAHE is safe here because we already found char boxes using raw values.
    # It improves Otsu thresholding on difficult backgrounds (low contrast,
    # semi-transparent overlays, HDR game scenes).
    cropped = _enhance_gray(cropped)

    # Step 6: simple brightness threshold — no contour masks!
    #
    # Determine the text brightness from the detected chars.
    # Then threshold: keep only pixels near text brightness,
    # everything else becomes background.
    text_brs = [cb[1] for cb in all_sub]
    avg_text_br = np.mean(text_brs)

    if text_is_bright:
        # Text is bright (white subtitles on dark/semi-dark bg).
        # Find the threshold that separates text from background.
        # Use a value between background and text brightness.
        # Text is typically 200-255, background 0-150.
        # Use Otsu on the cropped region — it naturally finds this boundary.
        _, binary = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # binary: text=255 (white), background=0 (black)
        # We want black text on white → invert
        result = cv2.bitwise_not(binary)
    else:
        # Dark text on light background
        _, binary = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # binary: text=0 (black), background=255 (white)
        result = binary

    # Step 6: clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
