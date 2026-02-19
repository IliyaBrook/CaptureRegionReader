"""Isolate subtitle text from a screenshot before OCR.

Pipeline:
1. Find character-like contours (brightness-filtered) on raw grayscale
2. Cluster into horizontal text lines
3. Score lines and pick the best subtitle block
4. Crop tightly around the text
5. HDR-adaptive enhancement on the cropped region (if needed)
6. Threshold by text color: if text is bright -> darken everything
   that isn't bright (background), invert to black-on-white.
   No contour masks -- just a global brightness threshold on the
   cropped region. This preserves letter interiors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IsolatorConfig:
    """Tunable parameters for the text isolation pipeline.

    All thresholds and limits are collected here so they can be adjusted
    without editing detection logic. Default values are calibrated for
    game subtitles (white or colored text on dark/semi-dark backgrounds).
    """

    # -- Blank-frame detection --
    # Standard deviation below which a grayscale frame is considered blank
    # (only JPEG noise, no visible content). Real text raises stdev above 8.
    blank_frame_std_threshold: float = 5.0

    # -- Character contour size limits (relative to image dimensions) --
    # Minimum character height as fraction of image height.
    # Prevents tiny noise from being treated as characters.
    char_min_height_ratio: float = 0.05
    # Absolute minimum character height in pixels (overrides ratio for small images).
    char_min_height_abs: int = 6
    # Maximum character height as fraction of image height.
    # Filters out large non-text contours (logos, UI panels).
    char_max_height_ratio: float = 0.65
    # Minimum character width in pixels.
    char_min_width: int = 2
    # Maximum character width as fraction of image width.
    # Prevents wide horizontal bars from being treated as characters.
    char_max_width_ratio: float = 0.25
    # Minimum contour area in pixels. Rejects tiny dots and speckles.
    char_min_area: int = 15

    # -- Brightness filtering --
    # For bright text: percentile of brightness values used as reference point.
    bright_text_percentile: int = 75
    # For bright text: minimum brightness as fraction of reference point.
    bright_text_min_ratio: float = 0.70
    # For bright text: absolute minimum brightness floor.
    # Characters dimmer than this are filtered even if ratio passes.
    bright_text_min_abs: int = 160
    # For dark text: percentile of brightness values used as reference point.
    dark_text_percentile: int = 25
    # For dark text: maximum brightness as multiple of reference point.
    dark_text_max_ratio: float = 1.5
    # For dark text: absolute maximum brightness ceiling.
    dark_text_max_abs: int = 100

    # Minimum number of char boxes to consider a valid text region.
    min_char_boxes: int = 3
    # Minimum number of boxes to fall back to unfiltered set.
    min_filtered_boxes: int = 3

    # -- Noise plausibility --
    # Minimum median area (px) for character contours. Noise blobs are ~4-16px.
    noise_min_median_area: int = 50
    # Maximum coefficient of variation for character heights.
    # Real text from a single font has consistent heights (CV < 0.8).
    noise_max_height_cv: float = 0.8
    # Minimum boxes to run height consistency check.
    noise_min_boxes_for_cv: int = 5

    # -- Line clustering --
    # Y-tolerance for clustering: fraction of average character height in line.
    line_cluster_y_tolerance_ratio: float = 0.6
    # Absolute minimum Y-tolerance in pixels.
    line_cluster_y_tolerance_min: int = 5

    # -- Dense core detection --
    # Gap threshold multiplier over median gap.
    dense_core_gap_median_mult: float = 5.0
    # Gap threshold multiplier over average character width.
    dense_core_gap_width_mult: float = 2.0
    # Absolute maximum gap between characters in a dense run.
    dense_core_gap_abs: int = 80
    # Minimum characters for dense core detection to activate.
    dense_core_min_chars: int = 3

    # -- Subtitle line scoring --
    # Weight for character count in line score.
    score_char_count_weight: float = 3.0
    # Weight for line width ratio in line score.
    score_width_ratio_weight: float = 15.0
    # Weight for height consistency in line score.
    score_height_consistency_weight: float = 5.0
    # Weight for character density in line score.
    score_density_weight: float = 5.0
    # Weight for vertical position in line score (bottom = higher score).
    score_y_position_weight: float = 8.0
    # Penalty for very narrow lines (width < this fraction of image width).
    score_narrow_line_threshold: float = 0.08
    score_narrow_line_penalty: float = 15.0
    # Minimum score for a companion line to be accepted.
    score_companion_min: float = 5.0
    # Character height ratio range for companion lines.
    companion_height_ratio_min: float = 0.5
    companion_height_ratio_max: float = 1.8
    # Maximum vertical distance for companion lines (multiplier of char height).
    companion_max_y_distance: float = 3.0
    # Maximum subtitle lines to accept.
    max_subtitle_lines: int = 6

    # -- Crop padding --
    # Horizontal padding as fraction of crop width.
    crop_pad_x_ratio: float = 0.03
    # Absolute minimum horizontal padding.
    crop_pad_x_min: int = 6
    # Vertical padding as fraction of average character height.
    crop_pad_y_ratio: float = 0.5
    # Absolute minimum vertical padding.
    crop_pad_y_min: int = 6

    # -- HDR-adaptive enhancement --
    # CLAHE clip limit for adaptive histogram equalization.
    clahe_clip_limit: float = 2.5
    # CLAHE tile grid size (width and height of tiles).
    clahe_tile_size: int = 8
    # Bilateral filter diameter for denoising.
    bilateral_d: int = 5
    # Bilateral filter sigma for color and space.
    bilateral_sigma: float = 50.0
    # Brightness standard deviation below which enhanced processing activates.
    enhance_std_threshold: float = 40.0
    # Mean brightness above which enhanced processing activates (washed out).
    enhance_bright_threshold: float = 200.0
    # Mean brightness below which enhanced processing activates (too dark).
    enhance_dark_threshold: float = 55.0

    # -- Morphological cleanup --
    # Kernel size for opening operation to remove small noise in final binary.
    morph_open_kernel_size: int = 2


# Singleton default config used by the module-level API.
_default_config = IsolatorConfig()


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# (bounding_box, mean_brightness)
CharBox = tuple[tuple[int, int, int, int], float]


# ---------------------------------------------------------------------------
# HDR-adaptive enhancement
# ---------------------------------------------------------------------------

def _enhance_gray(gray: np.ndarray, cfg: IsolatorConfig) -> np.ndarray:
    """Apply HDR-adaptive enhancement to a grayscale image.

    Analyzes brightness distribution and applies CLAHE + bilateral
    filtering + sharpening when the image has low contrast, extreme
    brightness, or HDR artifacts (game screenshots, semi-transparent
    overlays). Otherwise returns the image unchanged.

    Used on cropped regions (after contour detection) to improve
    Otsu thresholding on difficult backgrounds.
    """
    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))

    needs_enhanced = (
        std_br < cfg.enhance_std_threshold
        or mean_br > cfg.enhance_bright_threshold
        or mean_br < cfg.enhance_dark_threshold
    )

    if not needs_enhanced:
        return gray

    # CLAHE: adaptive histogram equalization -- boosts local contrast
    # in tile regions without over-amplifying noise globally.
    tile = (cfg.clahe_tile_size, cfg.clahe_tile_size)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=tile)
    enhanced = clahe.apply(gray)

    # Bilateral filter: reduces noise while preserving text edges.
    denoised = cv2.bilateralFilter(
        enhanced, cfg.bilateral_d, cfg.bilateral_sigma, cfg.bilateral_sigma
    )

    # Sharpening: 3x3 unsharp mask to make text edges crisper.
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel=sharpen_kernel)

    return sharpened


def _preprocess_adaptive(rgb: np.ndarray) -> np.ndarray:
    """HDR-adaptive preprocessing for EasyOCR path.

    NOTE: This function is used by EasyOCR (which has its own text detector)
    and as a post-crop enhancement in the Tesseract pipeline. It must NOT
    be used before _find_char_boxes() because CLAHE redistributes brightness
    values and breaks the brightness thresholds used for filtering.

    Returns a grayscale image.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return _enhance_gray(gray, _default_config)


# ---------------------------------------------------------------------------
# Blank frame detection
# ---------------------------------------------------------------------------

def _is_blank_frame(gray: np.ndarray, cfg: IsolatorConfig) -> bool:
    """Detect frames with no meaningful text content.

    Checks pixel variance to reject uniform/empty frames before Otsu
    amplifies noise into fake characters.

    Conservative threshold: game subtitles can be dim/colored (not white),
    so only reject truly uniform images with near-zero contrast.
    """
    std = float(np.std(gray))
    return std < cfg.blank_frame_std_threshold


# ---------------------------------------------------------------------------
# Character detection
# ---------------------------------------------------------------------------

def _measure_contour_boxes(
    gray: np.ndarray,
    thresh_img: np.ndarray,
    contours: list,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_area: int,
) -> list[CharBox]:
    """Measure brightness of contour bounding boxes that pass size filters.

    For each contour that meets the size criteria, measures mean brightness
    of the thresholded foreground pixels within the bounding box.
    """
    boxes: list[CharBox] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (min_h <= h <= max_h and min_w <= w <= max_w):
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        roi = gray[y : y + h, x : x + w]
        roi_thresh = thresh_img[y : y + h, x : x + w]
        fg_pixels = roi[roi_thresh > 0]
        br = float(np.mean(fg_pixels)) if len(fg_pixels) > 0 else float(np.mean(roi))
        boxes.append(((x, y, w, h), br))

    return boxes


def _find_char_boxes(
    gray: np.ndarray,
    cfg: IsolatorConfig,
) -> tuple[list[CharBox], bool]:
    """Find character bounding boxes with brightness measurement.

    Returns (list_of_CharBox, text_is_bright).

    Strategy:
    1. Try BRIGHT text first (Otsu BINARY -- foreground = bright pixels).
    2. If few bright contours found, try DARK text (Otsu BINARY_INV).
    3. Pick whichever strategy found more plausible characters.
    4. Filter by brightness to remove background bleed-through.
    5. Validate against noise (median area, height consistency).
    """
    img_h, img_w = gray.shape

    # Compute size limits from image dimensions and config
    min_h = max(cfg.char_min_height_abs, int(img_h * cfg.char_min_height_ratio))
    max_h = int(img_h * cfg.char_max_height_ratio)
    min_w = cfg.char_min_width
    max_w = int(img_w * cfg.char_max_width_ratio)
    min_area = cfg.char_min_area

    text_is_bright = True

    # --- Strategy 1: find BRIGHT text ---
    _, bright_thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours_bright, _ = cv2.findContours(
        bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bright_boxes = _measure_contour_boxes(
        gray, bright_thresh, contours_bright, min_h, max_h, min_w, max_w, min_area
    )

    # --- Strategy 2: find DARK text if strategy 1 found few ---
    if len(bright_boxes) < cfg.min_char_boxes:
        _, dark_thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours_dark, _ = cv2.findContours(
            dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        dark_boxes = _measure_contour_boxes(
            gray, dark_thresh, contours_dark, min_h, max_h, min_w, max_w, min_area
        )

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
        target = float(np.percentile(brightnesses, cfg.bright_text_percentile))
        min_br = max(target * cfg.bright_text_min_ratio, cfg.bright_text_min_abs)
        filtered = [cb for cb in all_boxes if cb[1] >= min_br]
    else:
        target = float(np.percentile(brightnesses, cfg.dark_text_percentile))
        max_br = min(target * cfg.dark_text_max_ratio, cfg.dark_text_max_abs)
        filtered = [cb for cb in all_boxes if cb[1] <= max_br]

    if len(filtered) < cfg.min_filtered_boxes:
        filtered = all_boxes

    # --- Noise plausibility check ---
    # On noisy frames Otsu produces many tiny scattered contours.
    # Real text has consistent heights and reasonable character size.
    if len(filtered) >= cfg.min_char_boxes:
        areas = [cb[0][2] * cb[0][3] for cb in filtered]
        heights = [cb[0][3] for cb in filtered]
        median_area = float(np.median(areas))
        median_height = float(np.median(heights))

        # Real characters have median area >= noise_min_median_area;
        # noise blobs are typically 4--16 px.
        if median_area < cfg.noise_min_median_area:
            return [], text_is_bright

        # Real text from a single font has consistent heights.
        # Noise contours have wildly varying heights.
        if len(heights) >= cfg.noise_min_boxes_for_cv:
            h_cv = float(np.std(heights)) / (median_height + 1)
            if h_cv > cfg.noise_max_height_cv:
                return [], text_is_bright

    return filtered, text_is_bright


# ---------------------------------------------------------------------------
# Line clustering
# ---------------------------------------------------------------------------

def _cluster_into_lines(
    boxes: list[CharBox],
    cfg: IsolatorConfig,
) -> list[list[CharBox]]:
    """Cluster character boxes into horizontal text lines.

    Sorts boxes by vertical center, then groups boxes whose vertical
    centers are within a tolerance of the running line average.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda cb: cb[0][1] + cb[0][3] / 2)

    lines: list[list[CharBox]] = []
    current_line: list[CharBox] = [sorted_boxes[0]]
    current_y = sorted_boxes[0][0][1] + sorted_boxes[0][0][3] / 2

    for cb in sorted_boxes[1:]:
        box = cb[0]
        by = box[1] + box[3] / 2
        avg_h = float(np.mean([c[0][3] for c in current_line]))
        tolerance = max(avg_h * cfg.line_cluster_y_tolerance_ratio,
                        cfg.line_cluster_y_tolerance_min)

        if abs(by - current_y) <= tolerance:
            current_line.append(cb)
            current_y = float(np.mean([c[0][1] + c[0][3] / 2 for c in current_line]))
        else:
            lines.append(sorted(current_line, key=lambda c: c[0][0]))
            current_line = [cb]
            current_y = by

    lines.append(sorted(current_line, key=lambda c: c[0][0]))
    return lines


# ---------------------------------------------------------------------------
# Dense core detection
# ---------------------------------------------------------------------------

def _find_dense_core(line: list[CharBox], cfg: IsolatorConfig) -> list[CharBox]:
    """Find the densely packed core of a text line.

    Removes far-away outliers (logos, icons) while keeping
    normal text including punctuation and dashes.
    """
    if len(line) <= cfg.dense_core_min_chars:
        return line

    sorted_line = sorted(line, key=lambda c: c[0][0])

    gaps = []
    for i in range(1, len(sorted_line)):
        prev = sorted_line[i - 1][0]
        curr = sorted_line[i][0]
        gap = curr[0] - (prev[0] + prev[2])
        gaps.append(gap)

    if not gaps:
        return line

    median_gap = float(np.median(gaps))
    avg_w = float(np.mean([c[0][2] for c in sorted_line]))
    # Generous threshold: don't split on dashes, punctuation, etc.
    max_gap = max(
        median_gap * cfg.dense_core_gap_median_mult,
        avg_w * cfg.dense_core_gap_width_mult,
        cfg.dense_core_gap_abs,
    )

    # Find the longest run of characters without a gap > max_gap
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

    # Check last run
    run_count = len(sorted_line) - run_start
    if run_count > best_count:
        best_start = run_start
        best_end = len(sorted_line)

    core = sorted_line[best_start:best_end]
    return core if len(core) >= cfg.min_char_boxes else line


# ---------------------------------------------------------------------------
# Subtitle line selection
# ---------------------------------------------------------------------------

def _score_line(
    core: list[CharBox],
    img_w: int,
    img_h: int,
    cfg: IsolatorConfig,
) -> tuple[float, float, float]:
    """Score a single text line for subtitle likelihood.

    Returns (score, y_center, avg_char_height).

    Scoring criteria:
    - Character count: more characters = more likely real text
    - Line width: wider = more likely a sentence
    - Height consistency: uniform heights = same font
    - Character density: higher = fewer gaps = real words
    - Vertical position: subtitles tend to be at the bottom
    """
    boxes = [c[0] for c in core]
    n = len(boxes)

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
    score += n * cfg.score_char_count_weight
    score += (line_w / img_w) * cfg.score_width_ratio_weight
    score += max(0, 1 - h_var) * cfg.score_height_consistency_weight
    score += min(density, 0.9) * cfg.score_density_weight
    score += y_pos * cfg.score_y_position_weight

    if line_w < img_w * cfg.score_narrow_line_threshold:
        score -= cfg.score_narrow_line_penalty

    return score, y_center, avg_h


def _select_subtitle_lines(
    lines: list[list[CharBox]],
    img_w: int,
    img_h: int,
    cfg: IsolatorConfig,
) -> list[list[CharBox]]:
    """Score and select the best subtitle lines.

    Two-pass approach:
    1. Score each line individually, pick the best one.
    2. Include nearby lines that look like part of the same subtitle
       block (similar char height, close vertical distance).
    """
    if not lines:
        return []

    # Build scored list: (score, core_boxes, y_center, avg_char_height)
    scored: list[tuple[float, list[CharBox], float, float]] = []

    for line in lines:
        core = _find_dense_core(line, cfg)
        if len(core) < cfg.min_char_boxes:
            continue

        score, y_center, avg_h = _score_line(core, img_w, img_h, cfg)
        scored.append((score, core, y_center, avg_h))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_line, best_y, best_h = scored[0]

    # Pass 1: always include the best line
    result: list[list[CharBox]] = [best_line]
    accepted_ys: list[float] = [best_y]

    # Pass 2: include nearby companion lines.
    # A companion line must:
    #   - have similar character height (within configured ratio)
    #   - be vertically close to ANY already-accepted line
    #   - have a minimum score threshold
    # Loop multiple times so line 3 can be accepted via proximity to line 2.
    remaining = list(scored[1:])
    changed = True
    while changed and len(result) < cfg.max_subtitle_lines:
        changed = False
        still_remaining = []
        for score, line, y_center, avg_h in remaining:
            h_ratio = avg_h / (best_h + 1)
            if h_ratio < cfg.companion_height_ratio_min or h_ratio > cfg.companion_height_ratio_max:
                still_remaining.append((score, line, y_center, avg_h))
                continue

            # Vertical proximity to nearest accepted line.
            # Subtitle line spacing (leading) is typically 1.2-1.5x font size,
            # but contour height < font em-height, so center-to-center distance
            # can be 2.2-2.5x contour height. Use configured multiplier for headroom.
            ref_h = max(avg_h, best_h)
            min_dist = min(abs(y_center - ay) for ay in accepted_ys)
            if min_dist > ref_h * cfg.companion_max_y_distance:
                still_remaining.append((score, line, y_center, avg_h))
                continue

            if score > cfg.score_companion_min:
                result.append(line)
                accepted_ys.append(y_center)
                changed = True
            else:
                still_remaining.append((score, line, y_center, avg_h))
        remaining = still_remaining

    result.sort(key=lambda line: float(np.mean([c[0][1] for c in line])))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def isolate_text(
    rgb: np.ndarray,
    config: IsolatorConfig | None = None,
) -> np.ndarray | None:
    """Isolate subtitle text from a screenshot.

    Returns a clean black-text-on-white RGB image, or None if no text found.

    Approach: find text lines -> crop around them -> apply a simple
    brightness threshold to separate text from background.
    No contour masks -- preserves full letter shapes.

    Args:
        rgb: Input image in RGB format (H x W x 3).
        config: Optional configuration override. Uses module defaults
                if not provided.
    """
    cfg = config or _default_config

    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        return None

    # Use RAW grayscale for contour detection -- brightness thresholds in
    # _find_char_boxes() are calibrated for unprocessed pixel values.
    # CLAHE/HDR enhancement is applied later, only on the cropped region.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if _is_blank_frame(gray, cfg):
        return None

    # Step 1: find character boxes (brightness-filtered)
    boxes, text_is_bright = _find_char_boxes(gray, cfg)
    if len(boxes) < cfg.min_char_boxes:
        return None

    # Step 2: cluster into lines
    all_lines = _cluster_into_lines(boxes, cfg)

    # Step 3: select subtitle lines
    sub_lines = _select_subtitle_lines(all_lines, w, h, cfg)
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

    pad_x = max(cfg.crop_pad_x_min, int((cx2 - cx1) * cfg.crop_pad_x_ratio))
    # Vertical padding based on character height to prevent clipping
    # descenders/ascenders and to catch nearby text.
    avg_char_h = float(np.mean([b[3] for b in all_rects]))
    pad_y = max(cfg.crop_pad_y_min, int(avg_char_h * cfg.crop_pad_y_ratio))
    cx1 = max(0, cx1 - pad_x)
    cx2 = min(w, cx2 + pad_x)
    cy1 = max(0, cy1 - pad_y)
    cy2 = min(h, cy2 + pad_y)

    cropped = gray[cy1:cy2, cx1:cx2]

    # Step 5: HDR-adaptive enhancement on the cropped region.
    # CLAHE is safe here because we already found char boxes using raw values.
    # It improves Otsu thresholding on difficult backgrounds (low contrast,
    # semi-transparent overlays, HDR game scenes).
    cropped = _enhance_gray(cropped, cfg)

    # Step 6: binarize using Otsu threshold
    if text_is_bright:
        # Text is bright (white subtitles on dark/semi-dark bg).
        # Otsu finds the boundary between background and text.
        _, binary = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # binary: text=255, background=0. Invert to get black-on-white.
        result = cv2.bitwise_not(binary)
    else:
        # Dark text on light background.
        _, binary = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # binary: text=0 (black), background=255 (white). Already correct.
        result = binary

    # Step 7: morphological cleanup -- remove small noise speckles
    k = cfg.morph_open_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
