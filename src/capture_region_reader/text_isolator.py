"""Isolate subtitle text from a screenshot before OCR.

Pipeline:
0. Dark box detection: if subtitles sit inside a dark rectangular bar
   (common in broadcasts/news), crop to that bar first to eliminate
   noisy multi-colored backgrounds before Otsu thresholding.
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

    # -- Dark subtitle box detection --
    # Some broadcasts place subtitles inside a dark/black rectangular bar.
    # When such a box is detected, the pipeline crops to it first, removing
    # the noisy multi-colored background that confuses Otsu thresholding.

    # Grayscale threshold to classify pixels as "dark" (0-255).
    # Pixels below this value are considered part of a potential dark box.
    dark_box_threshold: int = 50
    # Minimum area of the dark box as a fraction of total image area.
    # Subtitle bars are thin wide strips -- typically 5-15% of image area.
    # Set low enough to catch thin bars, but combined with aspect ratio
    # and width checks, this still rejects small noise patches.
    dark_box_min_area_ratio: float = 0.05
    # Minimum aspect ratio (width / height) of the dark box.
    # Subtitle bars are wider than tall. 2.0 means at least 2x wider.
    dark_box_min_aspect_ratio: float = 2.0
    # Minimum width of the dark box as a fraction of image width.
    # Ensures the box spans a meaningful portion of the screen.
    dark_box_min_width_ratio: float = 0.30
    # Maximum standard deviation of pixel values inside the dark box.
    # A true dark subtitle bar has uniformly dark pixels (low std).
    # Higher values are more permissive (allow gradient backgrounds).
    dark_box_max_internal_std: float = 35.0
    # Padding around the detected dark box as a fraction of box dimensions.
    # Default 0 because adding padding pulls in non-dark pixels from the
    # surrounding background, which confuses Otsu thresholding inside the
    # cropped region. The character detection pipeline already applies its
    # own crop padding in a later step. Only increase this if text is
    # being clipped at the very edges of the dark bar.
    dark_box_padding_ratio: float = 0.0

    # -- Color box detection (box_search mode) --
    # Generic version of dark_box detection: finds a rectangular region whose
    # background matches a user-selected color. Used for YouTube subtitles,
    # news tickers, or any colored subtitle bar.
    #
    # box_search_color: target RGB color (set by user via color picker)
    box_search_color: tuple[int, int, int] | None = None
    # How far from the target color (in Euclidean RGB distance) a pixel
    # can be and still count as "matching". 0=exact match, 60=default.
    box_search_tolerance: int = 60
    # Minimum area as fraction of total image area.
    box_search_min_area_ratio: float = 0.03
    # Minimum aspect ratio (width / height).
    box_search_min_aspect_ratio: float = 1.5
    # Minimum width as fraction of image width.
    box_search_min_width_ratio: float = 0.25
    # Maximum internal standard deviation (uniformity check on matching pixels).
    box_search_max_internal_std: float = 40.0
    # Padding around the detected box (fraction of box dimensions).
    box_search_padding_ratio: float = 0.0

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
# Dark subtitle box detection
# ---------------------------------------------------------------------------

def _find_dark_subtitle_box(
    gray: np.ndarray,
    cfg: IsolatorConfig,
) -> tuple[int, int, int, int] | None:
    """Detect a dark rectangular subtitle bar in the image.

    Broadcasts and news programs often place subtitles inside a dark/black
    rectangular box overlaid on a complex background. When such a box is
    present, the standard Otsu thresholding on the full image picks up
    noise from logos, colored bars, and photos, drowning out the actual
    subtitle text.

    This function looks for a large, wide, uniformly dark rectangle. If
    found, isolate_text() can crop to it and run the character detection
    pipeline on a much cleaner input.

    Returns (x, y, w, h) of the dark box bounding rectangle, or None if
    no suitable dark box is found.
    """
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    # Step 1: threshold to find very dark regions.
    # Everything below dark_box_threshold becomes white (foreground),
    # everything above becomes black (background).
    _, dark_mask = cv2.threshold(
        gray, cfg.dark_box_threshold, 255, cv2.THRESH_BINARY_INV
    )

    # Step 2: morphological closing to merge nearby dark patches into
    # a single solid region. Without this, thin bright text inside the
    # dark box creates holes that split the contour.
    close_k = max(5, img_w // 80)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_k, close_k)
    )
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)

    # Step 3: find contours of dark regions
    contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Step 4: evaluate each contour against the criteria.
    # We want the largest qualifying contour (by area).
    min_area = img_area * cfg.dark_box_min_area_ratio
    min_width = img_w * cfg.dark_box_min_width_ratio

    best_candidate: tuple[int, int, int, int] | None = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h

        # Size check: must be a significant portion of the image
        if box_area < min_area:
            continue

        # Shape check: must be wider than tall (subtitle bar shape)
        if h == 0:
            continue
        aspect = w / h
        if aspect < cfg.dark_box_min_aspect_ratio:
            continue

        # Width check: must span a meaningful portion of the screen
        if w < min_width:
            continue

        # Uniformity check: the dark BACKGROUND pixels inside the bounding
        # box should have consistently low brightness (low std). We mask
        # out bright pixels (text) before computing std, because bright
        # subtitle text inside the box would inflate std and cause a
        # false rejection.
        roi = gray[y : y + h, x : x + w]
        dark_pixels = roi[roi <= cfg.dark_box_threshold]
        # If less than 40% of the box is dark, it's not a true dark bar
        if len(dark_pixels) < 0.4 * roi.size:
            continue
        internal_std = float(np.std(dark_pixels))
        if internal_std > cfg.dark_box_max_internal_std:
            continue

        # This contour passes all criteria. Keep if it's the largest.
        if box_area > best_area:
            best_area = box_area
            best_candidate = (x, y, w, h)

    return best_candidate


def _crop_to_dark_box(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    cfg: IsolatorConfig,
) -> tuple[np.ndarray, int, int]:
    """Crop grayscale image to the dark box region with padding.

    Returns (cropped_gray, offset_x, offset_y) where offsets are the
    top-left corner of the crop in the original image coordinate space.
    These offsets are needed if the caller needs to map coordinates back
    to the original image.
    """
    img_h, img_w = gray.shape
    bx, by, bw, bh = box

    pad_x = int(bw * cfg.dark_box_padding_ratio)
    pad_y = int(bh * cfg.dark_box_padding_ratio)

    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(img_w, bx + bw + pad_x)
    y2 = min(img_h, by + bh + pad_y)

    return gray[y1:y2, x1:x2], x1, y1


# ---------------------------------------------------------------------------
# Color box detection (box_search mode — universal, any background color)
# ---------------------------------------------------------------------------

def _max_rectangle_in_binary(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the largest axis-aligned rectangle of 1s in a binary matrix.

    Uses the histogram-stack algorithm: O(rows * cols).
    Height computation is vectorized with numpy; the monotonic stack
    runs per-row in Python (stack operations don't vectorize well).

    Returns (x, y, w, h) of the largest rectangle, or None if empty.
    """
    if binary.size == 0:
        return None

    rows, cols = binary.shape

    # Pre-compute all histogram heights using numpy (vectorized).
    # heights[r, c] = number of consecutive 1s ending at row r in column c.
    all_heights = np.zeros((rows, cols), dtype=np.int32)
    all_heights[0] = binary[0]
    for r in range(1, rows):
        all_heights[r] = np.where(binary[r], all_heights[r - 1] + 1, 0)

    best_area = 0
    best_rect: tuple[int, int, int, int] | None = None

    for r in range(rows):
        heights = all_heights[r]

        # Largest rectangle in histogram (monotonic stack)
        stack: list[int] = []
        for c in range(cols + 1):
            h = int(heights[c]) if c < cols else 0
            while stack and int(heights[stack[-1]]) > h:
                height = int(heights[stack.pop()])
                width = c if not stack else c - stack[-1] - 1
                area = height * width
                if area > best_area:
                    best_area = area
                    x = stack[-1] + 1 if stack else 0
                    y = r - height + 1
                    best_rect = (int(x), int(y), int(width), int(height))
            stack.append(c)

    return best_rect


def _find_colored_subtitle_box(
    rgb: np.ndarray,
    cfg: IsolatorConfig,
) -> tuple[int, int, int, int] | None:
    """Detect a rectangular subtitle bar with a user-specified background color.

    Algorithm:
    1. Create binary mask: pixels within tolerance of cfg.box_search_color.
    2. Large morphological closing to fill text-sized holes inside the box.
       Text characters are NOT the target color, so they create gaps in the
       mask. Closing bridges those gaps so the box becomes a solid filled
       rectangle.
    3. Downscale the closed mask (for speed) and find the largest inscribed
       rectangle using the histogram-stack O(n*m) algorithm.
    4. Scale coordinates back to original resolution.

    Works for any background color: black, yellow, blue, semi-transparent
    overlays, etc. — as long as the user has picked the correct color.

    Returns (x, y, w, h) of the colored box, or None if not found.
    """
    if cfg.box_search_color is None:
        return None

    img_h, img_w = rgb.shape[:2]

    target = np.array(cfg.box_search_color, dtype=np.float32)
    tolerance = cfg.box_search_tolerance

    # Step 1: Euclidean RGB distance → binary mask.
    rgb_f = rgb.astype(np.float32)
    dist = np.sqrt(np.sum((rgb_f - target) ** 2, axis=2))
    mask = (dist <= tolerance).astype(np.uint8) * 255

    # Step 2: Large morphological closing to fill text holes.
    # Subtitle text is typically 10-20% of the box height. The closing
    # kernel must be large enough to bridge character-sized gaps.
    close_k = max(15, img_h // 6)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_k, close_k)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Step 3: Downscale for fast max-rectangle search.
    # The histogram-stack algo is O(n*m) with Python loops, so we
    # reduce resolution to keep it under ~20ms. Scale factor ≤ 4.
    max_dim = max(img_h, img_w)
    scale = max(1, max_dim // 500)  # target ~500px on longest side
    scale = min(scale, 4)

    if scale > 1:
        small_h = img_h // scale
        small_w = img_w // scale
        small = cv2.resize(closed, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        small = closed
        small_h, small_w = img_h, img_w

    rect = _max_rectangle_in_binary((small > 0).astype(np.uint8))

    if rect is None:
        return None

    # Scale coordinates back to original resolution.
    sx, sy, sw, sh = rect
    x = sx * scale
    y = sy * scale
    w = sw * scale
    h = sh * scale

    # Clamp to image bounds
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    # Validate: the rectangle must be a meaningful subtitle box.
    img_area = img_h * img_w
    if w * h < img_area * cfg.box_search_min_area_ratio:
        return None
    if h == 0:
        return None
    if w / h < cfg.box_search_min_aspect_ratio:
        return None
    if w < img_w * cfg.box_search_min_width_ratio:
        return None

    return (x, y, w, h)


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
    mode: str = "default",
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
        mode: Isolation mode. "default" uses the standard pipeline with
              automatic dark-box detection. "box_search" uses the
              user-selected color to find and crop to a subtitle box.
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

    if mode == "box_search":
        # Box search mode: find a subtitle box matching the user-picked color
        # and return ONLY the cropped rectangle — no binarization, no character
        # detection, no further processing. The raw cropped image goes straight
        # to OCR (e.g., white text on black background is already ideal for
        # Tesseract without any Otsu thresholding).
        color_box = _find_colored_subtitle_box(rgb, cfg)
        if color_box is not None:
            bx, by, bw, bh = color_box
            cropped = rgb[by : by + bh, bx : bx + bw]
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return None
            # Return the cropped RGB directly — no further pipeline steps.
            return cropped.copy()
        # Box not found — return None (no text to process)
        return None
    else:
        # Default mode: dark subtitle box detection.
        # Broadcasts and news programs place subtitles in a dark rectangular
        # bar on a complex multi-colored background. Otsu on the full image
        # picks up noise from logos, red bars, photos, etc. If we detect such
        # a box, we crop to it first so the character detection pipeline runs
        # on a clean, uniformly dark background.
        dark_box = _find_dark_subtitle_box(gray, cfg)
        if dark_box is not None:
            gray, _, _ = _crop_to_dark_box(gray, dark_box, cfg)
            # Update dimensions to match the cropped region.
            # All subsequent steps operate on this smaller image.
            h, w = gray.shape

            if h < 10 or w < 10:
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
