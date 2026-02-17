from __future__ import annotations

import os
import numpy as np
import pytesseract
from mss import mss
from PIL import Image, ImageEnhance, ImageFilter
from PyQt6.QtCore import QThread, pyqtSignal


# Minimum dimensions for good Tesseract accuracy
MIN_WIDTH = 600
MIN_HEIGHT = 100

# Debug: save captures to /tmp for inspection
DEBUG_SAVE = os.environ.get("CRR_DEBUG", "0") == "1"


def _find_text_bands(rgb: np.ndarray) -> list[tuple[int, int]]:
    """Find horizontal bands that contain text (bright or high-contrast).

    Returns list of (row_start, row_end) slices, each representing
    a group of consecutive rows that likely contain text.
    """
    h, w = rgb.shape[:2]
    if h < 10 or w < 10:
        return []

    gray = np.mean(rgb, axis=2)

    # For each row, compute local contrast (std deviation)
    row_std = np.std(gray, axis=1)

    # Rows with bright pixels (potential white/bright text)
    bright_mask = gray > 180
    bright_fraction = np.mean(bright_mask, axis=1)

    # A "text row" has high contrast AND some bright pixels
    text_rows = (row_std > 30) & (bright_fraction > 0.01)

    # Group consecutive text rows into bands, allowing small gaps
    bands: list[tuple[int, int]] = []
    in_band = False
    band_start = 0
    gap = 0
    max_gap = 5

    for i in range(h):
        if text_rows[i]:
            if not in_band:
                band_start = i
                in_band = True
            gap = 0
        elif in_band:
            gap += 1
            if gap > max_gap:
                band_end = i - gap
                if band_end - band_start >= 8:
                    bands.append((band_start, band_end))
                in_band = False
                gap = 0

    if in_band:
        band_end = h - gap if gap else h
        if band_end - band_start >= 8:
            bands.append((band_start, band_end))

    return bands


def _extract_subtitle_region(rgb: np.ndarray) -> np.ndarray | None:
    """Extract the subtitle text region from the image.

    Subtitles are almost always in the bottom portion of the frame.
    This function finds text bands and picks only those in the bottom ~45%
    of the image, which is where subtitles appear in movies, YouTube,
    games, etc.

    If multiple bands exist in the bottom area (multi-line subtitles),
    they are merged. Bands in the upper part of the image (scene text,
    titles, logos, news tickers at top) are ignored.

    Returns cropped RGB array of the subtitle area, or None
    if no suitable region was found.
    """
    h, w = rgb.shape[:2]
    bands = _find_text_bands(rgb)
    if not bands:
        return None

    # Subtitle zone: bottom 45% of the image
    # This covers subtitles in all common positions while excluding
    # scene titles, channel logos, upper text overlays
    subtitle_zone_top = int(h * 0.55)

    # Filter bands to only those in the subtitle zone
    bottom_bands = []
    for band_start, band_end in bands:
        band_center = (band_start + band_end) / 2
        if band_center >= subtitle_zone_top:
            bottom_bands.append((band_start, band_end))

    if not bottom_bands:
        # No text in subtitle zone — try all bands but only if there's
        # a single band (could be a centered subtitle or small region)
        if len(bands) == 1:
            bottom_bands = bands
        else:
            return None

    # Merge bottom bands (multi-line subtitles)
    row_start = bottom_bands[0][0]
    row_end = bottom_bands[-1][1]

    # Add vertical padding
    pad = 6
    row_start = max(0, row_start - pad)
    row_end = min(h, row_end + pad)

    cropped = rgb[row_start:row_end, :, :]

    # Only return if actually smaller than original
    if cropped.shape[0] < rgb.shape[0] * 0.75:
        return cropped

    return None


def _filter_ocr_garbage(text: str) -> str:
    """Filter out OCR garbage lines.

    Removes lines that are likely misrecognized noise:
    - Lines with too few alphanumeric characters
    - Lines where single words mix Cyrillic and Latin (OCR confusion)
    - Lines that are mostly digits and punctuation
    """
    lines = text.split("\n")
    good_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Basic quality: must have enough alphanumeric chars
        alnum = sum(1 for c in line if c.isalnum())
        non_space = len(line.replace(" ", ""))
        if non_space == 0 or alnum / non_space < 0.4:
            continue

        # Must have at least 2 alphabetic chars
        alpha = sum(1 for c in line if c.isalpha())
        if alpha < 2:
            continue

        # Check for mixed-script words (strong signal of OCR garbage)
        # E.g., "ОРЕБРТ" is pure Cyrillic but "МОвЕОМЕМ$" has mixed chars
        words = line.split()
        garbage_words = 0
        for word in words:
            cyr = sum(1 for c in word if "\u0400" <= c <= "\u04FF")
            lat = sum(1 for c in word if ("A" <= c <= "Z") or ("a" <= c <= "z"))
            word_alpha = cyr + lat
            # A word with both Cyrillic AND Latin is likely OCR garbage
            if cyr > 0 and lat > 0 and word_alpha >= 3:
                garbage_words += 1

        # If more than half the words are garbage, skip the line
        if len(words) > 0 and garbage_words / len(words) > 0.3:
            continue

        good_lines.append(line)

    return "\n".join(good_lines)


class OcrWorker(QThread):
    text_recognized = pyqtSignal(str)
    frame_captured = pyqtSignal(bytes, int, int)  # raw RGB bytes, width, height
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._region: tuple[int, int, int, int] | None = None
        self._language: str = "eng+rus"
        self._interval_ms: int = 500
        self._running: bool = False
        self._capture_count: int = 0
        self._subtitle_mode: bool = False
        self._last_emitted_text: str = ""  # dedup: don't emit identical text

    def configure(
        self,
        region: tuple[int, int, int, int],
        language: str,
        interval_ms: int,
    ) -> None:
        self._region = region
        self._language = language
        self._interval_ms = interval_ms
        self._capture_count = 0
        self._last_emitted_text = ""
        print(f"[OCR] Configured region: left={region[0]}, top={region[1]}, "
              f"width={region[2]}, height={region[3]}, lang={language}, "
              f"subtitle_mode={self._subtitle_mode}")

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def set_subtitle_mode(self, enabled: bool) -> None:
        self._subtitle_mode = enabled

    def _preprocess(self, rgb: np.ndarray) -> Image.Image:
        """Convert RGB array to optimized image for OCR."""
        img = Image.fromarray(rgb)
        w, h = img.size

        # Upscale to ensure minimum dimensions
        scale = 1
        if w < MIN_WIDTH:
            scale = max(scale, (MIN_WIDTH + w - 1) // w)
        if h < MIN_HEIGHT:
            scale = max(scale, (MIN_HEIGHT + h - 1) // h)
        scale = max(scale, 2)
        scale = min(scale, 4)

        if scale > 1:
            img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

        # Convert to grayscale
        gray = img.convert("L")
        gray_arr = np.array(gray, dtype=np.float32)

        threshold = 180

        # Check if image has bright text or dark text
        bright_pixels = np.sum(gray_arr > threshold)
        dark_pixels = np.sum(gray_arr < 80)
        total_pixels = gray_arr.size

        if bright_pixels / total_pixels > 0.02 and bright_pixels < total_pixels * 0.5:
            # Bright text on dark background (subtitles)
            binary = np.where(gray_arr > threshold, 0, 255).astype(np.uint8)
            img = Image.fromarray(binary, mode="L")
        elif dark_pixels / total_pixels > 0.02 and dark_pixels < total_pixels * 0.5:
            # Dark text on light background
            binary = np.where(gray_arr < 80, 0, 255).astype(np.uint8)
            img = Image.fromarray(binary, mode="L")
        else:
            enhancer = ImageEnhance.Contrast(gray)
            img = enhancer.enhance(2.0)

        img = img.filter(ImageFilter.SHARPEN)
        return img

    def run(self) -> None:
        self._running = True
        with mss() as sct:
            while self._running:
                if not self._region:
                    self.msleep(100)
                    continue

                left, top, width, height = self._region
                monitor = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }

                try:
                    screenshot = sct.grab(monitor)
                    img_array = np.array(screenshot, dtype=np.uint8)

                    # BGRA → RGB
                    raw_rgb = img_array[:, :, :3][:, :, ::-1].copy()
                    h_px, w_px = raw_rgb.shape[:2]
                    self.frame_captured.emit(raw_rgb.tobytes(), w_px, h_px)

                    # Subtitle mode: try to isolate subtitle background region
                    ocr_rgb = raw_rgb
                    if self._subtitle_mode:
                        sub_region = _extract_subtitle_region(raw_rgb)
                        if sub_region is not None:
                            ocr_rgb = sub_region

                    # Debug saves
                    if DEBUG_SAVE and self._capture_count < 5:
                        Image.fromarray(raw_rgb).save(
                            f"/tmp/crr_raw_{self._capture_count}.png"
                        )
                        if self._subtitle_mode:
                            Image.fromarray(ocr_rgb).save(
                                f"/tmp/crr_subtitle_{self._capture_count}.png"
                            )

                    processed = self._preprocess(ocr_rgb)

                    if DEBUG_SAVE and self._capture_count < 5:
                        processed.save(
                            f"/tmp/crr_processed_{self._capture_count}.png"
                        )
                        if self._subtitle_mode and ocr_rgb is not raw_rgb:
                            Image.fromarray(ocr_rgb).save(
                                f"/tmp/crr_subtitle_crop_{self._capture_count}.png"
                            )
                        print(
                            f"[OCR] Saved debug captures #{self._capture_count}"
                        )

                    self._capture_count += 1

                    ocr_lang = self._language

                    # PSM 6 = uniform text block (best for subtitles)
                    # PSM 3 = fully automatic (better for mixed content)
                    psm = 6 if self._subtitle_mode else 3

                    text = pytesseract.image_to_string(
                        processed,
                        lang=ocr_lang,
                        config=f"--psm {psm} --oem 1",
                    ).strip()

                    # Filter out garbage and deduplicate
                    if text:
                        text = _filter_ocr_garbage(text)
                        if text and text != self._last_emitted_text:
                            self._last_emitted_text = text
                            self.text_recognized.emit(text)
                except Exception as e:
                    self.error_occurred.emit(str(e))

                self.msleep(self._interval_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
