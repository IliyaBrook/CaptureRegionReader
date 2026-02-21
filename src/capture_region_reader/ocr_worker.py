"""OCR worker thread: captures screen region, runs OCR, emits recognized text.

Handles:
- Screen capture via mss
- Image preprocessing (text isolation via 3-step pipeline)
- OCR via Tesseract
- OCR garbage filtering (removes non-text noise before emitting)
- Upscaling for small capture regions

The worker emits raw OCR text (after garbage filtering). Text deduplication
and change detection are handled by TextDiffer in the app layer.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytesseract
from mss import mss
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from capture_region_reader.text_isolator import isolate_text

logger = logging.getLogger(__name__)



# Minimum dimensions for Tesseract OCR accuracy.
# Small capture regions produce blurry characters after Otsu thresholding.
MIN_WIDTH = 600
MIN_HEIGHT = 100


# ---------------------------------------------------------------------------
# OCR engine (Tesseract only)
# ---------------------------------------------------------------------------

_TESSERACT_CONFIG = "--psm 6 --oem 1"


def _recognize(image: Image.Image, language: str) -> str:
    """Run Tesseract OCR on a PIL image.

    Uses PSM 6 (single uniform block of text) and OEM 1 (LSTM neural net).
    """
    return pytesseract.image_to_string(
        image, lang=language, config=_TESSERACT_CONFIG
    ).strip()


# ---------------------------------------------------------------------------
# Garbage filtering
# ---------------------------------------------------------------------------

# Minimum alphanumeric-to-total ratio for a line to be considered text.
_GARBAGE_ALNUM_RATIO = 0.4
# Minimum alphabetic characters per line.
_GARBAGE_MIN_ALPHA = 2
# Maximum ratio of mixed-script words (Cyrillic+Latin in same word).
_GARBAGE_MIXED_SCRIPT_RATIO = 0.3
# Maximum ratio of very short words (<=2 chars) in lines with >=3 words.
_GARBAGE_SHORT_WORD_RATIO = 0.6
# Minimum unique character ratio (catches "aaaaaaa" garbage).
_GARBAGE_MIN_UNIQUE_RATIO = 0.15
# Minimum characters to run unique ratio check.
_GARBAGE_MIN_CHARS_FOR_UNIQUE = 4
# Maximum total words for average-word-length check.
_GARBAGE_SHORT_RESULT_MAX_WORDS = 3
# Minimum average word length for short results.
_GARBAGE_SHORT_RESULT_MIN_AVG_LEN = 3


def _is_mixed_script_word(word: str) -> bool:
    """Check if a word mixes Cyrillic and Latin characters."""
    cyr = sum(1 for c in word if "\u0400" <= c <= "\u052f")
    lat = sum(1 for c in word if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    return cyr > 0 and lat > 0 and (cyr + lat) >= 3


def filter_ocr_garbage(text: str) -> str:
    """Filter out OCR garbage lines.

    Applies multiple heuristics to detect and remove lines that are
    OCR noise rather than real text:
    1. Low alphanumeric ratio (mostly symbols/punctuation)
    2. Too few alphabetic characters
    3. Mixed Cyrillic/Latin within same word (always OCR error)
    4. Too many very short words (fragmented noise)
    5. Low character diversity (repeated characters)
    6. Final check: very short results with tiny average word length
    """
    lines = text.split("\n")
    good_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check 1: alphanumeric ratio
        alnum = sum(1 for c in line if c.isalnum())
        non_space = len(line.replace(" ", ""))
        if non_space == 0 or alnum / non_space < _GARBAGE_ALNUM_RATIO:
            continue

        # Check 2: minimum alphabetic characters
        alpha = sum(1 for c in line if c.isalpha())
        if alpha < _GARBAGE_MIN_ALPHA:
            continue

        # Check 3: mixed-script words
        words = line.split()
        if len(words) > 0:
            mixed_count = sum(1 for w in words if _is_mixed_script_word(w))
            if mixed_count / len(words) > _GARBAGE_MIXED_SCRIPT_RATIO:
                continue

        # Check 4: too many short words
        if len(words) >= 3:
            short_words = sum(1 for w in words if len(w) <= 2)
            if short_words / len(words) > _GARBAGE_SHORT_WORD_RATIO:
                continue

        # Check 5: character diversity
        alpha_chars = [c.lower() for c in line if c.isalpha()]
        if len(alpha_chars) >= _GARBAGE_MIN_CHARS_FOR_UNIQUE:
            unique_ratio = len(set(alpha_chars)) / len(alpha_chars)
            if unique_ratio < _GARBAGE_MIN_UNIQUE_RATIO:
                continue

        good_lines.append(line)

    result = "\n".join(good_lines)

    # Final check: very short results with tiny average word length
    # are likely fragmented noise.
    if result:
        all_words = result.split()
        if len(all_words) <= _GARBAGE_SHORT_RESULT_MAX_WORDS:
            avg_word_len = sum(len(w) for w in all_words) / len(all_words)
            if avg_word_len < _GARBAGE_SHORT_RESULT_MIN_AVG_LEN:
                return ""

    return result


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _upscale(img: Image.Image) -> Image.Image:
    """Upscale image to minimum dimensions needed for Tesseract OCR accuracy.

    Small capture regions produce blurry characters. This ensures
    the image is at least MIN_WIDTH x MIN_HEIGHT, scaling by an integer
    factor between 2x and 4x.
    """
    w, h = img.size
    scale = 1
    if w < MIN_WIDTH:
        scale = max(scale, (MIN_WIDTH + w - 1) // w)
    if h < MIN_HEIGHT:
        scale = max(scale, (MIN_HEIGHT + h - 1) // h)
    scale = max(scale, 2)
    scale = min(scale, 4)
    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# OCR Worker thread
# ---------------------------------------------------------------------------

class OcrWorker(QThread):
    """Background thread that captures screen region and runs OCR in a loop.

    Signals:
    - text_recognized(str): emitted when OCR produces non-garbage text
    - frame_captured(bytes, int, int): preview image (RGB bytes, width, height)
    - error_occurred(str): error message for UI display
    """
    text_recognized = pyqtSignal(str)
    frame_captured = pyqtSignal(bytes, int, int)
    raw_frame_captured = pyqtSignal(bytes, int, int)  # original RGB (before isolation)
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._region: tuple[int, int, int, int] | None = None
        self._language: str = "eng+rus"
        self._interval_ms: int = 500
        self._running: bool = False
        self._capture_count: int = 0

    def configure(
        self,
        region: tuple[int, int, int, int],
        language: str,
        interval_ms: int,
    ) -> None:
        """Configure capture region and OCR parameters.

        Called before start() or when settings change while running.
        """
        self._region = region
        self._language = language
        self._interval_ms = interval_ms
        self._capture_count = 0
        logger.info(
            "OCR configured: region=(%d,%d,%d,%d), lang=%s",
            *region, language,
        )

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def run(self) -> None:
        """Main OCR loop: capture -> preprocess -> recognize -> emit."""
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
                    self._process_frame(sct, monitor)
                except Exception as e:
                    logger.error("OCR frame error: %s", e, exc_info=True)
                    self.error_occurred.emit(str(e))

                self.msleep(self._interval_ms)

    def _process_frame(self, sct, monitor: dict) -> None:
        """Process a single capture frame through the OCR pipeline."""
        screenshot = sct.grab(monitor)
        img_array = np.array(screenshot, dtype=np.uint8)

        # BGRA -> RGB
        raw_rgb = img_array[:, :, :3][:, :, ::-1].copy()

        # Always emit the raw frame (used by eyedropper for color picking)
        raw_h, raw_w = raw_rgb.shape[:2]
        self.raw_frame_captured.emit(raw_rgb.tobytes(), raw_w, raw_h)

        # Text isolation: detect subtitle region, binarize, clean artifacts
        isolated = isolate_text(raw_rgb)

        if isolated is None:
            logger.debug(
                "isolate_text returned None for frame %d (input shape=%s), "
                "falling back to raw image for OCR",
                self._capture_count, raw_rgb.shape,
            )
            # Fallback: use raw image directly for OCR (e.g. black text on white)
            ocr_img = _upscale(Image.fromarray(raw_rgb))
            # Show raw as the processed preview (no isolation was applied)
            p_h, p_w = raw_rgb.shape[:2]
            self.frame_captured.emit(raw_rgb.tobytes(), p_w, p_h)
        else:
            ocr_img = _upscale(Image.fromarray(isolated))
            # Send the processed (isolated) image to preview
            preview_rgb = np.array(ocr_img)
            p_h, p_w = preview_rgb.shape[:2]
            self.frame_captured.emit(preview_rgb.tobytes(), p_w, p_h)

        self._capture_count += 1

        # Run OCR
        t0 = time.monotonic()
        text = _recognize(ocr_img, self._language)
        ocr_ms = int((time.monotonic() - t0) * 1000)

        logger.debug(
            "OCR (%dms) raw: %s",
            ocr_ms, repr(text[:200] if text else ""),
        )

        # Filter garbage and emit
        if text:
            filtered = filter_ocr_garbage(text)
            if filtered != text:
                logger.debug("OCR after filter: %s", repr(filtered[:200] if filtered else "<empty>"))
            text = filtered

        # Always emit (even empty) so TextDiffer can track text disappearance.
        self.text_recognized.emit(text)

    def stop(self) -> None:
        """Stop the OCR loop and wait for thread to finish."""
        self._running = False
        self.wait(3000)
