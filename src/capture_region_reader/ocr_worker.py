"""OCR worker thread: captures screen region, runs OCR, emits recognized text.

Supports pluggable OCR engines (Tesseract, EasyOCR) and handles:
- Screen capture via mss
- Image preprocessing (text isolation for Tesseract, HDR enhancement for EasyOCR)
- OCR garbage filtering (removes non-text noise before emitting)
- Upscaling for small capture regions

The worker emits raw OCR text (after garbage filtering). Text deduplication
and change detection are handled by TextDiffer in the app layer.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Protocol

import numpy as np
import pytesseract
from mss import mss
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from capture_region_reader.text_isolator import isolate_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum dimensions for Tesseract OCR accuracy.
# Small capture regions produce blurry characters after Otsu thresholding.
MIN_WIDTH = 600
MIN_HEIGHT = 100

# Minimum dimensions for EasyOCR (from RSTGameTranslation).
EASYOCR_MIN_WIDTH = 1024
EASYOCR_MIN_HEIGHT = 768

# Debug: save captures to .tests/debug/ for inspection.
DEBUG_SAVE = os.environ.get("CRR_DEBUG", "0") == "1"
_DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".tests", "debug")


# ---------------------------------------------------------------------------
# OCR engine abstraction
# ---------------------------------------------------------------------------

class OcrEngine(Protocol):
    """Protocol for OCR engines."""
    def recognize(self, image: Image.Image, language: str) -> str: ...


class TesseractEngine:
    """Tesseract OCR engine (default, lightweight).

    Uses PSM 6 (single uniform block of text) and OEM 1 (LSTM neural net).
    """

    def recognize(self, image: Image.Image, language: str) -> str:
        return pytesseract.image_to_string(
            image, lang=language, config="--psm 6 --oem 1"
        ).strip()


class EasyOcrEngine:
    """EasyOCR engine with GPU support (optional, requires easyocr package).

    EasyOCR has its own CRAFT text detection model so it works best with
    the original (non-binarized) image. We apply HDR-adaptive preprocessing
    but skip the text_isolator pipeline.
    """

    # EasyOCR works on raw images, not binarized text_isolator output.
    needs_text_isolation = False

    _LANG_MAP = {
        "eng": ["en"],
        "rus": ["ru"],
        "eng+rus": ["en", "ru"],
    }

    def __init__(self) -> None:
        self._reader = None
        self._current_langs: list[str] | None = None

    def recognize(self, image: Image.Image, language: str) -> str:
        langs = self._LANG_MAP.get(language, [language])

        if self._reader is None or self._current_langs != langs:
            import easyocr
            import torch

            gpu = torch.cuda.is_available()
            logger.info("EasyOCR init: langs=%s, gpu=%s", langs, gpu)
            self._reader = easyocr.Reader(langs, gpu=gpu)
            self._current_langs = langs

        img_array = np.array(image)
        results = self._reader.readtext(img_array, detail=0)
        return "\n".join(results).strip()


def create_engine(name: str) -> OcrEngine:
    """Create an OCR engine by name. Raises ImportError for unavailable engines."""
    if name == "easyocr":
        try:
            import easyocr  # noqa: F401
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. Install with: "
                "uv pip install easyocr torch"
            )
        return EasyOcrEngine()
    return TesseractEngine()


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


def _preprocess_for_easyocr(raw_rgb: np.ndarray) -> Image.Image:
    """Preprocess image for EasyOCR following RSTGameTranslation approach.

    Pipeline:
    1. HDR-adaptive enhancement (CLAHE + bilateral + sharpen) on grayscale
       OR basic contrast + median filter
    2. Convert back to RGB (EasyOCR accepts both but RGB is standard)
    3. Upscale to minimum 1024x768 with LANCZOS

    No text isolation or contour detection -- EasyOCR has its own
    CRAFT text detector that works best on natural/enhanced images.
    """
    import cv2
    from PIL import ImageEnhance, ImageFilter

    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))

    needs_enhanced = std_br < 40 or mean_br > 200 or mean_br < 55

    if needs_enhanced:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel=kernel)
        result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(result_rgb)
    else:
        img = Image.fromarray(raw_rgb).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.convert("RGB")

    # Upscale to EasyOCR minimum dimensions
    w, h = img.size
    if w < EASYOCR_MIN_WIDTH or h < EASYOCR_MIN_HEIGHT:
        scale = max(EASYOCR_MIN_WIDTH / w, EASYOCR_MIN_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._region: tuple[int, int, int, int] | None = None
        self._language: str = "eng+rus"
        self._interval_ms: int = 500
        self._running: bool = False
        self._capture_count: int = 0
        self._engine: OcrEngine | None = None
        self._engine_name: str = ""

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
            "OCR configured: region=(%d,%d,%d,%d), lang=%s, engine=%s",
            *region, language, self._engine_name,
        )

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def set_engine(self, engine_name: str) -> None:
        """Switch OCR engine. Safe to call while running."""
        if engine_name == self._engine_name:
            return
        try:
            old_name = self._engine_name
            self._engine = create_engine(engine_name)
            self._engine_name = engine_name
            logger.info("OCR engine switched: %s -> %s", old_name, engine_name)
        except ImportError as e:
            logger.error("Failed to switch to %s: %s", engine_name, e)
            self.error_occurred.emit(str(e))
            if self._engine is None:
                self._engine = TesseractEngine()
                self._engine_name = "tesseract"
                logger.info("Fallback: using Tesseract")

    def run(self) -> None:
        """Main OCR loop: capture -> preprocess -> recognize -> emit."""
        self._running = True
        if self._engine is None:
            self._engine = TesseractEngine()
            self._engine_name = "tesseract"
            logger.warning("Engine was None at run(), defaulting to Tesseract")

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

        # Choose preprocessing path based on engine
        use_isolation = getattr(self._engine, "needs_text_isolation", True)

        if use_isolation:
            # Tesseract path: text_isolator creates clean black-on-white image
            isolated = isolate_text(raw_rgb)

            if isolated is None:
                p_h, p_w = raw_rgb.shape[:2]
                self.frame_captured.emit(raw_rgb.tobytes(), p_w, p_h)
                # Emit empty string so TextDiffer knows text disappeared
                self.text_recognized.emit("")
                self.msleep(self._interval_ms)
                return

            ocr_img = _upscale(Image.fromarray(isolated))
        else:
            # EasyOCR path: HDR enhancement + upscale
            ocr_img = _preprocess_for_easyocr(raw_rgb)

        # Send the processed image to preview
        preview_rgb = np.array(ocr_img)
        p_h, p_w = preview_rgb.shape[:2]
        self.frame_captured.emit(preview_rgb.tobytes(), p_w, p_h)

        # Debug saves
        if DEBUG_SAVE and self._capture_count < 20:
            os.makedirs(_DEBUG_DIR, exist_ok=True)
            Image.fromarray(raw_rgb).save(
                os.path.join(_DEBUG_DIR, f"crr_raw_{self._capture_count}.png")
            )
            ocr_img.save(
                os.path.join(_DEBUG_DIR, f"crr_isolated_{self._capture_count}.png")
            )

        self._capture_count += 1

        # Run OCR
        t0 = time.monotonic()
        text = self._engine.recognize(ocr_img, self._language)
        ocr_ms = int((time.monotonic() - t0) * 1000)

        logger.debug(
            "OCR (%dms) %s raw: %s",
            ocr_ms, self._engine_name, repr(text[:200] if text else ""),
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
