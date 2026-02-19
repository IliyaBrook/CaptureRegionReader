from __future__ import annotations

import os
import re
import time
from typing import Protocol

import numpy as np
import pytesseract
from mss import mss
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from capture_region_reader.text_differ import is_text_similar
from capture_region_reader.text_isolator import isolate_text


# Minimum dimensions for good OCR accuracy (Tesseract path)
MIN_WIDTH = 600
MIN_HEIGHT = 100

# Minimum dimensions for EasyOCR (from RSTGameTranslation)
EASYOCR_MIN_WIDTH = 1024
EASYOCR_MIN_HEIGHT = 768

# Debug: save captures to .tests/debug/ for inspection
DEBUG_SAVE = os.environ.get("CRR_DEBUG", "0") == "1"
_DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".tests", "debug")


# --- OCR engine abstraction ---

class OcrEngine(Protocol):
    """Protocol for OCR engines."""
    def recognize(self, image: Image.Image, language: str) -> str: ...


class TesseractEngine:
    """Tesseract OCR engine (default, lightweight)."""

    def recognize(self, image: Image.Image, language: str) -> str:
        return pytesseract.image_to_string(
            image, lang=language, config="--psm 6 --oem 1"
        ).strip()


class EasyOcrEngine:
    """EasyOCR engine with GPU support (optional, requires easyocr package).

    EasyOCR has its own text detection model so it works best with
    the original (non-binarized) image.  We apply HDR-adaptive preprocessing
    (CLAHE + bilateral + sharpening) but skip the text_isolator pipeline.
    """

    # EasyOCR works on raw images, not binarized text_isolator output
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
            print(f"[EasyOCR] Initializing with langs={langs}, gpu={gpu}")
            self._reader = easyocr.Reader(langs, gpu=gpu)
            self._current_langs = langs

        img_array = np.array(image)
        results = self._reader.readtext(img_array, detail=0)
        return "\n".join(results).strip()


def create_engine(name: str) -> OcrEngine:
    """Create an OCR engine by name. Raises ImportError for unavailable engines."""
    if name == "easyocr":
        # Verify easyocr is importable before creating the engine
        try:
            import easyocr  # noqa: F401
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. Install with: "
                "uv pip install easyocr torch"
            )
        return EasyOcrEngine()
    return TesseractEngine()


# --- Helper functions ---

def _extract_real_words(text: str) -> list[str]:
    """Extract plausible real words from OCR text, ignoring garbage."""
    words = []
    for w in text.split():
        if len(w) < 3:
            continue
        alpha = sum(1 for c in w if c.isalpha())
        if alpha / len(w) >= 0.7:
            words.append(w.lower())
    return words


def _strip_ocr_artifacts(text: str) -> str:
    """Remove common OCR cursor/boundary artifacts for comparison."""
    text = re.sub(r"[\|\]\[\)\\\}]+\s*$", "", text)
    text = re.sub(r"[\|\]\[]", "", text)
    return text.strip()


def _texts_similar(
    new_text: str, last_emitted: str, threshold: float = 0.75
) -> bool:
    """Check if *new_text* is similar enough to *last_emitted* to skip."""
    if not new_text or not last_emitted:
        return False
    norm_new = " ".join(new_text.split())
    norm_old = " ".join(last_emitted.split())
    if norm_new == norm_old:
        return True

    clean_new = _strip_ocr_artifacts(norm_new)
    clean_old = _strip_ocr_artifacts(norm_old)

    if clean_new in clean_old:
        return True
    if clean_old in clean_new:
        return False

    words_new = _extract_real_words(new_text)
    words_old = _extract_real_words(last_emitted)
    if len(words_new) >= 2 and len(words_old) >= 2:
        set_new = set(words_new)
        set_old = set(words_old)
        overlap = len(set_new & set_old)
        smaller = min(len(set_new), len(set_old))

        if overlap / smaller >= 0.6:
            if len(set_new) > len(set_old):
                return False
            return True

    if is_text_similar(norm_old, norm_new, threshold):
        return True

    return False


def _filter_ocr_garbage(text: str) -> str:
    """Filter out OCR garbage lines."""
    lines = text.split("\n")
    good_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        alnum = sum(1 for c in line if c.isalnum())
        non_space = len(line.replace(" ", ""))
        if non_space == 0 or alnum / non_space < 0.4:
            continue

        alpha = sum(1 for c in line if c.isalpha())
        if alpha < 2:
            continue

        words = line.split()
        garbage_words = 0
        for word in words:
            cyr = sum(1 for c in word if "\u0400" <= c <= "\u04FF")
            lat = sum(1 for c in word if ("A" <= c <= "Z") or ("a" <= c <= "z"))
            word_alpha = cyr + lat
            if cyr > 0 and lat > 0 and word_alpha >= 3:
                garbage_words += 1

        if len(words) > 0 and garbage_words / len(words) > 0.3:
            continue

        if len(words) >= 3:
            short_words = sum(1 for w in words if len(w) <= 2)
            if short_words / len(words) > 0.6:
                continue

        alpha_chars = [c.lower() for c in line if c.isalpha()]
        if len(alpha_chars) >= 4:
            unique_ratio = len(set(alpha_chars)) / len(alpha_chars)
            if unique_ratio < 0.15:
                continue

        good_lines.append(line)

    result = "\n".join(good_lines)

    if result:
        all_words = result.split()
        if len(all_words) <= 3:
            avg_word_len = sum(len(w) for w in all_words) / len(all_words)
            if avg_word_len < 3:
                return ""

    return result


def _upscale(img: Image.Image) -> Image.Image:
    """Upscale image to minimum dimensions needed for Tesseract OCR accuracy."""
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

    Pipeline (from RST's process_image_easyocr.py):
    1. HDR-adaptive enhancement (CLAHE + bilateral + sharpen) on grayscale
       OR basic contrast + median filter
    2. Convert back to RGB (EasyOCR accepts both but RGB is standard)
    3. Upscale to minimum 1024×768 with LANCZOS

    No text isolation or contour detection — EasyOCR has its own
    CRAFT text detector that works best on natural/enhanced images.
    """
    import cv2
    from PIL import ImageEnhance, ImageFilter

    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))

    # Auto-detect when enhanced processing is needed (same thresholds as RST)
    needs_enhanced = std_br < 40 or mean_br > 200 or mean_br < 55

    if needs_enhanced:
        # Enhanced mode: CLAHE + bilateral + sharpen (RST parameters)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel=kernel)
        # Convert gray back to RGB for EasyOCR
        result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(result_rgb)
    else:
        # Basic mode: contrast boost + median filter (RST parameters)
        img = Image.fromarray(raw_rgb).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.convert("RGB")

    # Upscale to EasyOCR minimum dimensions (RST uses 1024×768)
    w, h = img.size
    if w < EASYOCR_MIN_WIDTH or h < EASYOCR_MIN_HEIGHT:
        scale = max(EASYOCR_MIN_WIDTH / w, EASYOCR_MIN_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


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
        self._last_emitted_text: str = ""
        self._growth_detected: bool = False
        self._stable_emits: int = 0
        self._engine: OcrEngine | None = None
        self._engine_name: str = ""  # empty so first set_engine() always triggers

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
        self._growth_detected = False
        self._stable_emits = 0
        print(f"[OCR] Configured region: left={region[0]}, top={region[1]}, "
              f"width={region[2]}, height={region[3]}, lang={language}, "
              f"engine={self._engine_name}")

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def set_engine(self, engine_name: str) -> None:
        """Switch OCR engine. Safe to call while running."""
        print(f"[OCR] set_engine called: '{engine_name}' (current: '{self._engine_name}')")
        if engine_name == self._engine_name:
            print(f"[OCR] set_engine: same name, skipping")
            return
        try:
            old_name = self._engine_name
            self._engine = create_engine(engine_name)
            self._engine_name = engine_name
            print(f"[OCR] Switched engine: {old_name} -> {engine_name} (now: {type(self._engine).__name__})")
        except ImportError as e:
            print(f"[OCR] Failed to switch to {engine_name}: {e}")
            self.error_occurred.emit(str(e))
            # Fallback to Tesseract so OCR keeps working
            if self._engine is None:
                self._engine = TesseractEngine()
                self._engine_name = "tesseract"
                print("[OCR] Fallback: using Tesseract")

    def run(self) -> None:
        self._running = True
        if self._engine is None:
            self._engine = TesseractEngine()
            self._engine_name = "tesseract"
            print("[OCR] Warning: engine was None at run(), defaulting to Tesseract")
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

                    # BGRA -> RGB
                    raw_rgb = img_array[:, :, :3][:, :, ::-1].copy()

                    # Choose preprocessing path based on engine
                    use_isolation = getattr(self._engine, "needs_text_isolation", True)

                    if use_isolation:
                        # Tesseract path: text_isolator creates clean
                        # black-on-white image optimized for Tesseract
                        isolated = isolate_text(raw_rgb)

                        if isolated is None:
                            p_h, p_w = raw_rgb.shape[:2]
                            self.frame_captured.emit(raw_rgb.tobytes(), p_w, p_h)
                            if self._growth_detected:
                                self.text_recognized.emit("")
                                self._growth_detected = False
                                self._stable_emits = 0
                            self.msleep(self._interval_ms)
                            continue

                        ocr_img = _upscale(Image.fromarray(isolated))
                    else:
                        # EasyOCR path (matches RSTGameTranslation approach):
                        # HDR-adaptive preprocessing on raw image, then
                        # upscale to min 1024x768.  EasyOCR has its own
                        # CRAFT text detector — no text_isolator needed.
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
                        print(f"[OCR] Saved debug captures #{self._capture_count}")

                    self._capture_count += 1

                    # Run OCR via the configured engine
                    t0 = time.monotonic()
                    text = self._engine.recognize(ocr_img, self._language)
                    ocr_ms = int((time.monotonic() - t0) * 1000)

                    print(f"[OCR] ({ocr_ms}ms) {self._engine_name} raw: "
                          f"{repr(text[:200] if text else '')}")

                    # Filter out garbage and deduplicate
                    if text:
                        filtered = _filter_ocr_garbage(text)
                        if filtered != text:
                            print(f"[OCR] After filter: "
                                  f"{repr(filtered[:200] if filtered else '<empty>')}")
                        text = filtered
                        if text:
                            similar = _texts_similar(text, self._last_emitted_text)
                            if similar and not self._growth_detected:
                                print(f"[OCR] Dedup: similar to last")
                            elif similar and self._growth_detected:
                                self._stable_emits += 1
                                print(f"[OCR] Emit for stability ({self._stable_emits}/3)")
                                self.text_recognized.emit(text)
                                if self._stable_emits >= 2:
                                    self._growth_detected = False
                                    self._stable_emits = 0
                            else:
                                clean_new = _strip_ocr_artifacts(text)
                                clean_old = _strip_ocr_artifacts(
                                    self._last_emitted_text
                                )
                                is_growth = (
                                    clean_old
                                    and clean_old in clean_new
                                    and len(clean_new) > len(clean_old) + 3
                                )
                                if not is_growth:
                                    words_old = _extract_real_words(
                                        self._last_emitted_text
                                    )
                                    words_new = _extract_real_words(text)
                                    if (
                                        len(words_old) >= 1
                                        and len(words_new) > len(words_old)
                                    ):
                                        set_old = set(words_old)
                                        set_new = set(words_new)
                                        overlap = len(set_old & set_new)
                                        if (
                                            len(set_old) > 0
                                            and overlap / len(set_old) >= 0.6
                                        ):
                                            is_growth = True
                                if is_growth:
                                    self._growth_detected = True
                                    self._stable_emits = 0
                                else:
                                    self._growth_detected = False
                                    self._stable_emits = 0
                                self._last_emitted_text = text
                                self.text_recognized.emit(text)
                except Exception as e:
                    self.error_occurred.emit(str(e))

                self.msleep(self._interval_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
