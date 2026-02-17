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
        words = line.split()
        garbage_words = 0
        for word in words:
            cyr = sum(1 for c in word if "\u0400" <= c <= "\u04FF")
            lat = sum(1 for c in word if ("A" <= c <= "Z") or ("a" <= c <= "z"))
            word_alpha = cyr + lat
            # A word with both Cyrillic AND Latin is likely OCR garbage
            if cyr > 0 and lat > 0 and word_alpha >= 3:
                garbage_words += 1

        # If more than 30% of words are garbage, skip the line
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
              f"width={region[2]}, height={region[3]}, lang={language}")

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def _preprocess(self, rgb: np.ndarray) -> Image.Image:
        """Convert RGB array to optimized image for OCR.

        Following the Frog approach: minimal preprocessing.
        Tesseract's own internal preprocessing (Leptonica) handles
        binarization better than we can, especially for complex backgrounds.
        Aggressive binarization turns background logos/text into black text
        that Tesseract then tries to read as real content.

        We only upscale small images (Tesseract needs ~300 DPI equivalent)
        and do light contrast enhancement.
        """
        img = Image.fromarray(rgb)
        w, h = img.size

        # Upscale small images to ensure minimum dimensions for Tesseract
        scale = 1
        if w < MIN_WIDTH:
            scale = max(scale, (MIN_WIDTH + w - 1) // w)
        if h < MIN_HEIGHT:
            scale = max(scale, (MIN_HEIGHT + h - 1) // h)
        # At least 2x for better OCR accuracy on screen captures
        scale = max(scale, 2)
        scale = min(scale, 4)

        if scale > 1:
            img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

        # Light contrast boost — helps Tesseract without destroying color info
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

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

                    # Debug saves
                    if DEBUG_SAVE and self._capture_count < 5:
                        Image.fromarray(raw_rgb).save(
                            f"/tmp/crr_raw_{self._capture_count}.png"
                        )

                    processed = self._preprocess(raw_rgb)

                    if DEBUG_SAVE and self._capture_count < 5:
                        processed.save(
                            f"/tmp/crr_processed_{self._capture_count}.png"
                        )
                        print(
                            f"[OCR] Saved debug captures #{self._capture_count}"
                        )

                    self._capture_count += 1

                    # PSM 3 = fully automatic page segmentation
                    # OEM 1 = legacy Tesseract engine (consistent results)
                    text = pytesseract.image_to_string(
                        processed,
                        lang=self._language,
                        config="--psm 3 --oem 1",
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
