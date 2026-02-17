from __future__ import annotations

import os
import numpy as np
import pytesseract
from mss import mss
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from capture_region_reader.text_isolator import isolate_text


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
            if cyr > 0 and lat > 0 and word_alpha >= 3:
                garbage_words += 1

        if len(words) > 0 and garbage_words / len(words) > 0.3:
            continue

        good_lines.append(line)

    return "\n".join(good_lines)


def _upscale(img: Image.Image) -> Image.Image:
    """Upscale image to minimum dimensions needed for Tesseract accuracy."""
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

                    # Text isolation: detect text contours, crop, clean bg
                    isolated = isolate_text(raw_rgb)

                    if isolated is not None:
                        ocr_img = _upscale(Image.fromarray(isolated))
                    else:
                        # Fallback: send raw image (upscaled) if isolation fails
                        ocr_img = _upscale(Image.fromarray(raw_rgb))

                    # Send the processed image to preview so user can debug
                    preview_rgb = np.array(ocr_img)
                    p_h, p_w = preview_rgb.shape[:2]
                    self.frame_captured.emit(preview_rgb.tobytes(), p_w, p_h)

                    # Debug saves
                    if DEBUG_SAVE and self._capture_count < 5:
                        Image.fromarray(raw_rgb).save(
                            f"/tmp/crr_raw_{self._capture_count}.png"
                        )
                        ocr_img.save(
                            f"/tmp/crr_isolated_{self._capture_count}.png"
                        )
                        print(
                            f"[OCR] Saved debug captures #{self._capture_count}"
                        )

                    self._capture_count += 1

                    # PSM 6 = uniform text block (isolated text is a clean block)
                    # OEM 1 = legacy engine (consistent)
                    text = pytesseract.image_to_string(
                        ocr_img,
                        lang=self._language,
                        config="--psm 6 --oem 1",
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
