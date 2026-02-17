from __future__ import annotations

import numpy as np
import pytesseract
from mss import mss
from PIL import Image, ImageEnhance
from PyQt6.QtCore import QThread, pyqtSignal


# Minimum dimensions for good Tesseract accuracy
MIN_WIDTH = 600
MIN_HEIGHT = 100


class OcrWorker(QThread):
    text_recognized = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._region: tuple[int, int, int, int] | None = None
        self._language: str = "eng+rus"
        self._interval_ms: int = 500
        self._running: bool = False

    def configure(
        self,
        region: tuple[int, int, int, int],
        language: str,
        interval_ms: int,
    ) -> None:
        self._region = region
        self._language = language
        self._interval_ms = interval_ms

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def _preprocess(self, img_array: np.ndarray) -> Image.Image:
        """Convert BGRA screenshot to optimized image for OCR.

        Key insights for Tesseract accuracy:
        - Color images work better than grayscale for screen content
        - Upscaling small text dramatically improves recognition
        - Contrast enhancement helps with low-contrast backgrounds
        - Do NOT binarize — it destroys anti-aliased text
        """
        # BGRA -> RGB
        rgb = img_array[:, :, :3][:, :, ::-1].copy()
        img = Image.fromarray(rgb)
        w, h = img.size

        # Always upscale to ensure minimum dimensions for Tesseract
        scale = 1
        if w < MIN_WIDTH:
            scale = max(scale, (MIN_WIDTH + w - 1) // w)
        if h < MIN_HEIGHT:
            scale = max(scale, (MIN_HEIGHT + h - 1) // h)
        # Always at least 2x for better accuracy on screen text
        scale = max(scale, 2)
        # Cap at 4x to avoid excessive processing time
        scale = min(scale, 4)

        if scale > 1:
            img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)

        # Boost contrast (helps with semi-transparent overlays, game UIs)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        # Slight sharpening for cleaner text edges
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)

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
                    processed = self._preprocess(img_array)

                    text = pytesseract.image_to_string(
                        processed,
                        lang=self._language,
                        config="--psm 3",
                    ).strip()

                    # Filter out garbage: ignore results that are mostly
                    # non-alphanumeric (common OCR artifacts)
                    if text:
                        alnum = sum(1 for c in text if c.isalnum())
                        total = len(text.replace(" ", "").replace("\n", ""))
                        if total > 0 and alnum / total > 0.3:
                            self.text_recognized.emit(text)
                except Exception as e:
                    self.error_occurred.emit(str(e))

                self.msleep(self._interval_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
