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
        print(f"[OCR] Configured region: left={region[0]}, top={region[1]}, "
              f"width={region[2]}, height={region[3]}, lang={language}")

    def set_language(self, language: str) -> None:
        self._language = language

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = interval_ms

    def _preprocess(self, img_array: np.ndarray) -> Image.Image:
        """Convert BGRA screenshot to optimized image for OCR.

        Uses multiple strategies optimized for subtitle recognition:
        bright text on complex/dark backgrounds.
        """
        # BGRA -> RGB
        rgb = img_array[:, :, :3][:, :, ::-1].copy()
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

        # Convert to grayscale for better subtitle extraction
        gray = img.convert("L")

        # Apply adaptive-like binarization for subtitle text:
        # Subtitles are typically bright (white/yellow) text on darker background.
        # Use a high threshold to isolate bright text.
        gray_arr = np.array(gray, dtype=np.float32)

        # Method: extract bright regions (subtitle text is usually >180 brightness)
        # Then create black text on white background (best for Tesseract)
        threshold = 180

        # Check if image has bright text (subtitle-like) or dark text
        bright_pixels = np.sum(gray_arr > threshold)
        dark_pixels = np.sum(gray_arr < 80)
        total_pixels = gray_arr.size

        if bright_pixels / total_pixels > 0.02 and bright_pixels < total_pixels * 0.5:
            # Likely bright text on dark/complex background (subtitles)
            # Isolate bright text: bright → black text, rest → white background
            binary = np.where(gray_arr > threshold, 0, 255).astype(np.uint8)
            img = Image.fromarray(binary, mode="L")
        elif dark_pixels / total_pixels > 0.02 and dark_pixels < total_pixels * 0.5:
            # Dark text on light background (normal text)
            binary = np.where(gray_arr < 80, 0, 255).astype(np.uint8)
            img = Image.fromarray(binary, mode="L")
        else:
            # Mixed/unclear — use enhanced grayscale
            enhancer = ImageEnhance.Contrast(gray)
            img = enhancer.enhance(2.0)

        # Light sharpening to clean up edges
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

                    # Send raw frame for preview (BGRA → RGB)
                    raw_rgb = img_array[:, :, :3][:, :, ::-1].copy()
                    h_px, w_px = raw_rgb.shape[:2]
                    self.frame_captured.emit(raw_rgb.tobytes(), w_px, h_px)

                    # Debug: save raw + processed screenshots
                    if DEBUG_SAVE and self._capture_count < 5:
                        Image.fromarray(raw_rgb).save(f"/tmp/crr_raw_{self._capture_count}.png")

                    processed = self._preprocess(img_array)

                    if DEBUG_SAVE and self._capture_count < 5:
                        processed.save(f"/tmp/crr_processed_{self._capture_count}.png")
                        print(f"[OCR] Saved debug captures #{self._capture_count} to /tmp/")

                    self._capture_count += 1

                    # Use --psm 6 for uniform text blocks (subtitles)
                    # --oem 3 = default LSTM + legacy
                    text = pytesseract.image_to_string(
                        processed,
                        lang=self._language,
                        config="--psm 6 --oem 1",
                    ).strip()

                    # Filter out garbage
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
