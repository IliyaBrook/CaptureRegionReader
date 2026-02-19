from __future__ import annotations

import os
import numpy as np
import pytesseract
from difflib import SequenceMatcher
from mss import mss
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from capture_region_reader.text_isolator import isolate_text


# Minimum dimensions for good Tesseract accuracy
MIN_WIDTH = 600
MIN_HEIGHT = 100

# Debug: save captures to .tests/debug/ for inspection
DEBUG_SAVE = os.environ.get("CRR_DEBUG", "0") == "1"
_DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".tests", "debug")


def _extract_real_words(text: str) -> list[str]:
    """Extract plausible real words from OCR text, ignoring garbage.

    A "real word" is ≥3 characters and predominantly alphabetic
    (Cyrillic or Latin).  This filters out noise tokens like
    'ФМ/ЮРГОМЕМС', '01зпеу', '|', 'Р\\м$' that come from
    background elements being OCR'd.
    """
    words = []
    for w in text.split():
        if len(w) < 3:
            continue
        alpha = sum(1 for c in w if c.isalpha())
        if alpha / len(w) >= 0.7:
            words.append(w.lower())
    return words


def _texts_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """Check if two OCR texts are similar enough to be the same subtitle.

    Uses three strategies, from fast to thorough:
    1. Normalized full-text exact/containment match
    2. SequenceMatcher on normalized text (catches OCR jitter)
    3. Real-word overlap — compares only plausible dictionary words,
       ignoring garbage tokens from background noise.  This is the key
       defence against repeated reads when the background behind
       subtitles changes and adds/removes noise around the same text.
    """
    if not a or not b:
        return False
    norm_a = " ".join(a.split())
    norm_b = " ".join(b.split())
    if norm_a == norm_b:
        return True
    # Containment check (partial OCR of same subtitle)
    if norm_a in norm_b or norm_b in norm_a:
        return True
    # Full-text similarity
    if SequenceMatcher(None, norm_a, norm_b).ratio() >= threshold:
        return True

    # Real-word overlap: if most real words are shared, it's the
    # same subtitle with different background garbage.
    words_a = _extract_real_words(a)
    words_b = _extract_real_words(b)
    if len(words_a) >= 2 and len(words_b) >= 2:
        set_a = set(words_a)
        set_b = set(words_b)
        overlap = len(set_a & set_b)
        # What fraction of the SMALLER set is covered?
        smaller = min(len(set_a), len(set_b))
        if overlap / smaller >= 0.6:
            return True

    return False


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

        # Reject lines dominated by short words (1-2 chars).
        # Garbage like "ООО ООО О Ооо бо И О" is mostly tiny words.
        if len(words) >= 3:
            short_words = sum(1 for w in words if len(w) <= 2)
            if short_words / len(words) > 0.6:
                continue

        # Reject lines with very low character diversity.
        # "ООО ООО О Ооо" has few unique chars relative to length.
        alpha_chars = [c.lower() for c in line if c.isalpha()]
        if len(alpha_chars) >= 4:
            unique_ratio = len(set(alpha_chars)) / len(alpha_chars)
            if unique_ratio < 0.15:
                continue

        good_lines.append(line)

    result = "\n".join(good_lines)

    # Final: reject very short results with tiny average word length.
    if result:
        all_words = result.split()
        if len(all_words) <= 3:
            avg_word_len = sum(len(w) for w in all_words) / len(all_words)
            if avg_word_len < 3:
                return ""

    return result


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

                    if isolated is None:
                        # No text detected — show raw frame in preview, skip OCR
                        p_h, p_w = raw_rgb.shape[:2]
                        self.frame_captured.emit(raw_rgb.tobytes(), p_w, p_h)
                        self.msleep(self._interval_ms)
                        continue

                    ocr_img = _upscale(Image.fromarray(isolated))

                    # Send the processed image to preview so user can debug
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

                    print(f"[OCR] Tesseract raw: {repr(text[:200] if text else '')}")

                    # Filter out garbage and deduplicate
                    if text:
                        filtered = _filter_ocr_garbage(text)
                        if filtered != text:
                            print(f"[OCR] After filter: {repr(filtered[:200] if filtered else '<empty>')}")
                        text = filtered
                        if text:
                            if _texts_similar(text, self._last_emitted_text):
                                print(f"[OCR] Dedup: similar to last")
                            else:
                                self._last_emitted_text = text
                                self.text_recognized.emit(text)
                except Exception as e:
                    self.error_occurred.emit(str(e))

                self.msleep(self._interval_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
