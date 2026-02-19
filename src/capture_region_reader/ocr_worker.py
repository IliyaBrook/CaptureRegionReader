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


def _strip_ocr_artifacts(text: str) -> str:
    """Remove common OCR cursor/boundary artifacts for comparison."""
    import re
    # Remove trailing artifacts: |, ], [, ), \, trailing punctuation junk
    text = re.sub(r"[\|\]\[\)\\\}]+\s*$", "", text)
    # Also remove mid-text stray | ] [ chars
    text = re.sub(r"[\|\]\[]", "", text)
    return text.strip()


def _texts_similar(
    new_text: str, last_emitted: str, threshold: float = 0.75
) -> bool:
    """Check if *new_text* is similar enough to *last_emitted* to skip.

    Returns True  → "same subtitle, don't emit again"
    Returns False → "different or grew, let it through"

    Important asymmetry for typewriter-style subtitles:
      - last_emitted is a prefix of new_text → text GREW → False
      - new_text is a subset of last_emitted → partial OCR → True (skip)
    """
    if not new_text or not last_emitted:
        return False
    norm_new = " ".join(new_text.split())
    norm_old = " ".join(last_emitted.split())
    if norm_new == norm_old:
        return True

    # Strip OCR artifacts (|, ], etc.) for containment checks —
    # typewriter subtitles have a blinking cursor that OCR reads as |/]/etc.
    clean_new = _strip_ocr_artifacts(norm_new)
    clean_old = _strip_ocr_artifacts(norm_old)

    # new is a subset of old → partial OCR of same subtitle → skip
    if clean_new in clean_old:
        return True

    # old is a subset of new → text GREW (typewriter subtitles) → let through
    if clean_old in clean_new:
        return False

    # Word-level growth detection: handles OCR jitter within words
    # (e.g., "ВВЕАКЕН" vs "ВНЕАКЕН") while still detecting growth.
    words_new = _extract_real_words(new_text)
    words_old = _extract_real_words(last_emitted)
    if len(words_new) >= 2 and len(words_old) >= 2:
        set_new = set(words_new)
        set_old = set(words_old)
        overlap = len(set_new & set_old)
        smaller = min(len(set_new), len(set_old))

        if overlap / smaller >= 0.6:
            # Most old words are in new — check if new has MORE words
            if len(set_new) > len(set_old):
                # Text grew — let through for TextDiffer to handle
                return False
            return True

    # Full-text similarity (catch OCR jitter with same word count)
    if SequenceMatcher(None, norm_old, norm_new).ratio() >= threshold:
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
        self._growth_detected: bool = False  # True when growth was recently seen
        self._stable_emits: int = 0          # count of stable emits after growth

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
                        # If growth was recently detected, notify TextDiffer
                        # that text disappeared so it can flush pending buffer
                        if self._growth_detected:
                            self.text_recognized.emit("")
                            self._growth_detected = False
                            self._stable_emits = 0
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
                            similar = _texts_similar(text, self._last_emitted_text)
                            if similar and not self._growth_detected:
                                # Same subtitle, no pending growth — skip
                                print(f"[OCR] Dedup: similar to last")
                            elif similar and self._growth_detected:
                                # Text stabilized after growth — emit for
                                # TextDiffer stability counting, then stop
                                self._stable_emits += 1
                                print(f"[OCR] Emit for stability ({self._stable_emits}/3)")
                                self.text_recognized.emit(text)
                                if self._stable_emits >= 3:
                                    # Enough stable emits, TextDiffer
                                    # should have flushed by now
                                    self._growth_detected = False
                                    self._stable_emits = 0
                            else:
                                # Different text — detect growth
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
                                    # Also check word-level growth
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
                                    # Completely new subtitle
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
