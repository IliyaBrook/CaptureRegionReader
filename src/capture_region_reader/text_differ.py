from __future__ import annotations

from difflib import SequenceMatcher


class TextDiffer:
    """Detects new/changed text to avoid repeating already-spoken content.

    The key challenge: OCR runs every ~500ms and may produce:
    1. Identical text (same subtitle still on screen) → must NOT re-speak
    2. Slightly different text (OCR jitter: extra space, punctuation change) → must NOT re-speak
    3. Genuinely new text (subtitle changed) → MUST speak
    4. Growing text (scrolling/streaming subtitles) → speak only the NEW portion
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self._last_text: str = ""
        self._threshold = similarity_threshold

    def get_new_text(self, current_text: str) -> str | None:
        """Return text to speak, or None if no meaningful change detected."""
        if not current_text:
            return None

        if not self._last_text:
            self._last_text = current_text
            return current_text

        # Normalize for comparison (collapse whitespace)
        norm_old = " ".join(self._last_text.split())
        norm_new = " ".join(current_text.split())

        # Exact match after normalization — definitely skip
        if norm_old == norm_new:
            return None

        ratio = SequenceMatcher(None, norm_old, norm_new).ratio()

        # High similarity = OCR jitter, not a real change
        if ratio >= self._threshold:
            return None

        # Check if text grew (scrolling/streaming subtitles)
        old_lines = self._last_text.strip().splitlines()
        new_lines = current_text.strip().splitlines()

        if len(new_lines) > len(old_lines):
            # Find where old text ends in new text
            overlap = 0
            for i, old_line in enumerate(old_lines):
                if i < len(new_lines):
                    line_ratio = SequenceMatcher(
                        None, old_line.strip(), new_lines[i].strip()
                    ).ratio()
                    if line_ratio > 0.7:
                        overlap = i + 1
                    else:
                        break

            if overlap > 0:
                new_portion = "\n".join(new_lines[overlap:])
                self._last_text = current_text
                return new_portion.strip() if new_portion.strip() else None

        # Text changed substantially — return new text
        self._last_text = current_text
        return current_text

    def reset(self) -> None:
        self._last_text = ""
